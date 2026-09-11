import argparse
import multiprocessing as mp
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime

import cloudpickle
import numpy as np
import submitit
import torch
from loguru import logger
from rdkit import RDLogger, rdBase
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from molgen3D.config.paths import (
    get_base_path,
    get_ckpt,
    get_data_path,
    get_tokenizer_path,
)
from molgen3D.config.sampling_config import gen_num_codes, sampling_configs
from molgen3D.data_processing.smiles_encoder_decoder import (
    decode_conformer_by_serialization,
    load_bin_config_for_mode,
    strip_smiles,
)
from molgen3D.data_processing.smiles_encoder_decoder_fsq import (
    MolFSQModel,
    _resolve_fsq_ckpt_path,
)
from molgen3D.evaluation.utils import (
    detect_peak_flops,
    estimate_decoder_flops_per_token,
    extract_between,
    log_cuda_memory,
    log_mfu,
    same_molecular_graph,
)

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")
rdBase.DisableLog("rdApp.error")


def set_seed(seed=42):
    random.seed(seed)  # Python random module
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # All GPUs (if using multi-GPU)

    # Ensure deterministic behavior. NB: touching torch.backends.cudnn makes this
    # function unpicklable *by value* (CudnnModule), so anything submitit submits
    # must reach it by reference -- see the import in the __main__ block below.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_tokenizer(
    model_path,
    tokenizer_path,
    torch_dtype="bfloat16",
    attention_imp="sdpa",
    device="auto",
):
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), padding_side="left", local_files_only=True
    )
    dtype_obj = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype_obj,
        attn_implementation=attention_imp,
        device_map=device,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    model._flops_per_token = estimate_decoder_flops_per_token(model.config)
    model._peak_device_flops = detect_peak_flops(model.device)

    log_cuda_memory("Post-load")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = torch.compile(model, mode="reduce-overhead")
    logger.info(
        f"torch.compile succeeded; using optimized graph. Compiled type={type(model)}"
    )
    log_cuda_memory("Post-compile")

    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    print(f"{model.dtype=}, {model.device=}")

    return model, tokenizer


def save_results(results_path, generations, stats):
    with open(os.path.join(results_path, "generation_results.pickle"), 'wb') as results_file_pickle:
        cloudpickle.dump(generations, results_file_pickle, protocol=4)

    with open(os.path.join(results_path, "generation_results.txt"), 'w') as results_file_txt:
        results_file_txt.write(f"{stats=}")


def process_batch(
    model,
    tokenizer,
    batch: list[list],
    gen_config,
    eos_token_id,
    binned: bool,
    serialization_tag: str = "cartesian",
    uniform_bin_config_path: str = None,
    quantile_bin_config_path: str = None,
    uniform_bin_config=None,
    quantile_bin_config=None,
    fsq_decoder=None,
):
    bins = None
    if binned and serialization_tag == "cartesian_binned":
        ranges = [(-13.0, 13.0), (-13.0, 13.0), (-13.0, 13.0)]
        bin_size = 0.104
        bins = [np.arange(start, end, bin_size) for start, end in ranges]
    generations = defaultdict(list)
    stats = {"smiles_mismatch":0, "mol_parse_fail" :0, "no_eos":0}

    # Extract prompts and geom_smiles from batch
    prompts = [item[1] for item in batch]
    geom_smiles_list = [item[0] for item in batch]

    tokenized_prompts = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        pad_to_multiple_of=8,
    )
    tokenized_prompts = {k: v.to(model.device, non_blocking=True) for k, v in tokenized_prompts.items()}
    tokenized_prompts["attention_mask"] = tokenized_prompts["attention_mask"].contiguous()
    start_time = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=tokenized_prompts["input_ids"],
            attention_mask=tokenized_prompts["attention_mask"],
            max_new_tokens=1000,
            eos_token_id=eos_token_id,
            generation_config=gen_config,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        sequences = outputs.sequences.detach().cpu()
        del outputs
    elapsed = time.perf_counter() - start_time
    prompt_lens = tokenized_prompts["attention_mask"].sum(dim=1).cpu()
    seq_pad_mask = (sequences != tokenizer.pad_token_id).to(torch.int32)
    seq_lens = seq_pad_mask.sum(dim=1)
    gen_lens = (seq_lens - prompt_lens).clamp(min=0)
    total_generated_tokens = int(gen_lens.sum().item())
    log_mfu(model, total_generated_tokens, elapsed)
    log_cuda_memory("Post-first-forward")
    decoded_outputs = tokenizer.batch_decode(sequences, skip_special_tokens=False)
    for i, out in enumerate(decoded_outputs):
        canonical_smiles = extract_between(out, "[SMILES]", "[/SMILES]")
        generated_conformer = extract_between(out, "[CONFORMER]", "[/CONFORMER]")
        geom_smiles = geom_smiles_list[i]

        if generated_conformer:
            generated_smiles = strip_smiles(generated_conformer)
            if not same_molecular_graph(canonical_smiles, generated_smiles):
                logger.info(f"smiles mismatch: \n{canonical_smiles=}\n{generated_smiles=}\n{generated_conformer=}")
                stats["smiles_mismatch"] += 1
            else:
                try:
                    if serialization_tag == "fsq":
                        recon_enriched = fsq_decoder.decode_text(generated_conformer)
                        mol_obj = decode_conformer_by_serialization(recon_enriched, "cartesian")
                    else:
                        mol_obj = decode_conformer_by_serialization(
                            generated_conformer,
                            serialization_tag,
                            bins=bins,
                            uniform_config_path=uniform_bin_config_path,
                            quantile_config_path=quantile_bin_config_path,
                            uniform_config=uniform_bin_config,
                            quantile_config=quantile_bin_config,
                        )
                    generations[geom_smiles].append(mol_obj)
                except Exception:
                    logger.info(f"smiles fails parsing: \n{canonical_smiles=}\n{generated_smiles=}\n{generated_conformer=}")
                    stats["mol_parse_fail"] += 1
        else:
            stats["no_eos"] += 1
            logger.info(f"no eos: \n{out[:1000]=}")
    return generations, stats


def resolve_repeat_count(num_gens, count: int = 1) -> int:
    if isinstance(num_gens, str):
        s = num_gens.strip().lower()
        if s.endswith("k"):
            multiplier = int(s[:-1]) if s[:-1] else 1
            return max(1, multiplier * count)
        return int(s)
    return int(num_gens)


def split_batch_on_geom_size(batch: list[list], max_geom_len: int = 80) -> list[list]:
    if not batch:
        return []
    if len(batch) == 1:
        return [batch]
    if any(len(geom_smiles) > max_geom_len for geom_smiles, _ in batch):
        mid = len(batch) // 2
        if mid:
            return [batch[:mid], batch[mid:]]
    return [batch]


# Test sets whose pickles share the {geom_smiles: {"sub_smiles_counts": {...}}} layout.
SUB_SMILES_COUNT_TEST_SETS = (
    "distinct",
    "xl",
    "qm9",
    "valid",
    "revisited",
    "casf16",
    "casf16_opt",
    "casf16_ref_chembl",
    "casf16_core_chembl",
    "rnp",
    "druglike",
)


def build_mols_list(inference_config: dict, test_data: dict) -> list:
    mols_list = []
    num_gens = inference_config.get("num_gens", 2)
    test_set: str = inference_config.get("test_set", "distinct")
    logger.info(f"Processing as {test_set} dataset")
    if test_set == "clean":
        for geom_smiles, data in test_data.items():
            prompt = f"[SMILES]{data['corrected_smi']}[/SMILES]"
            mols_list.extend([(geom_smiles, prompt)] * resolve_repeat_count(num_gens))
    elif test_set == "icl":
        for geom_smiles, data in test_data.items():
            icl_prompt = data.get("icl_prompt")
            if icl_prompt:
                mols_list.extend([(geom_smiles, icl_prompt)] * resolve_repeat_count(num_gens))
    elif test_set in SUB_SMILES_COUNT_TEST_SETS:
        for geom_smiles, data in test_data.items():
            for sub_smiles, count in data["sub_smiles_counts"].items():
                prompt = f"[SMILES]{sub_smiles}[/SMILES]"
                mols_list.extend([(geom_smiles, prompt)] * resolve_repeat_count(num_gens, count))
    else:
        logger.warning(f"Unknown test_set '{test_set}'; no prompts built.")
    logger.info(f"mols_list length: {len(mols_list)}, mols_list_distinct: {len(set(mols_list))}, mols_list: {mols_list[:10]}")

    mols_list.sort(key=lambda x: len(x[0]))

    limit = inference_config.get("limit")
    return mols_list[:limit]


def load_test_data(inference_config: dict) -> dict:
    with open(inference_config["test_data_path"], "rb") as test_data_file:
        return cloudpickle.load(test_data_file)


def resolve_serialization(inference_config: dict) -> tuple[bool, str]:
    """Resolve the binned / serialization flags once, so every shard agrees on them."""
    binned = inference_config.get("binned", False)
    if not binned and "binned" in str(inference_config["model_path"]):
        logger.info("Auto-detecting binned=True based on model path")
        binned = True

    serialization_tag = inference_config.get("serialization_tag", "cartesian")
    if serialization_tag == "cartesian" and "fsq" in str(inference_config["model_path"]).lower():
        logger.info("Auto-detecting serialization_tag=fsq based on model path")
        serialization_tag = "fsq"
    if serialization_tag == "cartesian_binned":
        binned = True
    logger.info(f"Using serialization_tag={serialization_tag}")
    return binned, serialization_tag


def generate_conformers(inference_config: dict, mols_list: list):
    """Load the model onto the visible GPU and generate conformers for `mols_list`."""
    model, tokenizer = load_model_tokenizer(
        model_path=inference_config["model_path"],
        tokenizer_path=inference_config["tokenizer_path"],
        torch_dtype=inference_config["torch_dtype"],
        attention_imp=inference_config.get("attention_imp", "sdpa"),
        device=inference_config["device"],
    )
    logger.info(f"model loaded: {model.dtype=}, {model.device=}")

    # Use [/CONFORMER] as the primary stop token, falling back to <|endoftext|>
    eos_token_id = tokenizer.convert_tokens_to_ids("[/CONFORMER]")
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    logger.info(f"Using eos_token_id: {eos_token_id} for generation")

    binned = inference_config["binned"]
    serialization_tag = inference_config["serialization_tag"]
    stats = Counter({"smiles_mismatch": 0, "mol_parse_fail": 0, "no_eos": 0})
    batch_size = int(inference_config["batch_size"])
    generations_all = defaultdict(list)

    uniform_bin_config = None
    quantile_bin_config = None
    fsq_decoder = None
    if serialization_tag == "uniform":
        uniform_bin_config = load_bin_config_for_mode(
            "uniform",
            inference_config.get("uniform_bin_config_path"),
        )
    elif serialization_tag == "quantile":
        quantile_bin_config = load_bin_config_for_mode(
            "quantile",
            inference_config.get("quantile_bin_config_path"),
        )
    elif serialization_tag == "fsq":
        fsq_ckpt_path = _resolve_fsq_ckpt_path(inference_config.get("fsq_ckpt_path"))
        logger.info(f"Loading FSQ codec from {fsq_ckpt_path}")
        fsq_decoder = MolFSQModel.load_from_checkpoint(str(fsq_ckpt_path), device=model.device)
        fsq_decoder.eval()

    for start in tqdm(range(0, len(mols_list), batch_size), desc="generating"):
        batch = mols_list[start:start + batch_size]
        for sub_batch in split_batch_on_geom_size(batch, max_geom_len=80):
            outputs, stats_ = process_batch(
                model,
                tokenizer,
                sub_batch,
                gen_config=inference_config["gen_config"],
                eos_token_id=eos_token_id,
                binned=binned,
                serialization_tag=serialization_tag,
                uniform_bin_config_path=inference_config.get("uniform_bin_config_path"),
                quantile_bin_config_path=inference_config.get("quantile_bin_config_path"),
                uniform_bin_config=uniform_bin_config,
                quantile_bin_config=quantile_bin_config,
                fsq_decoder=fsq_decoder,
            )
            stats.update(stats_)
            for k, v in outputs.items():
                generations_all[k].extend(v)

    return generations_all, stats


def _visible_cuda_devices() -> list[str]:
    """GPU ids this process is allowed to use, honouring a SLURM-set allocation."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [d.strip() for d in visible.split(",") if d.strip()]
    return [str(i) for i in range(torch.cuda.device_count())]


def _shard_worker(
    inference_config: dict,
    results_path: str,
    shard_idx: int,
    num_shards: int,
    cuda_device: str,
) -> None:
    """Generate one stride-slice of the prompt list on a single dedicated GPU.

    Pinning happens through CUDA_VISIBLE_DEVICES rather than a `cuda:N` device_map:
    the spawn start method gives each worker a fresh interpreter with no CUDA
    context yet, so the model, the FSQ codec and torch.compile's CUDA graphs all
    land on the same device without cross-device bookkeeping.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    set_seed(42 + shard_idx)
    logger.add(os.path.join(results_path, f"logs_shard{shard_idx}.txt"), rotation="50 MB")

    # Rebuilt here rather than shipped through the pipe: build_mols_list is
    # deterministic, and the full list is far larger than the test-set pickle.
    mols_list = build_mols_list(inference_config, load_test_data(inference_config))[shard_idx::num_shards]
    logger.info(f"shard {shard_idx}/{num_shards} on GPU {cuda_device}: {len(mols_list)} prompts")

    generations, stats = generate_conformers(inference_config, mols_list)
    with open(os.path.join(results_path, f"shard_{shard_idx}.pickle"), "wb") as shard_file:
        cloudpickle.dump({"generations": dict(generations), "stats": dict(stats)}, shard_file, protocol=4)
    logger.info(f"shard {shard_idx} done: {sum(len(v) for v in generations.values())} conformers, {stats=}")


def run_sharded_inference(inference_config: dict, results_path: str, num_gpus: int):
    devices = _visible_cuda_devices()
    if not devices:
        raise RuntimeError("Sharded inference requested but no CUDA devices are visible.")
    if len(devices) < num_gpus:
        logger.warning(
            f"Requested {num_gpus} GPUs but only {len(devices)} visible ({devices}); using {len(devices)}."
        )
        num_gpus = max(1, len(devices))

    ctx = mp.get_context("spawn")
    procs = []
    for shard_idx in range(num_gpus):
        proc = ctx.Process(
            target=_shard_worker,
            args=(inference_config, results_path, shard_idx, num_gpus, devices[shard_idx]),
        )
        proc.start()
        procs.append((shard_idx, proc))
    logger.info(f"Launched {len(procs)} inference shards on GPUs {devices[:num_gpus]}")

    for shard_idx, proc in procs:
        proc.join()
        if proc.exitcode != 0:
            logger.error(f"Shard {shard_idx} exited with code {proc.exitcode}")

    generations_all = defaultdict(list)
    stats = Counter({"smiles_mismatch": 0, "mol_parse_fail": 0, "no_eos": 0})
    merged = 0
    for shard_idx, _ in procs:
        shard_path = os.path.join(results_path, f"shard_{shard_idx}.pickle")
        if not os.path.exists(shard_path):
            logger.error(f"Shard {shard_idx} produced no results ({shard_path} missing); skipping it.")
            continue
        with open(shard_path, "rb") as shard_file:
            payload = cloudpickle.load(shard_file)
        for k, v in payload["generations"].items():
            generations_all[k].extend(v)
        stats.update(payload["stats"])
        os.remove(shard_path)
        merged += 1

    if merged < len(procs):
        stats["failed_shards"] = len(procs) - merged
    logger.info(f"Merged {merged}/{len(procs)} shards: {len(generations_all)} molecules")
    return generations_all, stats


def run_inference(inference_config: dict):
    results_path = os.path.join(*[inference_config["results_path"],
                                  datetime.now().strftime('%Y%m%d_%H%M%S') +
                                  '_' + inference_config["run_name"]])
    os.makedirs(results_path, exist_ok=True)
    logger.add(os.path.join(results_path, "logs.txt"), rotation="50 MB")
    logger.info(inference_config)

    binned, serialization_tag = resolve_serialization(inference_config)
    inference_config = dict(inference_config)
    inference_config["binned"] = binned
    inference_config["serialization_tag"] = serialization_tag

    num_gpus = int(inference_config.get("num_gpus", 1))
    if num_gpus > 1:
        # The prompt list is built per shard, not here: it is far larger than the
        # test-set pickle and would otherwise sit in this process for the whole run.
        generations_all, stats = run_sharded_inference(inference_config, results_path, num_gpus)
    else:
        set_seed(42)
        mols_list = build_mols_list(inference_config, load_test_data(inference_config))
        generations_all, stats = generate_conformers(inference_config, mols_list)

    save_results(results_path, dict(generations_all), stats)

    return generations_all, stats


# Test sets whose prompts are long enough that the default batch size OOMs.
TEST_SET_BATCH_SIZES = {"xl": 100, "qm9": 100, "valid": 128}


def apply_test_set(config: dict, test_set_name: str) -> dict:
    """Point `config` at one test set's data, with that set's batch-size override."""
    config["test_set"] = test_set_name
    if test_set_name in TEST_SET_BATCH_SIZES:
        config["batch_size"] = TEST_SET_BATCH_SIZES[test_set_name]
    if test_set_name == "valid":
        config["test_data_path"] = get_data_path("validation_pickle")
    else:
        config["test_data_path"] = get_data_path(f"{test_set_name}_smi")
    return config


def parse_model_specs(specs: list[str] | None) -> list[tuple[str, str]] | None:
    """Turn `alias:epoch` CLI strings into the (alias, epoch) pairs the grid expects."""
    if not specs:
        return None
    parsed = []
    for spec in specs:
        alias, sep, epoch = spec.partition(":")
        if not sep:
            raise ValueError(f"Model spec '{spec}' must be 'alias:epoch', e.g. qw600_fsq_bigdata:1e")
        parsed.append((alias.strip(), epoch.strip()))
    return parsed


def launch_inference_from_cli(
    device: str,
    grid_run_inference: bool,
    test_set: str | list[str] | None = None,
    xl: bool = False,
    qm9: bool = False,
    valid: bool = False,
    limit: int = None,
    binned: bool = False,
    serialization_tag: str = "cartesian",
    uniform_bin_config_path: str = None,
    quantile_bin_config_path: str = None,
    fsq_ckpt_path: str = None,
    icl: bool = False,
    icl_n: int = 5,
    parallel_jobs: int = 1,
    models: list[str] = None,
    num_gpus: int = 1,
    batch_size: int = None,
    num_gens: str = "1000_per_mol",
) -> None:
    # Determine which test sets to run
    test_sets_to_run = []
    if test_set:
        test_sets_to_run.extend([test_set] if isinstance(test_set, str) else list(test_set))
    if xl:
        test_sets_to_run.append("xl")
    if qm9:
        test_sets_to_run.append("qm9")
    if valid:
        test_sets_to_run.append("valid")
    if icl:
        test_sets_to_run.append(f"icl_{icl_n}")

    if not test_sets_to_run:
        logger.info("No test sets specified. Skipping inference.")
        return

    n_gpus = max(1, int(num_gpus))
    _slurm_devices = ["a100", "h100", "all", "research"]
    node = device if device in _slurm_devices else "local"
    executor = None

    # Set SLURM_CONF if not already set (needed for sbatch to work)
    if device in _slurm_devices:
        if not os.environ.get("SLURM_CONF"):
            # Try common SLURM config paths
            possible_paths = [
                "/cm/shared/apps/slurm/etc/slurm/slurm.conf",
                "/etc/slurm/slurm.conf",
                "/usr/local/etc/slurm.conf",
                "/opt/slurm/etc/slurm.conf",
            ]

            slurm_conf_found = None
            for path in possible_paths:
                if os.path.exists(path):
                    slurm_conf_found = path
                    break

            if slurm_conf_found:
                os.environ["SLURM_CONF"] = slurm_conf_found
                logger.info(f"Using SLURM_CONF: {slurm_conf_found}")
            else:
                logger.warning(
                    "SLURM_CONF is not set and no default config found. "
                    "Tried: " + ", ".join(possible_paths) + ". "
                    "sbatch may fail to locate slurmctld."
                )

    if device in _slurm_devices:
        executor = submitit.AutoExecutor(folder="outputs/slurm_jobs/conf_gen/job_%j")
    elif device == "local":
        executor = submitit.LocalExecutor(folder="outputs/slurm_jobs/conf_gen/job_%j")

    if executor is not None:
        executor.update_parameters(
            name="conf_gen",
            timeout_min=5* 24 * 60,
            gpus_per_node=n_gpus,
            nodes=1,
            # Each shard is a full worker process, and the parent merges every
            # shard's conformers before writing, so memory scales with the GPU count.
            mem_gb=80 * n_gpus,
            cpus_per_task=n_gpus * 12,
            slurm_additional_parameters={"partition": node},
        )
        if device in _slurm_devices:
            logger.info("Disabling srun to avoid PMIx/MPI plugin issues.")
            executor.update_parameters(slurm_use_srun=False)

    # Base configuration template
    base_inference_config = {
        "model_path": get_ckpt("qw600_pre_binned_revisited_cartesian_isomeric", "4e"),
        "tokenizer_path": get_tokenizer_path("qwen3_0.6b_binned_258"),
        "torch_dtype": "bfloat16",
        "batch_size": batch_size or 256,
        "num_gpus": n_gpus,
        "num_gens": gen_num_codes[num_gens],
        "gen_config": sampling_configs["top_p_sampling1"],
        "device": "cuda",
        "results_path": get_base_path("gen_results_root"),
        "run_name": "qwen_pre_binned",
        "limit": limit,
        "binned": binned,
        "serialization_tag": serialization_tag,
        "uniform_bin_config_path": uniform_bin_config_path,
        "quantile_bin_config_path": quantile_bin_config_path,
        "fsq_ckpt_path": fsq_ckpt_path,
    }

    if grid_run_inference:
        param_grid = [
            ("qwen600_pre_binned_uniform_bigdata", "1e"),
            # ("qw600_pre_binned_uniform_revisited_finetuned_from_bigdata", "1e"),
            # ("qw600_pre_binned_uniform_revisited_finetuned_from_bigdata", "2e"),
            # ("qw600_pre_binned_uniform_revisited_finetuned_from_bigdata", "3e"),
            ("qw600_pre_binned_uniform_revisited_finetuned_from_bigdata", "4e"),
            # ("qw1700_pre_binned_uniform_revisited_isomeric", "1e"),
            # ("qw1700_pre_binned_uniform_revisited_isomeric", "2e"),
            # ("qw1700_pre_binned_uniform_revisited_isomeric", "3e"),
            ("qw1700_pre_binned_uniform_revisited_isomeric", "4e"),
            ("qw1700_pre_binned_uniform_bigdata", "1e"),
            # ("qwen1700_pre_binned_uniform_revisited_isomeric_4e_from_bigdata", "1e"),
            # ("qwen1700_pre_binned_uniform_revisited_isomeric_4e_from_bigdata", "2e"),
            # ("qwen1700_pre_binned_uniform_revisited_isomeric_4e_from_bigdata", "3e"),
            ("qwen1700_pre_binned_uniform_revisited_isomeric_4e_from_bigdata", "4e"),
            # ("qwen4000_pre_binned_uniform_revisited_isomeric", "1e"),
            # ("qwen4000_pre_binned_uniform_revisited_isomeric", "2e"),
            # ("qwen4000_pre_binned_uniform_revisited_isomeric", "3e"),
            ("qwen4000_pre_binned_uniform_revisited_isomeric", "4e"),
            ("qw600_fsq_bigdata_sft", "6e"),
            ("qw1700_fsq_bigdata_sft", "6e"),
            ("qwen4000_pre_binned_uniform_bigdata", "1e"),
            ("qw4000_revisited_isomeric_4e_from_bigdata", "4e"),
            # ("qwen4000_pre_binned_uniform_revisited_isomeric", "1e"),
            # ("qwen4000_pre_binned_uniform_revisited_isomeric", "2e"),
            # ("qwen4000_pre_binned_uniform_revisited_isomeric", "3e"),
            ("qw600_fsq_bigdata", "1e"),
            ("qw1700_fsq_bigdata", "1e"),
            # Checkpoints of the 0.6B from-base run (train job 234444, now finished).
            # 1e/2e generated under 234783, 3e under 234832.
            # ("qw600_geom_isomeric_4e_from_base", "1e"),
            # ("qw600_geom_isomeric_4e_from_base", "2e"),
            # ("qw600_geom_isomeric_4e_from_base", "3e"),
            # ("qw600_geom_isomeric_4e_from_base", "4e"),
            # ("qw600_geom_isomeric_4e_from_bigdata", "4e"),
            # ("qw1700_geom_isomeric_4e_from_bigdata", "4e"),
            # ("qw1700_geom_isomeric_4e_from_base", "4e"),
            ("qwen600_pre_binned_uniform_bigdata", "1e"),
            ("qw600_pre_binned_paired", "4e"),
        ]
        param_grid = parse_model_specs(models) or param_grid
        logger.info(f"Grid models: {param_grid}")
        jobs = []
        def _build_grid_config(base_config, model_key_epoch, test_set_name):
            model_alias, epoch = model_key_epoch
            resolved_path = get_ckpt(model_alias, epoch)
            cfg = dict(base_config)
            cfg["model_path"] = resolved_path
            if "fsq" in model_alias:
                cfg["tokenizer_path"] = get_tokenizer_path(
                    "qwen3_1.7b_fsq_4096" if "1700" in model_alias else "qwen3_0.6b_fsq_4096"
                )
                cfg["binned"] = False
                cfg["serialization_tag"] = "fsq"
            else:
                # Model alias substrings don't reliably encode which tokenizer /
                # serialization a checkpoint was trained with (e.g. the revisited
                # uniform-binned models carry neither "binned" nor "qwen3"), so
                # keep the tokenizer_path and serialization_tag from the base
                # config / CLI instead of guessing from the name.
                cfg["binned"] = cfg.get("serialization_tag") == "cartesian_binned"
            apply_test_set(cfg, test_set_name)
            path_parts = str(resolved_path).rstrip('/').split('/')
            model_name = path_parts[-2] if path_parts[-1] == "model" else path_parts[-1]
            cfg["run_name"] = f"{model_alias}_{model_name}_{test_set_name}"
            return cfg

        if executor is not None:
            with executor.batch():
                for model_path in param_grid:
                    for test_set_name in test_sets_to_run:
                        grid_config = _build_grid_config(base_inference_config, model_path, test_set_name)
                        job = executor.submit(run_inference, inference_config=grid_config)
                        jobs.append(job)
        else:
            for model_path in param_grid:
                for test_set_name in test_sets_to_run:
                    grid_config = _build_grid_config(base_inference_config, model_path, test_set_name)
                    logger.info(f"Running grid inference for {grid_config['model_path']} on {test_set_name}")
                    run_inference(inference_config=grid_config)
    else:
        if executor is not None:
            with executor.batch():
                for test_set_name in test_sets_to_run:
                    inference_config = dict(base_inference_config)
                    apply_test_set(inference_config, test_set_name)
                    inference_config["run_name"] = f"new_data_p1_{test_set_name}"

                    logger.info(f"Running inference for {test_set_name} with config: {inference_config}")
                    job = executor.submit(run_inference, inference_config=inference_config)
        else:
            for test_set_name in test_sets_to_run:
                inference_config = dict(base_inference_config)
                apply_test_set(inference_config, test_set_name)
                inference_config["run_name"] = f"new_data_p1_{test_set_name}"

                logger.info(f"Running inference for {test_set_name} with config: {inference_config}")
                run_inference(inference_config=inference_config)


if __name__ == "__main__":
    # Under `python -m ...` every function here belongs to __main__, and cloudpickle
    # serialises __main__ functions by value. submitit then has to pickle the whole
    # reference graph of run_inference. Going through the imported module instead
    # keeps the submitted callables pickled by reference.
    from molgen3D.evaluation.inference import launch_inference_from_cli, set_seed

    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["local", "a100", "h100", "all", "research"], default="research")
    parser.add_argument("--grid_run_inference", action="store_true")
    parser.add_argument(
        "--test_set",
        type=str,
        nargs="+",
        choices=[
            "clean", "distinct", "corrected", "xl", "qm9", "valid", "revisited",
            "casf16", "casf16_opt", "casf16_ref_chembl", "casf16_core_chembl", "rnp",
            "druglike",
        ],
        default=["revisited"],
        help="One or more test sets to run.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Grid models as 'alias:epoch' pairs, e.g. "
            "--models qw600_fsq_bigdata:1e qw1700_fsq_bigdata:1e. "
            "Overrides the hardcoded param_grid."
        ),
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="GPUs per job; >1 shards the prompt list across one worker process per GPU.",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Override the generation batch size.")
    parser.add_argument(
        "--num_gens",
        type=str,
        choices=sorted(gen_num_codes.keys()),
        default="10x_per_mol",
        help="How many conformers to generate per prompt (see gen_num_codes).",
    )
    parser.add_argument("--binned", action="store_true", default=False)
    parser.add_argument(
        "--serialization_tag",
        type=str,
        choices=["cartesian", "uniform", "quantile", "fsq"],
        default="cartesian",
        help="Select decoding scheme.",
    )
    parser.add_argument(
        "--fsq_ckpt_path",
        type=str,
        default=None,
        help="Optional path to the FSQ codec checkpoint (defaults to the standard d1024_v4096 codec).",
    )
    parser.add_argument(
        "--uniform_bin_config_path",
        type=str,
        default=None,
        help="Optional BinConfig path for uniform decoding.",
    )
    parser.add_argument(
        "--quantile_bin_config_path",
        type=str,
        default=None,
        help="Optional BinConfig path for quantile decoding.",
    )
    parser.add_argument("--xl", action="store_true")
    parser.add_argument("--qm9", action="store_true")
    parser.add_argument("--valid", action="store_true", help="Run inference on validation set")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--icl", action="store_true")
    parser.add_argument("--icl_n", type=int, default=5)
    parser.add_argument("--parallel_jobs", type=int, default=1, help="Number of parallel inference jobs for local execution")
    args = parser.parse_args()
    launch_inference_from_cli(
        device=args.device,
        grid_run_inference=args.grid_run_inference,
        test_set=args.test_set,
        xl=args.xl,
        qm9=args.qm9,
        valid=args.valid,
        limit=args.limit,
        binned=args.binned,
        serialization_tag=args.serialization_tag,
        uniform_bin_config_path=args.uniform_bin_config_path,
        quantile_bin_config_path=args.quantile_bin_config_path,
        fsq_ckpt_path=args.fsq_ckpt_path,
        icl=args.icl,
        icl_n=args.icl_n,
        parallel_jobs=args.parallel_jobs,
        models=args.models,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        num_gens=args.num_gens,
    )
