import os
import json
import argparse
from datetime import datetime
import time
import random
from collections import defaultdict, Counter
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import cloudpickle
from tqdm import tqdm
from loguru import logger
import submitit
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from molgen3D.config.paths import get_ckpt, get_tokenizer_path, get_data_path, get_base_path
from molgen3D.config.sampling_config import sampling_configs
from molgen3D.data_processing.smiles_encoder_decoder import (
    decode_cartesian_v2,
    strip_smiles,
    decode_cartesian_binned_v2,
    get_bins_for_coords,
)
from molgen3D.evaluation.utils import (
    same_molecular_graph,
    log_mfu,
    log_cuda_memory,
    log_cuda_summary,
    estimate_decoder_flops_per_token,
    detect_peak_flops,
)
from molgen3D.training.grpo.logits_constraints_optimized import (
    ConformerControlLogitsProcessorOptimized as ConformerControlLogitsProcessor,
    ConformerCountStoppingCriteriaPerSequence,
)


def _run_from_config_file(config_path: str):
    """Submitit entry point: loads config from file and runs inference in the worker."""
    with open(config_path) as f:
        inference_config = json.load(f)
    from molgen3D.evaluation.inference_multiconf import run_multiconf_inference
    return run_multiconf_inference(inference_config)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_model_tokenizer(
    model_path,
    tokenizer_path,
    torch_dtype="bfloat16",
    attention_imp="sdpa",
    device="auto",
):
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

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

    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model = torch.compile(model, mode="reduce-overhead")
        logger.info(f"torch.compile succeeded. type={type(model)}")
        log_cuda_summary("Post-compile")
    except Exception as e:
        logger.warning(f"torch.compile failed, using eager mode: {e}")
    finally:
        log_cuda_memory("Post-compile")

    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    logger.info(f"{model.dtype=}, {model.device=}")
    return model, tokenizer


def save_results(results_path, generations, stats):
    with open(os.path.join(results_path, "generation_results.pickle"), "wb") as f:
        cloudpickle.dump(generations, f, protocol=4)
    with open(os.path.join(results_path, "generation_results.txt"), "w") as f:
        f.write(f"{stats=}")


def _decode_single_conformer(args):
    conformer_string, canonical_smiles, binned, bins_tuple = args
    bins = None
    if binned and bins_tuple is not None:
        ranges, bin_size = bins_tuple
        bins = get_bins_for_coords(ranges, bin_size=bin_size)
    try:
        if not same_molecular_graph(canonical_smiles, strip_smiles(conformer_string)):
            return None, "smiles_mismatch"
        mol_obj = decode_cartesian_binned_v2(conformer_string, bins) if binned else decode_cartesian_v2(conformer_string)
        return mol_obj, None
    except Exception:
        return None, "mol_parse_fail"


def decode_conformers_parallel(
    conformer_strings: List[str],
    canonical_smiles: str,
    binned: bool,
    bins,
    max_workers: int = 4,
) -> tuple[List, dict]:
    if not conformer_strings:
        return [], {}

    if len(conformer_strings) < 4:
        max_workers = 1

    bins_tuple = ([(-13.0, 13.0)] * 3, 0.104) if (binned and bins is not None) else None
    args_list = [(s, canonical_smiles, binned, bins_tuple) for s in conformer_strings]

    mol_objects = []
    error_counts = defaultdict(int)

    def _collect(results):
        for mol_obj, error in results:
            if mol_obj is not None:
                mol_objects.append(mol_obj)
            elif error:
                error_counts[error] += 1

    if max_workers <= 1:
        _collect(_decode_single_conformer(a) for a in args_list)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            _collect(executor.map(_decode_single_conformer, args_list))

    return mol_objects, dict(error_counts)


def _extract_conformer_strings(decoded_output: str, stats: Counter) -> List[str]:
    conformer_strings = []
    idx = 0
    while True:
        start = decoded_output.find("[CONFORMER]", idx)
        if start == -1:
            break
        content_start = start + len("[CONFORMER]")
        end = decoded_output.find("[/CONFORMER]", content_start)
        next_start = decoded_output.find("[CONFORMER]", content_start)
        if end == -1 or (next_start != -1 and next_start < end):
            stats["no_eos"] += 1
            idx = next_start if next_start != -1 else content_start
            continue
        conformer_strings.append(decoded_output[content_start:end])
        idx = end + len("[/CONFORMER]")
    return conformer_strings


def generate_multiple_conformers_batched(
    model,
    tokenizer,
    batch_prompts: List[str],
    batch_num_conformers: List[int],
    gen_config,
    binned: bool,
    stats: Counter,
) -> List[List]:
    """Generate conformers for a batch of SMILES in parallel."""
    conformer_start_ids = tokenizer.encode("[CONFORMER]", add_special_tokens=False)
    conformer_end_ids = tokenizer.encode("[/CONFORMER]", add_special_tokens=False)
    smiles_start_ids = tokenizer.encode("[SMILES]", add_special_tokens=False)
    smiles_end_ids = tokenizer.encode("[/SMILES]", add_special_tokens=False)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    banned_ids = set(smiles_start_ids) | set(smiles_end_ids)
    if pad_token_id is not None:
        banned_ids.add(pad_token_id)
    for tok in conformer_start_ids + conformer_end_ids:
        banned_ids.discard(tok)

    bins = get_bins_for_coords([(-13.0, 13.0)] * 3, bin_size=0.104) if binned else None

    batch_canonical_smiles = []
    for prompt in batch_prompts:
        smi = ""
        i = prompt.rfind("[SMILES]")
        if i != -1:
            j = prompt.find("[/SMILES]", i + len("[SMILES]"))
            if j != -1:
                smi = prompt[i + len("[SMILES]"):j]
        batch_canonical_smiles.append(smi)

    max_conformers = max(batch_num_conformers)

    tokenized = tokenizer(batch_prompts, return_tensors="pt", padding=True, pad_to_multiple_of=8)
    input_ids = tokenized["input_ids"].to(model.device, non_blocking=True)
    attention_mask = tokenized["attention_mask"].to(model.device, non_blocking=True).contiguous()

    logits_processor = ConformerControlLogitsProcessor(
        conformer_start_ids=conformer_start_ids,
        conformer_end_ids=conformer_end_ids,
        banned_token_ids=banned_ids,
        target_k=max_conformers,
        force_hard=True,
        eos_token_id=eos_token_id,
        target_counts=batch_num_conformers,
    )
    stopping_criteria = ConformerCountStoppingCriteriaPerSequence(
        conformer_end_ids=conformer_end_ids,
        target_counts=batch_num_conformers,
    )

    start_time = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=min(650 * max_conformers, 5000),
            generation_config=gen_config,
            logits_processor=[logits_processor],
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        sequences = outputs.sequences.detach().cpu()
        del outputs

    elapsed = time.perf_counter() - start_time
    prompt_lens = attention_mask.sum(dim=1).cpu()
    seq_lens = (sequences != pad_token_id).to(torch.int32).sum(dim=1)
    total_generated_tokens = int((seq_lens - prompt_lens).clamp(min=0).sum().item())
    log_mfu(model, total_generated_tokens, elapsed)

    decoded_outputs = tokenizer.batch_decode(sequences, skip_special_tokens=False)

    batch_results = []
    for decoded_output, canonical_smiles, target_count in zip(
        decoded_outputs, batch_canonical_smiles, batch_num_conformers
    ):
        decoded_output = (
            decoded_output
            .replace(tokenizer.eos_token, "")
            .replace(tokenizer.pad_token, "")
            .replace(";", "")
        )
        conformer_strings = _extract_conformer_strings(decoded_output, stats)
        conformers_to_decode = conformer_strings[:target_count]

        stats["conformers_requested"] += target_count
        stats["conformers_extracted"] += len(conformers_to_decode)
        stats["conformers_never_generated"] += target_count - len(conformers_to_decode)

        mol_objects, decode_errors = decode_conformers_parallel(
            conformer_strings=conformers_to_decode,
            canonical_smiles=canonical_smiles,
            binned=binned,
            bins=bins,
            max_workers=4,
        )
        for error_type, count in decode_errors.items():
            stats[error_type] += count

        batch_results.append(mol_objects)

    return batch_results


def _log_stats(stats: Counter, total_generated: int):
    req = max(1, stats["conformers_requested"])
    ext = max(1, stats["conformers_extracted"])
    lines = [
        "=" * 70,
        f"Success rate: {100 * total_generated / req:.1f}%  ({total_generated:,} / {req:,})",
        f"Extracted:    {stats['conformers_extracted']:,}  ({100 * stats['conformers_extracted'] / req:.1f}%)",
        f"Never gen'd:  {stats['conformers_never_generated']:,}  ({100 * stats['conformers_never_generated'] / req:.1f}%)",
        "-" * 70,
        f"Parsed ok:    {total_generated:,}  ({100 * total_generated / ext:.1f}% of extracted)",
        f"SMILES mism:  {stats['smiles_mismatch']:,}  ({100 * stats['smiles_mismatch'] / ext:.1f}% of extracted)",
        f"Parse fail:   {stats['mol_parse_fail']:,}  ({100 * stats['mol_parse_fail'] / ext:.1f}% of extracted)",
        f"No EOS:       {stats['no_eos']:,}",
        "=" * 70,
    ]
    for line in lines:
        logger.info(line)
    accounted = total_generated + stats["smiles_mismatch"] + stats["mol_parse_fail"]
    if accounted != stats["conformers_extracted"]:
        logger.warning(f"Accounting mismatch: {accounted} != {stats['conformers_extracted']}")


def _get_test_data_path(test_set_name: str) -> str:
    if test_set_name == "valid":
        return str(get_data_path("validation_pickle"))
    return str(get_data_path(f"{test_set_name}_smi"))


def run_multiconf_inference(inference_config: dict):
    torch.set_grad_enabled(False)

    if isinstance(inference_config.get("gen_config"), dict):
        inference_config["gen_config"] = GenerationConfig.from_dict(inference_config["gen_config"])

    target_device = inference_config.get("device", "cuda")
    set_seed(42)

    results_path = os.path.join(
        inference_config["results_path"],
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + inference_config["run_name"],
    )
    os.makedirs(results_path, exist_ok=True)
    logger.add(os.path.join(results_path, "logs.txt"), rotation="50 MB")
    logger.info(inference_config)

    model, tokenizer = load_model_tokenizer(
        model_path=inference_config["model_path"],
        tokenizer_path=inference_config["tokenizer_path"],
        torch_dtype=inference_config["torch_dtype"],
        device=target_device,
    )

    with open(inference_config["test_data_path"], "rb") as f:
        test_data = cloudpickle.load(f)

    test_set: str = inference_config.get("test_set", "distinct")
    multiplier = inference_config.get("conformer_multiplier", 2)
    max_target = inference_config.get("max_target_per_smiles")

    def cap(n):
        return min(n, max_target) if max_target is not None else n

    smiles_to_process = []
    if test_set == "clean":
        for geom_smiles, data in test_data.items():
            smiles_to_process.append((geom_smiles, data["corrected_smi"], cap(data["num_confs"] * multiplier)))
    elif test_set in ("distinct", "xl", "qm9"):
        for geom_smiles, data in test_data.items():
            for sub_smiles, count in data["sub_smiles_counts"].items():
                smiles_to_process.append((geom_smiles, sub_smiles, cap(count * multiplier)))
    elif test_set == "valid":
        for smiles, conf_list in test_data.items():
            if conf_list:
                smiles_to_process.append((smiles, smiles, cap(len(conf_list) * multiplier)))

    total_conformers_to_generate = sum(c for _, _, c in smiles_to_process)
    logger.info(f"SMILES: {len(smiles_to_process)}, conformers to generate: {total_conformers_to_generate}")

    limit = inference_config.get("limit")
    if limit:
        smiles_to_process = smiles_to_process[:limit]
        total_conformers_to_generate = sum(c for _, _, c in smiles_to_process)
        logger.info(f"Limited to {len(smiles_to_process)} SMILES, {total_conformers_to_generate} conformers")

    conformers_per_batch = inference_config.get("conformers_per_batch", 8)
    binned = inference_config.get("binned", False)
    if not binned and "binned" in str(inference_config["model_path"]):
        logger.info("Auto-detecting binned=True from model path")
        binned = True

    smiles_batch_size = inference_config.get("smiles_batch_size", 32)
    logger.info(f"Batch: {smiles_batch_size} SMILES × {conformers_per_batch} conformers")

    stats = Counter({
        "smiles_mismatch": 0,
        "mol_parse_fail": 0,
        "no_eos": 0,
        "no_conformer_start": 0,
        "conformers_requested": 0,
        "conformers_extracted": 0,
        "conformers_never_generated": 0,
    })
    generations_all = defaultdict(list)
    total_conformers_generated = 0

    remaining_conformers = {
        (geom_smiles, sub_smiles): target_count
        for geom_smiles, sub_smiles, target_count in smiles_to_process
    }

    pbar = tqdm(total=total_conformers_to_generate, desc="Generating conformers")
    batch_num = 0

    while remaining_conformers:
        batch_num += 1
        current_batch = []
        batch_generation_tracker = {}

        for (geom_smiles, sub_smiles), remaining in list(remaining_conformers.items()):
            if len(current_batch) >= smiles_batch_size:
                break
            batch_generation_tracker[(geom_smiles, sub_smiles)] = 0
            remaining_for_smiles = remaining
            while remaining_for_smiles > 0 and len(current_batch) < smiles_batch_size:
                to_generate = min(remaining_for_smiles, conformers_per_batch)
                current_batch.append((geom_smiles, sub_smiles, to_generate))
                batch_generation_tracker[(geom_smiles, sub_smiles)] += to_generate
                remaining_for_smiles -= to_generate

        if not current_batch:
            break

        logger.info(f"Batch {batch_num}: {len(current_batch)} slots, {len(batch_generation_tracker)} unique SMILES")

        batch_prompts = [f"[SMILES]{sub_smiles}[/SMILES]" for _, sub_smiles, _ in current_batch]
        batch_targets = [min(rem, conformers_per_batch) for _, _, rem in current_batch]

        batch_results = generate_multiple_conformers_batched(
            model=model,
            tokenizer=tokenizer,
            batch_prompts=batch_prompts,
            batch_num_conformers=batch_targets,
            gen_config=inference_config["gen_config"],
            binned=binned,
            stats=stats,
        )

        smiles_accumulated = defaultdict(list)
        for (geom_smiles, sub_smiles, _), mol_objects in zip(current_batch, batch_results):
            smiles_accumulated[(geom_smiles, sub_smiles)].extend(mol_objects)

        for (geom_smiles, sub_smiles), mol_objects_list in smiles_accumulated.items():
            num_generated = len(mol_objects_list)
            if num_generated > 0:
                generations_all[geom_smiles].extend(mol_objects_list)
                total_conformers_generated += num_generated
                pbar.update(num_generated)

            new_remaining = remaining_conformers[(geom_smiles, sub_smiles)] - batch_generation_tracker[(geom_smiles, sub_smiles)]
            if new_remaining <= 0:
                del remaining_conformers[(geom_smiles, sub_smiles)]
            else:
                remaining_conformers[(geom_smiles, sub_smiles)] = new_remaining

    pbar.close()
    logger.info(f"Done: {total_conformers_generated}/{total_conformers_to_generate} conformers generated")
    _log_stats(stats, total_conformers_generated)
    save_results(results_path, dict(generations_all), stats)
    return generations_all, stats


def launch_multiconf_inference_from_cli(
    device: str = "all",
    grid_run_inference: bool = False,
    test_set: str = "distinct",
    xl: bool = False,
    qm9: bool = False,
    smiles_batch_size: int = 32,
    conformers_per_batch: int = 8,
    conformer_multiplier: int = 2,
    limit: Optional[int] = None,
    binned: bool = False,
    parallel_jobs: int = 1,
    max_target_per_smiles: Optional[int] = None,
) -> None:
    test_sets_to_run = [s for s, flag in [(test_set, True), ("xl", xl), ("qm9", qm9)] if flag and s]
    if not test_sets_to_run:
        logger.info("No test sets specified.")
        return

    logger.info(f"Test sets: {test_sets_to_run}, device: {device}, batch: {smiles_batch_size}×{conformers_per_batch}")

    executor = None
    if device in ("a100", "h100", "all"):
        executor = submitit.AutoExecutor(folder="outputs/slurm_jobs/multiconf_gen/job_%j")
        executor.update_parameters(
            name="multiconf_gen",
            timeout_min=24 * 60,
            gpus_per_node=1,
            nodes=1,
            mem_gb=80,
            cpus_per_task=12,
            slurm_additional_parameters={"partition": device},
            slurm_use_srun=False,
        )

    base_config = {
        "model_path": str(get_ckpt("qw600_pre_binned_grouped", "5e")),
        "tokenizer_path": str(get_tokenizer_path("qwen3_0.6b_binned")),
        "torch_dtype": "bfloat16",
        "gen_config": sampling_configs["top_p_sampling1"].to_dict(),
        "device": "cuda",
        "results_path": str(get_base_path("gen_results_root")),
        "run_name": "multiconf_grouped",
        "smiles_batch_size": smiles_batch_size,
        "conformers_per_batch": conformers_per_batch,
        "conformer_multiplier": conformer_multiplier,
        "limit": limit,
        "binned": binned,
        "max_target_per_smiles": max_target_per_smiles,
    }

    if grid_run_inference:
        param_grid = [
            ("qw600_pre_binned_grouped", "1e"),
            ("qw600_pre_binned_grouped", "2e"),
            ("qw600_pre_binned_grouped", "3e"),
            ("qw600_pre_binned_grouped", "4e"),
            ("qw600_pre_binned_grouped", "5e"),
        ]
        all_configs = []
        for model_key in param_grid:
            for test_set_name in test_sets_to_run:
                cfg = dict(base_config)
                cfg["model_path"] = str(get_ckpt(model_key[0], model_key[1]))
                cfg["test_data_path"] = _get_test_data_path(test_set_name)
                cfg["test_set"] = test_set_name
                cfg["run_name"] = f"multiconf_{model_key[0]}_{model_key[1]}_{test_set_name}"
                all_configs.append((cfg, cfg["run_name"]))
    else:
        all_configs = []
        for test_set_name in test_sets_to_run:
            cfg = dict(base_config)
            cfg["test_data_path"] = _get_test_data_path(test_set_name)
            cfg["test_set"] = test_set_name
            cfg["run_name"] = f"multiconf_{conformer_multiplier}x_{conformers_per_batch}batch_{test_set_name}"
            all_configs.append((cfg, cfg["run_name"]))

    if executor is not None:
        config_dir = os.path.join("outputs", "slurm_jobs", "multiconf_gen", "configs")
        os.makedirs(config_dir, exist_ok=True)
        with executor.batch():
            for config, run_name in all_configs:
                config_file = os.path.join(config_dir, f"{run_name}_config.json")
                with open(config_file, "w") as f:
                    json.dump(config, f, indent=2)
                executor.submit(_run_from_config_file, config_file)
    elif parallel_jobs <= 1:
        for config, run_name in all_configs:
            logger.info(f"Running {run_name}...")
            run_multiconf_inference(inference_config=config)
    else:
        max_workers = min(parallel_jobs, len(all_configs))
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(run_multiconf_inference, inference_config=cfg): name for cfg, name in all_configs}
            for future in as_completed(futures):
                try:
                    future.result()
                    logger.info(f"✓ {futures[future]}")
                except Exception as e:
                    logger.error(f"✗ {futures[future]}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, choices=["local", "a100", "h100", "all"], default="all")
    parser.add_argument("--grid_run_inference", action="store_true")
    parser.add_argument("--test_set", type=str, default="distinct")
    parser.add_argument("--xl", action="store_true")
    parser.add_argument("--qm9", action="store_true")
    parser.add_argument("--smiles_batch_size", type=int, default=32)
    parser.add_argument("--conformers_per_batch", type=int, default=8)
    parser.add_argument("--conformer_multiplier", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--binned", action="store_true", default=False)
    parser.add_argument("--parallel_jobs", type=int, default=1)
    parser.add_argument("--max_target_per_smiles", type=int)

    args = parser.parse_args()
    logger.info(f"Starting multiconf inference: {args}")

    launch_multiconf_inference_from_cli(
        device=args.device,
        grid_run_inference=args.grid_run_inference,
        test_set=args.test_set,
        xl=args.xl,
        qm9=args.qm9,
        smiles_batch_size=args.smiles_batch_size,
        conformers_per_batch=args.conformers_per_batch,
        conformer_multiplier=args.conformer_multiplier,
        limit=args.limit,
        binned=args.binned,
        parallel_jobs=args.parallel_jobs,
        max_target_per_smiles=args.max_target_per_smiles,
    )
