from transformers import AutoTokenizer, AutoModelForCausalLM
from rdkit import RDLogger, rdBase
import torch
import cloudpickle
import random
from transformers.generation.utils import GenerateDecoderOnlyOutput
import yaml
import itertools
import re
from tqdm import tqdm
from loguru import logger
from collections import defaultdict, Counter
import submitit
import os
import argparse
from datetime import datetime
import time

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

# from utils import parse_molecule_with_coordinates
from molgen3D.data_processing.utils import decode_cartesian_raw
from molgen3D.data_processing.smiles_encoder_decoder import decode_cartesian_v2, strip_smiles
from molgen3D.evaluation.utils import (
    extract_between,
    same_molecular_graph,
    log_cuda_memory,
    log_cuda_summary,
    estimate_decoder_flops_per_token,
    detect_peak_flops,
    log_mfu,
)
from molgen3D.config.paths import get_ckpt, get_tokenizer_path, get_data_path, get_base_path
from molgen3D.config.sampling_config import sampling_configs, gen_num_codes

torch.backends.cudnn.benchmark = False
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")
rdBase.DisableLog("rdApp.error")

# Reduce CUDA memory fragmentation for large batch inference
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def set_seed(seed=42):
    random.seed(seed)  # Python random module
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # All GPUs (if using multi-GPU)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model_tokenizer(
    model_path,
    tokenizer_path,
    torch_dtype="bfloat16",
    attention_imp="flash_attention_2",
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

    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model = torch.compile(model, mode="reduce-overhead")
        logger.info(
            f"torch.compile succeeded; using optimized graph. Compiled type={type(model)}"
        )
        log_cuda_summary("Post-compile")
    except Exception as compile_err:
        logger.warning(f"torch.compile failed, continuing with eager mode: {compile_err}")
    finally:
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

def process_batch(model, tokenizer, batch: list[list], gen_config, eos_token_id):
    generations = defaultdict(list)
    stats = {"smiles_mismatch":0, "mol_parse_fail" :0, "no_eos":0}
    
    # Extract prompts and geom_smiles from batch
    prompts = [item[1] for item in batch]
    geom_smiles_list = [item[0] for item in batch]
    
    tokenized_prompts = tokenizer(prompts,
                                  return_tensors="pt",
                                  padding=True,
                                  pad_to_multiple_of=8)
    tokenized_prompts = {k: v.to(model.device, non_blocking=True) for k, v in tokenized_prompts.items()}
    tokenized_prompts["attention_mask"] = tokenized_prompts["attention_mask"].contiguous()
    start_time = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=tokenized_prompts["input_ids"], 
            attention_mask=tokenized_prompts["attention_mask"],
            max_new_tokens=2500,
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
    decoded_outputs = tokenizer.batch_decode(sequences, skip_special_tokens=True)
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
                    mol_obj = decode_cartesian_v2(generated_conformer)
                    # logger.info(f"smiles match: \n{canonical_smiles=}\n{generated_smiles=}\n{generated_conformer=}")
                    generations[geom_smiles].append(mol_obj)
                except:
                    logger.info(f"smiles fails parsing: \n{canonical_smiles=}\n{generated_smiles=}\n{generated_conformer=}")
                    stats["mol_parse_fail"] += 1
        else:
            stats["no_eos"] += 1
            logger.info(f"no eos: \n{out[:1000]=}")
    return generations, stats

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


def shard_mols_list(mols_list: list, shard_id: int, total_shards: int) -> list:
    """Split mols_list into shards for multi-GPU data parallelism.

    Uses strided sharding (not contiguous chunks) to distribute molecule sizes
    evenly across GPUs, since mols_list is sorted by length.

    Args:
        mols_list: Full list of (geom_smiles, prompt) tuples, sorted by length
        shard_id: This GPU's shard index (0-indexed)
        total_shards: Total number of GPUs/shards

    Returns:
        Subset of mols_list for this shard
    """
    if total_shards <= 1:
        return mols_list
    # Strided sharding: GPU 0 gets items 0, N, 2N, ...; GPU 1 gets 1, N+1, 2N+1, ...
    return mols_list[shard_id::total_shards]


def concatenate_shard_results(results_path: str, total_shards: int) -> dict:
    """Merge per-GPU pickle files into a single result dictionary.

    Args:
        results_path: Directory containing generation_results_gpu_*.pickle files
        total_shards: Number of shards to merge

    Returns:
        Merged dictionary {geom_smiles: [mol_objects]}
    """
    merged = defaultdict(list)

    for shard_id in range(total_shards):
        shard_file = os.path.join(results_path, f"generation_results_gpu_{shard_id}.pickle")
        if not os.path.exists(shard_file):
            logger.warning(f"Missing shard file: {shard_file}")
            continue

        with open(shard_file, 'rb') as f:
            shard_data = cloudpickle.load(f)

        for k, v in shard_data.items():
            merged[k].extend(v)

        logger.info(f"Merged shard {shard_id}: {len(shard_data)} unique SMILES, {sum(len(v) for v in shard_data.values())} conformers")

    # Save merged result
    merged_file = os.path.join(results_path, "generation_results.pickle")
    with open(merged_file, 'wb') as f:
        cloudpickle.dump(dict(merged), f, protocol=4)

    total_conformers = sum(len(v) for v in merged.values())
    logger.info(f"Final merged result: {len(merged)} unique SMILES, {total_conformers} total conformers")
    return dict(merged)


def run_inference(inference_config: dict, shard_id: int | None = None, total_shards: int | None = None):
    # For multi-GPU: use pre-created shared results_path; for single GPU: create timestamped path
    if shard_id is not None and total_shards is not None and total_shards > 1:
        # Multi-GPU mode: use shared path created by launcher, set per-shard seed
        results_path = inference_config["results_path"]
        set_seed(42 + shard_id)
        log_suffix = f"_gpu_{shard_id}"
    else:
        # Single GPU mode: create timestamped path (original behavior)
        results_path = os.path.join(inference_config["results_path"],
                                    datetime.now().strftime('%Y%m%d_%H%M%S') +
                                    '_' + inference_config["run_name"])
        log_suffix = ""

    os.makedirs(results_path, exist_ok=True)
    logger.add(os.path.join(results_path, f"logs{log_suffix}.txt"), rotation="50 MB")
    logger.info(f"shard_id={shard_id}, total_shards={total_shards}")
    logger.info(inference_config)

    model, tokenizer = load_model_tokenizer(model_path=inference_config["model_path"],
                                            tokenizer_path=inference_config["tokenizer_path"],
                                            torch_dtype=inference_config["torch_dtype"])
    logger.info(f"model loaded: {model.dtype=}, {model.device=}")
    
    # eos_token_id = tokenizer.encode("[/CONFORMER]", add_special_tokens=False)
    eos_token_id = tokenizer.encode("<|endoftext|>", add_special_tokens=False)
    with open(inference_config["test_data_path"],'rb') as test_data_file:
        test_data = cloudpickle.load(test_data_file)

    mols_list = []
    test_set: str = inference_config.get("test_set", "distinct")
    if test_set in ("clean"):
        for geom_smiles, data in test_data.items():
            mols_list.extend([(geom_smiles, f"[SMILES]{data['corrected_smi']}[/SMILES]")] * data["num_confs"] * 2)
    elif test_set == "distinct":
        logger.info("Processing as distinct dataset")
        for geom_smiles, data in test_data.items():
            for sub_smiles, count in data["sub_smiles_counts"].items():
                mols_list.extend([(geom_smiles, f"[SMILES]{sub_smiles}[/SMILES]")] * count * 2)
    elif test_set == "xl":
        logger.info("Processing as xl dataset")
        for geom_smiles, data in test_data.items():
            for sub_smiles, count in data["sub_smiles_counts"].items():
                mols_list.extend([(geom_smiles, f"[SMILES]{sub_smiles}[/SMILES]")] * count * 2)
    elif test_set == "qm9":
        logger.info("Processing as qm9 dataset")
        for geom_smiles, data in test_data.items():
            for sub_smiles, count in data["sub_smiles_counts"].items():
                mols_list.extend([(geom_smiles, f"[SMILES]{sub_smiles}[/SMILES]")] * count * 2)
    logger.info(f"mols_list length: {len(mols_list)}, mols_list_distinct: {len(set(mols_list))}, mols_list: {mols_list[:10]}")

    mols_list.sort(key=lambda x: len(x[0]))

    limit = inference_config.get("limit")
    mols_list = mols_list[:limit]

    # Apply sharding for multi-GPU parallelism
    if shard_id is not None and total_shards is not None and total_shards > 1:
        full_len = len(mols_list)
        mols_list = shard_mols_list(mols_list, shard_id, total_shards)
        logger.info(f"GPU {shard_id}/{total_shards}: Processing {len(mols_list)}/{full_len} molecules")

    stats = Counter({"smiles_mismatch":0, "mol_parse_fail" :0, "no_eos":0})
    batch_size = int(inference_config["batch_size"])
    generations_all = defaultdict(list)

    for start in tqdm(range(0, len(mols_list), batch_size), desc="generating"):
        batch = mols_list[start:start + batch_size]
        for sub_batch in split_batch_on_geom_size(batch, max_geom_len=80):
            outputs, stats_ = process_batch(model, tokenizer, sub_batch, inference_config["gen_config"], eos_token_id)
            stats.update(stats_)
            for k, v in outputs.items():
                generations_all[k].extend(v)

    # Save results with shard-specific filename for multi-GPU, or standard name for single GPU
    if shard_id is not None and total_shards is not None and total_shards > 1:
        output_filename = f"generation_results_gpu_{shard_id}.pickle"
        with open(os.path.join(results_path, output_filename), 'wb') as f:
            cloudpickle.dump(dict(generations_all), f, protocol=4)
        with open(os.path.join(results_path, f"stats_gpu_{shard_id}.txt"), 'w') as f:
            f.write(f"{stats=}")
        logger.info(f"GPU {shard_id}: Saved {len(generations_all)} unique SMILES to {output_filename}")
    else:
        save_results(results_path, dict(generations_all), stats)

    return generations_all, stats


def launch_inference_from_cli(device: str, grid_run_inference: bool, test_set:str = None, xl:bool = False, qm9:bool = False, limit: int = None, num_gpus: int = 1) -> None:
    # Determine which test sets to run
    test_sets_to_run = []
    if test_set:
        test_sets_to_run.append(test_set)
    if xl:
        test_sets_to_run.append("xl")
    if qm9:
        test_sets_to_run.append("qm9")
    if not test_sets_to_run:
        logger.info("No test sets specified. Skipping inference.")
        return

    node = device if device in ["a100", "h100"] else "local"

    # Create executor for inference jobs (1 GPU per job for multi-GPU parallelism)
    executor = None
    if device in ["a100", "h100"]:
        executor = submitit.AutoExecutor(folder="outputs/slurm_jobs/conf_gen/job_%j")
    elif device == "local":
        executor = submitit.LocalExecutor(folder="outputs/slurm_jobs/conf_gen/job_%j")

    executor.update_parameters(
        name="conf_gen",
        timeout_min=24 * 24 * 60,
        gpus_per_node=1,  # Each job gets 1 GPU
        nodes=1,
        mem_gb=80,
        cpus_per_task=12,
        slurm_additional_parameters={"partition": node},
    )

    logger.info(f"Multi-GPU mode: {num_gpus} GPUs requested")
    
    # Base configuration template
    base_inference_config = {
        "model_path": get_ckpt("m380_conf_v2","2e"),
        "tokenizer_path": get_tokenizer_path("qwen3_0.6b_custom"),
        "torch_dtype": "bfloat16",
        "batch_size": 256,
        "num_gens": gen_num_codes["2k_per_conf"],
        "gen_config": sampling_configs["top_p_sampling1"],
        "device": "cuda",
        "results_path": get_base_path("gen_results_root"),
        "run_name": "qwen_pre",
        "limit": limit,
    }

    if grid_run_inference:
        param_grid = {
            # "model_path": [("m380_conf_v2", "4e")],
            # "model_path": [("m600_qwen_pre", "4e"), ("m600_qwen_scr", "4e")],
            "model_path": [("qwen3_grpo_251226_1635", "4000")],
            # , ("qwen3_grpo_251224_1839", "2000"), ("qwen3_grpo_251228_1438", "4000"), ("qwen3_grpo_251228_1438", "2000")],
            # "model_path": [("m380_conf_v2", "1e")],
        }
        jobs = []
        if executor is not None:
            with executor.batch():
                for model_key in param_grid["model_path"]:
                    for test_set_name in test_sets_to_run:
                        grid_config = dict(base_inference_config)
                        if isinstance(model_key, tuple):
                            grid_config["model_path"] = get_ckpt(model_key[0], model_key[1])
                            model_key_str = f"{model_key[0]}_{model_key[1]}"
                        else:
                            grid_config["model_path"] = get_ckpt(model_key)
                            model_key_str = model_key
                        
                        if test_set_name == "xl":
                            grid_config["batch_size"] = 100
                        
                        if test_set_name == "qm9":
                            grid_config["batch_size"] = 100

                        grid_config["test_data_path"] = get_data_path(f"{test_set_name}_smi")
                        grid_config["test_set"] = test_set_name
                        grid_config["run_name"] = f"{model_key_str}_{test_set_name}"
                        
                        job = executor.submit(run_inference, inference_config=grid_config)
                        jobs.append(job)
    else:
        if executor is not None:
            for test_set_name in test_sets_to_run:
                inference_config = dict(base_inference_config)
                if test_set_name == "xl":
                    inference_config["batch_size"] = 100
                if test_set_name == "qm9":
                    inference_config["batch_size"] = 100
                inference_config["test_data_path"] = get_data_path(f"{test_set_name}_smi")
                inference_config["test_set"] = test_set_name
                inference_config["run_name"] = f"new_data_p1_{test_set_name}"

                if num_gpus > 1:
                    # Multi-GPU mode: create shared results directory and submit N shard jobs
                    shared_results_path = os.path.join(
                        get_base_path("gen_results_root"),
                        datetime.now().strftime('%Y%m%d_%H%M%S') + f'_multi_gpu_{test_set_name}'
                    )
                    os.makedirs(shared_results_path, exist_ok=True)
                    inference_config["results_path"] = shared_results_path

                    logger.info(f"Submitting {num_gpus} parallel jobs for {test_set_name}")
                    shard_jobs = []
                    with executor.batch():
                        for gpu_id in range(num_gpus):
                            job = executor.submit(
                                run_inference,
                                inference_config=inference_config,
                                shard_id=gpu_id,
                                total_shards=num_gpus
                            )
                            shard_jobs.append(job)

                    # Submit concatenation job with dependency on all shard jobs
                    if device in ["a100", "h100"]:
                        concat_executor = submitit.AutoExecutor(folder="outputs/slurm_jobs/conf_gen/concat_%j")
                        job_ids = ":".join(str(j.job_id) for j in shard_jobs)
                        concat_executor.update_parameters(
                            name="concat_results",
                            timeout_min=60,
                            gpus_per_node=0,  # CPU only
                            nodes=1,
                            cpus_per_task=4,
                            mem_gb=32,
                            slurm_additional_parameters={
                                "partition": node,
                                "dependency": f"afterok:{job_ids}"
                            },
                        )
                        concat_job = concat_executor.submit(
                            concatenate_shard_results,
                            results_path=shared_results_path,
                            total_shards=num_gpus
                        )
                        logger.info(f"Submitted concatenation job {concat_job.job_id} (depends on {job_ids})")
                else:
                    # Single GPU mode: original behavior
                    logger.info(f"Running inference for {test_set_name} with config: {inference_config}")
                    with executor.batch():
                        job = executor.submit(run_inference, inference_config=inference_config)
        else:
            # Local execution
            for test_set_name in test_sets_to_run:
                inference_config = dict(base_inference_config)
                if test_set_name == "xl":
                    inference_config["batch_size"] = 100
                if test_set_name == "qm9":
                    inference_config["batch_size"] = 100
                inference_config["test_data_path"] = get_data_path(f"{test_set_name}_smi")
                inference_config["test_set"] = test_set_name
                inference_config["run_name"] = f"new_data_p1_{test_set_name}"

                logger.info(f"Running inference for {test_set_name} with config: {inference_config}")
                run_inference(inference_config=inference_config)

if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["local", "a100", "h100"], required=True)
    parser.add_argument("--grid_run_inference", action="store_true")
    parser.add_argument("--test_set", type=str, choices=["clean", "distinct", "corrected"], default=None)
    parser.add_argument("--xl", action="store_true")
    parser.add_argument("--qm9", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs for data-parallel inference (default: 1)")
    args = parser.parse_args()
    launch_inference_from_cli(device=args.device, grid_run_inference=args.grid_run_inference, test_set=args.test_set, xl=args.xl, qm9=args.qm9, limit=args.limit, num_gpus=args.num_gpus)

    
