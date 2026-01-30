"""
Multi-conformer inference with forced conformer generation using ConformerControlLogitsProcessor.

This module generates multiple conformers per SMILES by forcing the model to continue
generating conformers without generating SMILES or EOS tokens until the desired count is reached.

Key features:
- Each SMILES has a different target conformer count based on ground truth count (k)
- Generates N*k conformers for each SMILES (e.g., 2*k by default)
- Batched generation: generates `conformers_per_batch` conformers per model.generate() call
  * Example: 20 target conformers with batch_size=8 → 3 calls (8+8+4)
- Uses ConformerControlLogitsProcessor to:
  * Force [CONFORMER] tags at the right times
  * Ban SMILES/EOS tokens during generation
  * Track conformer counts within each batch
- Uses ConformerCountStoppingCriteria to stop when target_k conformers are generated
- Accumulates output across batches to maintain context
"""
import os
import argparse
from datetime import datetime
import time
import random
import fcntl
from collections import defaultdict, Counter
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import cloudpickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from loguru import logger
import submitit

from molgen3D.data_processing.smiles_encoder_decoder import (
    decode_cartesian_v2,
    strip_smiles,
    decode_cartesian_binned_v2,
    get_bins_for_coords
)
from molgen3D.evaluation.utils import (
    same_molecular_graph,
    log_cuda_memory,
    log_cuda_summary,
    estimate_decoder_flops_per_token,
    detect_peak_flops,
    log_mfu,
)
from molgen3D.config.paths import get_ckpt, get_tokenizer_path, get_data_path, get_base_path
from molgen3D.config.sampling_config import sampling_configs
from molgen3D.training.grpo.logits_constraints import (
    ConformerControlLogitsProcessor,
    ConformerCountStoppingCriteria,
)

torch.set_grad_enabled(False)

# Global for NVML state
NVML_AVAILABLE = None


# Note: ConformerControlLogitsProcessor and ConformerCountStoppingCriteria
# are now imported from molgen3D.training.grpo.logits_constraints


def init_nvml():
    """Initialize NVML for GPU monitoring. Returns True if available, False otherwise."""
    global NVML_AVAILABLE
    if NVML_AVAILABLE is not None:
        return NVML_AVAILABLE
    try:
        import pynvml
        pynvml.nvmlInit()
        NVML_AVAILABLE = True
    except (ImportError, Exception):
        NVML_AVAILABLE = False
    return NVML_AVAILABLE


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        try:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def get_gpu_status():
    """Get status of all available GPUs including memory usage."""
    if not init_nvml():
        logger.warning("pynvml not available, cannot get GPU status")
        return []
    
    try:
        import pynvml
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_status = []
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            total_memory = mem_info.total / (1024**3)
            used_memory = mem_info.used / (1024**3)
            free_memory = mem_info.free / (1024**3)
            usage_percent = (used_memory / total_memory) * 100
            
            gpu_status.append({
                "gpu_idx": i,
                "total_memory_gb": round(total_memory, 1),
                "used_memory_gb": round(used_memory, 1),
                "free_memory_gb": round(free_memory, 1),
                "usage_percent": round(usage_percent, 1)
            })
        
        return gpu_status
    except Exception as e:
        logger.error(f"Error getting GPU status: {e}")
        return []


_GPU_LOCK_FILE = None


def find_free_gpu(min_memory_gb=1.0, max_memory_usage_percent=20.0):
    """Find and lock a free GPU with sufficient memory and low usage."""
    global _GPU_LOCK_FILE
    
    if not init_nvml():
        return None
    
    try:
        gpu_status = get_gpu_status()
        if not gpu_status:
            return None
        
        gpu_status.sort(key=lambda x: x["free_memory_gb"], reverse=True)
        
        for gpu in gpu_status:
            idx = gpu["gpu_idx"]
            
            if gpu["free_memory_gb"] < min_memory_gb or gpu["usage_percent"] > max_memory_usage_percent:
                continue
            
            lock_path = f"/tmp/molgen3d_gpu_{idx}.lock"
            try:
                f = open(lock_path, 'w')
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                _GPU_LOCK_FILE = f
                logger.info(f"Locked GPU {idx} (Free: {gpu['free_memory_gb']}GB, Usage: {gpu['usage_percent']}%)")
                return idx
            except (IOError, OSError):
                f.close()
                continue
        
        return None
    except Exception as e:
        logger.error(f"Error in find_free_gpu: {e}")
        return None


def load_model_tokenizer(
    model_path,
    tokenizer_path,
    torch_dtype="bfloat16",
    attention_imp="sdpa",
    device="auto",
):
    """Load model and tokenizer with appropriate configurations."""
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
        logger.info(f"torch.compile succeeded; using optimized graph. Compiled type={type(model)}")
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
    """Save generation results to pickle and text files."""
    with open(os.path.join(results_path, "generation_results.pickle"), 'wb') as results_file_pickle:
        cloudpickle.dump(generations, results_file_pickle, protocol=4)
    
    with open(os.path.join(results_path, "generation_results.txt"), 'w') as results_file_txt:
        results_file_txt.write(f"{stats=}")


def generate_multiple_conformers(
    model,
    tokenizer,
    smiles_prompt: str,
    num_conformers: int,
    gen_config,
    binned: bool,
    stats: Counter,
    geom_smiles: str,
    current_output: str = None,
) -> tuple[List, str]:
    """
    Generate multiple conformers for a single SMILES by forcing conformer continuation.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        smiles_prompt: The SMILES prompt string (e.g., "[SMILES]C[/SMILES]")
        num_conformers: Number of conformers to generate in this batch
        gen_config: Generation configuration
        binned: Whether to use binned decoding
        stats: Statistics counter
        geom_smiles: Original geom SMILES for tracking
        current_output: Previous output to continue from (for batching)
    
    Returns:
        Tuple of (list of successfully decoded molecule objects, full decoded output)
    """
    # Get token IDs for special tokens (as lists for multi-token sequences)
    conformer_start_ids = tokenizer.encode("[CONFORMER]", add_special_tokens=False)
    conformer_end_ids = tokenizer.encode("[/CONFORMER]", add_special_tokens=False)
    smiles_start_ids = tokenizer.encode("[SMILES]", add_special_tokens=False)
    smiles_end_ids = tokenizer.encode("[/SMILES]", add_special_tokens=False)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    
    # Setup banned tokens (SMILES start/end, EOS, PAD)
    banned_ids = set(smiles_start_ids)
    banned_ids.update(smiles_end_ids)
    if eos_token_id is not None:
        banned_ids.add(eos_token_id)
    if pad_token_id is not None:
        banned_ids.add(pad_token_id)
    # Don't ban conformer tokens
    for tok in conformer_start_ids:
        banned_ids.discard(tok)
    for tok in conformer_end_ids:
        banned_ids.discard(tok)
    
    # Create bins for binned decoding if needed
    bins = None
    if binned:
        ranges = [(-13.0, 13.0), (-13.0, 13.0), (-13.0, 13.0)]
        bins = get_bins_for_coords(ranges, bin_size=0.104)
    
    # Extract canonical SMILES from prompt
    canonical_smiles = ""
    last_smiles_in_prompt = smiles_prompt.rfind("[SMILES]")
    if last_smiles_in_prompt != -1:
        smiles_content_start = last_smiles_in_prompt + len("[SMILES]")
        smiles_end = smiles_prompt.find("[/SMILES]", smiles_content_start)
        if smiles_end != -1:
            canonical_smiles = smiles_prompt[smiles_content_start:smiles_end]
    
    mol_objects = []
    
    # Use current output if continuing, otherwise start with initial prompt
    prompt_to_use = current_output if current_output is not None else smiles_prompt
    
    # Tokenize the prompt
    tokenized = tokenizer(
        prompt_to_use,
        return_tensors="pt",
        padding=False,
    )
    input_ids = tokenized["input_ids"].to(model.device, non_blocking=True)
    attention_mask = tokenized["attention_mask"].to(model.device, non_blocking=True)
    
    # Create the sophisticated logit processor and stopping criteria
    logits_processor = ConformerControlLogitsProcessor(
        conformer_start_ids=conformer_start_ids,
        conformer_end_ids=conformer_end_ids,
        banned_token_ids=banned_ids,
        target_k=num_conformers,
        force_hard=True,
    )
    
    stopping_criteria = ConformerCountStoppingCriteria(
        conformer_end_ids=conformer_end_ids,
        target_k=num_conformers,
    )
    
    # Generate all conformers in one go
    start_time = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=2500 * num_conformers,  # Scale by number of conformers
            generation_config=gen_config,
            logits_processor=[logits_processor],
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        sequence = outputs.sequences[0].detach().cpu()
        del outputs
    
    elapsed = time.perf_counter() - start_time
    prompt_len = attention_mask.sum().item()
    seq_len = len(sequence)
    gen_len = max(0, seq_len - prompt_len)
    log_mfu(model, gen_len, elapsed)
    
    # Decode the output
    decoded_output = tokenizer.decode(sequence, skip_special_tokens=False)
    decoded_output = decoded_output.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "").replace(";", "")
    
    # Extract ALL generated conformers from the output
    conformer_strings = []
    idx = 0
    while True:
        # Find next [CONFORMER]
        conformer_start = decoded_output.find("[CONFORMER]", idx)
        if conformer_start == -1:
            break
        
        # Find corresponding [/CONFORMER]
        conformer_content_start = conformer_start + len("[CONFORMER]")
        conformer_end = decoded_output.find("[/CONFORMER]", conformer_content_start)
        
        if conformer_end == -1:
            stats["no_eos"] += 1
            if stats["no_eos"] < 20:
                logger.info(f"no conformer end tag found for conformer starting at {conformer_start}")
            break
        
        # Extract conformer content
        generated_conformer = decoded_output[conformer_content_start:conformer_end]
        conformer_strings.append(generated_conformer)
        
        # Move to next potential conformer
        idx = conformer_end + len("[/CONFORMER]")
    
    # Validate and decode each conformer
    for generated_conformer in conformer_strings:
        # Validate SMILES match
        generated_smiles = strip_smiles(generated_conformer)
        if not same_molecular_graph(canonical_smiles, generated_smiles):
            if stats["smiles_mismatch"] < 20:
                logger.info(f"smiles mismatch: \n{canonical_smiles=}\n{generated_smiles=}\n{generated_conformer=}")
            stats["smiles_mismatch"] += 1
        else:
            # Try to decode the conformer
            try:
                if binned:
                    mol_obj = decode_cartesian_binned_v2(generated_conformer, bins)
                else:
                    mol_obj = decode_cartesian_v2(generated_conformer)
                mol_objects.append(mol_obj)
            except Exception as e:
                if stats["mol_parse_fail"] < 20:
                    logger.info(f"mol parse fail: {e}\n{canonical_smiles=}\n{generated_conformer[:200]=}")
                stats["mol_parse_fail"] += 1
    
    logger.debug(f"Generated {len(mol_objects)} valid conformers out of {len(conformer_strings)} total conformers for {canonical_smiles}")
    
    return mol_objects, decoded_output


def run_multiconf_inference(inference_config: dict):
    """
    Main inference function that generates multiple conformers per SMILES.
    
    This function iterates through SMILES one by one, generating a specified number
    of conformers for each until reaching the target total number of conformers.
    """
    # Handle GPU assignment
    device_arg = inference_config.get("device", "cuda")
    target_device = device_arg
    
    if device_arg == "cuda":
        time.sleep(random.uniform(0.1, 4.0))
        logger.info("Searching for GPU...")
        free_gpu = find_free_gpu()
        
        if free_gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpu)
            target_device = "cuda:0"
            try:
                torch.cuda.set_device(0)
            except Exception:
                pass
            logger.info(f"Assigned physical GPU {free_gpu} as logical cuda:0")
            set_seed(42)
        else:
            logger.warning("No free GPU, using default 'cuda'")
            set_seed(42)
    
    results_path = os.path.join(
        inference_config["results_path"],
        datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + inference_config["run_name"]
    )
    os.makedirs(results_path, exist_ok=True)
    logger.add(os.path.join(results_path, "logs.txt"), rotation="50 MB")
    logger.info(inference_config)
    
    # Load model and tokenizer
    model, tokenizer = load_model_tokenizer(
        model_path=inference_config["model_path"],
        tokenizer_path=inference_config["tokenizer_path"],
        torch_dtype=inference_config["torch_dtype"],
        device=target_device
    )
    logger.info(f"model loaded: {model.dtype=}, {model.device=}")
    
    # Load test data
    with open(inference_config["test_data_path"], 'rb') as test_data_file:
        test_data = cloudpickle.load(test_data_file)
    
    # Build list of SMILES to process with their target conformer counts
    # Each entry is (geom_smiles, sub_smiles, target_conformer_count)
    smiles_to_process = []
    test_set: str = inference_config.get("test_set", "distinct")
    multiplier = inference_config.get("conformer_multiplier", 2)  # Generate 2x ground truths by default
    
    if test_set in ("clean"):
        for geom_smiles, data in test_data.items():
            num_ground_truths = data["num_confs"]
            target_count = num_ground_truths * multiplier
            smiles_to_process.append((geom_smiles, data['corrected_smi'], target_count))
    elif test_set in ("distinct", "xl", "qm9"):
        logger.info(f"Processing as {test_set} dataset")
        for geom_smiles, data in test_data.items():
            for sub_smiles, count in data["sub_smiles_counts"].items():
                target_count = count * multiplier
                smiles_to_process.append((geom_smiles, sub_smiles, target_count))
    
    total_conformers_to_generate = sum(count for _, _, count in smiles_to_process)
    logger.info(f"Total unique SMILES to process: {len(smiles_to_process)}")
    logger.info(f"Total conformers to generate: {total_conformers_to_generate}")
    
    # Apply limit if specified
    limit = inference_config.get("limit")
    if limit:
        smiles_to_process = smiles_to_process[:limit]
        total_conformers_to_generate = sum(count for _, _, count in smiles_to_process)
        logger.info(f"Limited to {len(smiles_to_process)} SMILES, {total_conformers_to_generate} conformers")
    
    # Get configuration parameters
    conformers_per_batch = inference_config.get("conformers_per_batch", 8)
    binned = inference_config.get("binned", False)
    
    logger.info(f"Conformers per batch: {conformers_per_batch}")
    
    if not binned and "binned" in str(inference_config["model_path"]):
        logger.info("Auto-detecting binned=True based on model path")
        binned = True
    
    # Initialize statistics and results
    stats = Counter({
        "smiles_mismatch": 0,
        "mol_parse_fail": 0,
        "no_eos": 0,
        "no_conformer_start": 0,
    })
    generations_all = defaultdict(list)
    total_conformers_generated = 0
    
    # Main loop: iterate through SMILES one by one, generating exact target count for each
    logger.info(f"Starting generation with {conformers_per_batch} conformers per batch")
    logger.info(f"Processing {len(smiles_to_process)} SMILES, total target: {total_conformers_to_generate} conformers")
    
    pbar = tqdm(total=total_conformers_to_generate, desc="Generating conformers")
    
    for geom_smiles, sub_smiles, target_count in smiles_to_process:
        smiles_prompt = f"[SMILES]{sub_smiles}[/SMILES]"
        
        logger.info(f"Processing {sub_smiles}: target {target_count} conformers")
        
        # Generate conformers in batches
        conformers_generated = 0
        current_output = None
        
        while conformers_generated < target_count:
            # Calculate how many to generate in this batch
            remaining = target_count - conformers_generated
            batch_size = min(conformers_per_batch, remaining)
            
            logger.debug(f"  Batch: generating {batch_size} conformers ({conformers_generated}/{target_count} so far)")
            
            # Generate batch
            mol_objects, current_output = generate_multiple_conformers(
                model=model,
                tokenizer=tokenizer,
                smiles_prompt=smiles_prompt,
                num_conformers=batch_size,
                gen_config=inference_config["gen_config"],
                binned=binned,
                stats=stats,
                geom_smiles=geom_smiles,
                current_output=current_output,
            )
            
            # Add to results
            if mol_objects:
                generations_all[geom_smiles].extend(mol_objects)
                num_generated = len(mol_objects)
                conformers_generated += num_generated
                total_conformers_generated += num_generated
                pbar.update(num_generated)
            else:
                logger.warning(f"  No valid conformers in batch, stopping generation for {sub_smiles}")
                break
        
        logger.info(f"Completed {sub_smiles}: generated {conformers_generated}/{target_count} valid conformers")
    
    pbar.close()
    
    logger.info(f"Generation complete. Total conformers: {total_conformers_generated}/{total_conformers_to_generate}")
    logger.info(f"Stats: {dict(stats)}")
    
    # Save results
    save_results(results_path, dict(generations_all), stats)
    
    return generations_all, stats


def launch_multiconf_inference_from_cli(
    device: str,
    grid_run_inference: bool = False,
    test_set: str = "distinct",
    xl: bool = False,
    qm9: bool = False,
    conformers_per_batch: int = 8,
    conformer_multiplier: int = 2,
    limit: Optional[int] = None,
    binned: bool = False,
    parallel_jobs: int = 1,
) -> None:
    """Launch multi-conformer inference from CLI arguments.
    
    Args:
        device: Device to run on
        grid_run_inference: Whether to run a grid of models
        test_set: Test dataset to use
        xl: Whether to run on XL dataset
        qm9: Whether to run on QM9 dataset
        conformers_per_batch: Number of conformers to generate per model.generate() call
        conformer_multiplier: Multiply ground truth count by this (e.g., 2 = generate 2x ground truths)
        limit: Limit number of SMILES to process
        binned: Whether to use binned decoding
        parallel_jobs: Number of parallel inference jobs for local execution
    """
    
    from molgen3D.config.sampling_config import gen_num_codes
    
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

    n_gpus = 1
    node = device if device in ["a100", "h100"] else "local"
    executor = None
    
    if device in ["a100", "h100"]:
        executor = submitit.AutoExecutor(folder="outputs/slurm_jobs/conf_gen/job_%j")
        executor.update_parameters(
            name="conf_gen_multiconf",
            timeout_min=24 * 24 * 60,
            gpus_per_node=n_gpus,
            nodes=1,
            mem_gb=80,
            cpus_per_task=n_gpus * 12,
            slurm_additional_parameters={"partition": node},
        )

    base_inference_config = {
        "model_path": get_ckpt("m600_qwen_pre_4seq_binned", "4e"),
        "tokenizer_path": get_tokenizer_path("qwen3_0.6b_custom"),
        "torch_dtype": "bfloat16",
        "gen_config": sampling_configs["top_p_sampling1"],
        "device": "cuda",
        "results_path": get_base_path("gen_results_root"),
        "run_name": "multiconf_qwen_pre_4e_grouped",
        "conformers_per_batch": conformers_per_batch,
        "conformer_multiplier": conformer_multiplier,
        "limit": limit,
        "binned": binned,
    }
    
    if grid_run_inference:
        param_grid = [
            ("qw600_pre_binned_paired", "1e"),
            ("qw600_pre_binned_paired", "2e"),
            ("qw600_pre_binned_paired", "3e"),
            ("qw600_pre_binned_paired", "4e"),
            ("qw600_pre_binned_paired", "5e"),
            ("qw600_pre_binned_grouped", "1e"),
            ("qw600_pre_binned_grouped", "2e"),
            ("qw600_pre_binned_grouped", "3e"),
            ("qw600_pre_binned_grouped", "4e"),
            ("qw600_pre_binned_grouped", "5e"),
        ]
        
        all_configs = []
        for model_key in param_grid:
            for test_set_name in test_sets_to_run:
                grid_config = dict(base_inference_config)
                
                if isinstance(model_key, tuple):
                    grid_config["model_path"] = get_ckpt(model_key[0], model_key[1])
                    model_key_str = f"{model_key[0]}_{model_key[1]}"
                else:
                    grid_config["model_path"] = get_ckpt(model_key)
                    model_key_str = model_key
                
                grid_config["test_data_path"] = get_data_path(f"{test_set_name}_smi")
                grid_config["test_set"] = test_set_name
                grid_config["run_name"] = f"multiconf_{model_key_str}_{test_set_name}"
                all_configs.append((grid_config, grid_config["run_name"]))
    else:
        all_configs = []
        for test_set_name in test_sets_to_run:
            inference_config = dict(base_inference_config)
            inference_config["test_data_path"] = get_data_path(f"{test_set_name}_smi")
            inference_config["test_set"] = test_set_name
            inference_config["run_name"] = f"multiconf_{conformer_multiplier}x_{conformers_per_batch}batch_{test_set_name}"
            all_configs.append((inference_config, inference_config["run_name"]))

    if executor is not None:
        with executor.batch():
            for config, _ in all_configs:
                logger.info(f"Submitting job for {config['run_name']}...")
                executor.submit(run_multiconf_inference, inference_config=config)
    else:
        if parallel_jobs <= 1:
            logger.info(f"Running {len(all_configs)} jobs locally (sequential)")
            for config, run_name in all_configs:
                logger.info(f"Running inference for {run_name}...")
                run_multiconf_inference(inference_config=config)
        else:
            max_workers = min(parallel_jobs, len(all_configs))
            logger.info(f"Running {len(all_configs)} jobs locally in parallel (max workers: {max_workers})")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_name = {
                    executor.submit(run_multiconf_inference, inference_config=config): name
                    for config, name in all_configs
                }
                for future in as_completed(future_to_name):
                    run_name = future_to_name[future]
                    try:
                        future.result()
                        logger.info(f"✓ Completed: {run_name}")
                    except Exception as e:
                        logger.error(f"✗ Exception in {run_name}: {e}")
                        import traceback
                        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-conformer inference with forced generation using ConformerControlLogitsProcessor")
    parser.add_argument("--device", type=str, choices=["local", "a100", "h100"], default="local")
    parser.add_argument("--grid_run_inference", action="store_true")
    parser.add_argument("--test_set", type=str, default="distinct")
    parser.add_argument("--xl", action="store_true")
    parser.add_argument("--qm9", action="store_true")
    parser.add_argument("--conformers_per_batch", type=int, default=8,
                        help="Number of conformers to generate per model.generate() call (default: 8)")
    parser.add_argument("--conformer_multiplier", type=int, default=2,
                        help="Generate this many times the ground truth count (e.g., 2 = 2x ground truths)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of unique SMILES to process")
    parser.add_argument("--binned", action="store_true", default=False)
    parser.add_argument("--parallel_jobs", type=int, default=1, help="Number of parallel inference jobs for local execution")
    
    args = parser.parse_args()
    
    launch_multiconf_inference_from_cli(
        device=args.device,
        grid_run_inference=args.grid_run_inference,
        test_set=args.test_set,
        xl=args.xl,
        qm9=args.qm9,
        conformers_per_batch=args.conformers_per_batch,
        conformer_multiplier=args.conformer_multiplier,
        limit=args.limit,
        binned=args.binned,
        parallel_jobs=args.parallel_jobs,
    )
