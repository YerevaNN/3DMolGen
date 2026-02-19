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
from molgen3D.data_processing.smiles_encoder_decoder import decode_cartesian_v2, strip_smiles, decode_cartesian_binned_v2, get_bins_for_coords
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

torch.set_grad_enabled(False)

# Global for NVML state
NVML_AVAILABLE = None


def _run_from_config_file(config_path: str):
    """
    Minimal wrapper function for submitit that loads config from file and runs inference.
    This avoids pickling the main function and its dependencies.
    """
    import json
    with open(config_path, 'r') as f:
        inference_config = json.load(f)
    # Import here to avoid module-level CUDA state
    from molgen3D.evaluation.inference import run_inference
    return run_inference(inference_config)


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

    # Ensure deterministic behavior
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

def process_batch(model, tokenizer, batch: list[list], gen_config, eos_token_id, binned: bool):
    # Create bins for binned decoding (must match encoding bins)
    bins = None
    if binned:
        ranges = [(-13.0, 13.0), (-13.0, 13.0), (-13.0, 13.0)]
        bins = get_bins_for_coords(ranges, bin_size=0.104)
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
                    if binned:
                        mol_obj = decode_cartesian_binned_v2(generated_conformer, bins)
                    else:
                        mol_obj = decode_cartesian_v2(generated_conformer)
                    generations[geom_smiles].append(mol_obj)
                except Exception as e:
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

def run_inference(inference_config: dict):
    """Main inference function that handles GPU assignment and model execution."""
    # Convert gen_config from dict to GenerationConfig if needed (for submitit pickling)
    from transformers import GenerationConfig
    if isinstance(inference_config.get("gen_config"), dict):
        inference_config["gen_config"] = GenerationConfig.from_dict(inference_config["gen_config"])

    # Handle GPU assignment if using CUDA
    device_arg = inference_config.get("device", "cuda")
    target_device = device_arg
    
    if device_arg == "cuda" or device_arg.startswith("cuda:"):
        # Add small random delay to reduce race conditions
        time.sleep(random.uniform(0.1, 4.0))
        logger.info(f"Searching for GPU ({device_arg})...")
        
        # Check for specific GPU request (e.g., "cuda:8")
        requested_gpu = None
        if ":" in device_arg:
            try:
                requested_gpu = int(device_arg.split(":")[1])
            except (ValueError, IndexError):
                pass

        free_gpu = find_free_gpu(requested_gpu=requested_gpu)
        
        if free_gpu is not None:
            # Set CUDA_VISIBLE_DEVICES to isolate this process to the chosen physical GPU.
            # Physical GPU 'free_gpu' becomes logical GPU 'cuda:0' for this process.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpu)
            target_device = "cuda:0"
            try:
                # After setting CUDA_VISIBLE_DEVICES, we must use index 0.
                torch.cuda.set_device(0)
            except Exception:
                pass
            logger.info(f"Assigned physical GPU {free_gpu} as logical cuda:0")
            set_seed(42)
        else:
            logger.warning("No free GPU, using default 'cuda'")
            set_seed(42)
    
    results_path = os.path.join(
        *[inference_config["results_path"],
          datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + inference_config["run_name"]]
    )
    os.makedirs(results_path, exist_ok=True)
    logger.add(os.path.join(results_path, "logs.txt"), rotation="50 MB")
    logger.info(inference_config)

    model, tokenizer = load_model_tokenizer(
        model_path=inference_config["model_path"],
        tokenizer_path=inference_config["tokenizer_path"],
        torch_dtype=inference_config["torch_dtype"],
        attention_imp=inference_config["attention_imp"],
        device=inference_config["device"],
    )
    logger.info(f"model loaded: {model.dtype=}, {model.device=}")
    
    # Use [/CONFORMER] as the primary stop token, falling back to <|endoftext|>
    eos_token_id = tokenizer.convert_tokens_to_ids("[/CONFORMER]")
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    
    logger.info(f"Using eos_token_id: {eos_token_id} for generation")
    
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
    elif test_set == "valid":
        logger.info("Processing as validation dataset")
        # Validation set format: {smiles: [mol_obj1, mol_obj2, ...]}
        for geom_smiles, mol_list in test_data.items():
            if isinstance(mol_list, list):
                # Each entry has a list of ground truth conformers
                num_ground_truths = len(mol_list)
                # Generate 2x the ground truth count
                mols_list.extend([(geom_smiles, f"[SMILES]{geom_smiles}[/SMILES]")] * num_ground_truths * 2)
            else:
                logger.warning(f"Unexpected data format for {geom_smiles}, skipping")
    elif test_set == "icl":
        logger.info("Processing as icl dataset")
        for geom_smiles, data in test_data.items():
            icl_prompt = data.get('icl_prompt')
            if icl_prompt:
                mols_list.extend([(geom_smiles, icl_prompt)] * data.get("num_confs", 1) * 2)
    logger.info(f"mols_list length: {len(mols_list)}, mols_list_distinct: {len(set(mols_list))}, mols_list: {mols_list[:10]}")

    mols_list.sort(key=lambda x: len(x[0]))
    
    limit = inference_config.get("limit")
    mols_list = mols_list[:limit]
    stats = Counter({"smiles_mismatch": 0, "mol_parse_fail": 0, "no_eos": 0})
    batch_size = int(inference_config["batch_size"])
    generations_all = defaultdict(list)

    binned = inference_config.get("binned", False)
    if not binned and "binned" in str(inference_config["model_path"]):
        logger.info("Auto-detecting binned=True based on model path")
        binned = True

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
            )
            stats.update(stats_)
            for k, v in outputs.items():
                generations_all[k].extend(v)

    save_results(results_path, dict(generations_all), stats)

    return generations_all, stats


def launch_inference_from_cli(
    device: str,
    grid_run_inference: bool,
    test_set: str = None,
    xl: bool = False,
    qm9: bool = False,
    valid: bool = False,
    limit: int = None,
    binned: bool = False,
    icl: bool = False,
    icl_n: int = 5,
    parallel_jobs: int = 1
) -> None:
    # Determine which test sets to run
    test_sets_to_run = []
    if test_set:
        test_sets_to_run.append(test_set)
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
    
    n_gpus = 1
    executor = None

    if device in ["a100", "h100", "all"]:
        executor = submitit.AutoExecutor(folder="outputs/slurm_jobs/conf_gen/job_%j")
    elif device == "local":
        executor = submitit.LocalExecutor(folder="outputs/slurm_jobs/conf_gen/job_%j")
    
    if executor is not None:
        executor.update_parameters(
            name="conf_gen",
            timeout_min=24 * 24 * 60,
            gpus_per_node=n_gpus,
            nodes=1,
            mem_gb=80,
            cpus_per_task=n_gpus * 12,
            slurm_additional_parameters={"partition": device},
            slurm_use_srun=False,  # Don't use srun to avoid MPI issues
        )
        logger.info(f"✓ Submitit executor configured for partition: {device}")
    
    # Base configuration template
    base_inference_config = {
        "model_path": str(get_ckpt("m600_qwen_pre_4seq_binned", "4e")),  # Convert Path to str for JSON
        "tokenizer_path": str(get_tokenizer_path("qwen3_0.6b_binned" if binned else "qwen3_0.6b_custom")),
        "torch_dtype": "bfloat16",
        "batch_size": 256,
        "num_gens": gen_num_codes["2k_per_conf"],
        "gen_config": sampling_configs["top_p_sampling1"].to_dict(),  # Convert to dict for JSON
        "device": "cuda",
        "results_path": str(get_base_path("gen_results_root")),  # Convert Path to str for JSON
        "run_name": "qwen_pre_5e_grouped",
        "limit": limit,
        "binned": binned,
    }

    if grid_run_inference:
        param_grid = [
            ("qw600_pre_binned_grouped", "1e"),
            ("qw600_pre_binned_grouped", "2e"),
            ("qw600_pre_binned_grouped", "3e"),
            ("qw600_pre_binned_grouped", "4e"),
            ("qw600_pre_binned_grouped", "5e"),

        ]
        
        ]
        jobs = []
        if executor is not None:
            # Use config file submission to avoid pickling complex objects
            import json
            with executor.batch():
                for model_path in param_grid:
                    for test_set_name in test_sets_to_run:
                        grid_config = dict(base_inference_config)

                        if isinstance(model_key, tuple):
                            grid_config["model_path"] = str(get_ckpt(model_key[0], model_key[1]))
                            model_key_str = f"{model_key[0]}_{model_key[1]}"
                            # Auto-select tokenizer based on model name
                            if "binned" in model_key[0] or "qw600" in model_key[0]:
                                grid_config["tokenizer_path"] = str(get_tokenizer_path("qwen3_0.6b_binned"))
                                grid_config["binned"] = True
                            else:
                                grid_config["tokenizer_path"] = str(get_tokenizer_path("qwen3_0.6b_custom"))
                                grid_config["binned"] = False
                        else:
                            grid_config["model_path"] = str(get_ckpt(model_key))
                            model_key_str = model_key
                            if "binned" in model_key or "qw600" in model_key:
                                grid_config["tokenizer_path"] = str(get_tokenizer_path("qwen3_0.6b_binned"))
                                grid_config["binned"] = True
                            else:
                                grid_config["tokenizer_path"] = str(get_tokenizer_path("qwen3_0.6b_custom"))
                                grid_config["binned"] = False

                        if test_set_name == "xl":
                            grid_config["batch_size"] = 100

                        if test_set_name == "qm9":
                            grid_config["batch_size"] = 100

                        if test_set_name == "valid":
                            grid_config["batch_size"] = 128

                        if test_set_name == "icl":
                            grid_config["batch_size"] = 64

                        # Handle validation set path differently
                        if test_set_name == "valid":
                            grid_config["test_data_path"] = str(get_data_path("validation_pickle"))
                        else:
                            grid_config["test_data_path"] = str(get_data_path(f"{test_set_name}_smi"))
                        grid_config["test_set"] = test_set_name
                        grid_config["run_name"] = f"{model_key_str}_{test_set_name}"

                        logger.info(f"Submitting job for {grid_config['run_name']}...")

                        # Save config to a file
                        config_dir = os.path.join("outputs", "slurm_jobs", "conf_gen", "configs")
                        os.makedirs(config_dir, exist_ok=True)
                        config_file = os.path.join(config_dir, f"{grid_config['run_name']}_config.json")
                        with open(config_file, 'w') as f:
                            json.dump(grid_config, f, indent=2)

                        logger.info(f"  Config saved to: {config_file}")

                        # Submit the minimal wrapper function with just the config path
                        job = executor.submit(_run_from_config_file, config_file)
                        jobs.append(job)
        else:
            # Run grid search locally - either sequential or parallel
            # Build all configs first
            all_configs = []
            for model_key in param_grid:
                for test_set_name in test_sets_to_run:
                    grid_config = dict(base_inference_config)
                    
                    if isinstance(model_key, tuple):
                        grid_config["model_path"] = str(get_ckpt(model_key[0], model_key[1]))
                        model_key_str = f"{model_key[0]}_{model_key[1]}"
                        # Auto-select tokenizer based on model name
                        if "binned" in model_key[0] or "qw600" in model_key[0]:
                            grid_config["tokenizer_path"] = str(get_tokenizer_path("qwen3_0.6b_binned"))
                            grid_config["binned"] = True
                        else:
                            grid_config["tokenizer_path"] = str(get_tokenizer_path("qwen3_0.6b_custom"))
                            grid_config["binned"] = False
                    else:
                        grid_config["model_path"] = str(get_ckpt(model_key))
                        model_key_str = model_key
                        if "binned" in model_key or "qw600" in model_key:
                            grid_config["tokenizer_path"] = str(get_tokenizer_path("qwen3_0.6b_binned"))
                            grid_config["binned"] = True
                        else:
                            grid_config["tokenizer_path"] = str(get_tokenizer_path("qwen3_0.6b_custom"))
                            grid_config["binned"] = False
                    
                    if test_set_name == "xl":
                        grid_config["batch_size"] = 100

                    if test_set_name == "qm9":
                        grid_config["batch_size"] = 100

                    if test_set_name == "valid":
                        grid_config["batch_size"] = 128

                    if test_set_name == "icl":
                        grid_config["batch_size"] = 64

                    # Handle validation set path differently
                    if test_set_name == "valid":
                        grid_config["test_data_path"] = str(get_data_path("validation_pickle"))
                    else:
                        grid_config["test_data_path"] = str(get_data_path(f"{test_set_name}_smi"))
                    grid_config["test_set"] = test_set_name
                    grid_config["run_name"] = f"{model_key_str}_{test_set_name}"
                    all_configs.append((grid_config, grid_config["run_name"]))
            
            if parallel_jobs <= 1:
                # Sequential execution
                logger.info(f"Running {len(all_configs)} grid inference jobs locally (sequential)")
                for grid_config, run_name in all_configs:
                    logger.info(f"Running grid inference for {run_name}...")
                    run_inference(inference_config=grid_config)
            else:
                # Parallel execution
                max_workers = min(parallel_jobs, len(all_configs))
                logger.info(f"Running {len(all_configs)} grid inference jobs locally in parallel (max workers: {max_workers})")
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_name = {
                        executor.submit(run_inference, inference_config=config): name
                        for config, name in all_configs
                    }
                    
                    for future in as_completed(future_to_name):
                        run_name = future_to_name[future]
                        try:
                            result = future.result()
                            logger.info(f"✓ Completed: {run_name}")
                        except Exception as e:
                            logger.error(f"✗ Exception in {run_name}: {e}")
                            import traceback
                            traceback.print_exc()
    else:
        if executor is not None:
            # Use config file submission to avoid pickling complex objects
            import json
            with executor.batch():
                for test_set_name in test_sets_to_run:
                    inference_config = dict(base_inference_config)

                    if test_set_name == "xl":
                        inference_config["batch_size"] = 100
                    if test_set_name == "qm9":
                        inference_config["batch_size"] = 100
                    if test_set_name == "valid":
                        inference_config["batch_size"] = 128
                    if test_set_name == "icl":
                        inference_config["batch_size"] = 64

                    # Handle validation set path differently
                    if test_set_name == "valid":
                        inference_config["test_data_path"] = str(get_data_path("validation_pickle"))
                    else:
                        inference_config["test_data_path"] = str(get_data_path(f"{test_set_name}_smi"))
                    inference_config["test_set"] = test_set_name
                    inference_config["run_name"] = f"new_data_p1_{test_set_name}"

                    logger.info(f"Submitting job for {inference_config['run_name']}...")

                    # Save config to a file
                    config_dir = os.path.join("outputs", "slurm_jobs", "conf_gen", "configs")
                    os.makedirs(config_dir, exist_ok=True)
                    config_file = os.path.join(config_dir, f"{inference_config['run_name']}_config.json")
                    with open(config_file, 'w') as f:
                        json.dump(inference_config, f, indent=2)

                    logger.info(f"  Config saved to: {config_file}")

                    # Submit the minimal wrapper function with just the config path
                    job = executor.submit(_run_from_config_file, config_file)
        else:
            for test_set_name in test_sets_to_run:
                inference_config = dict(base_inference_config)

                if test_set_name == "xl":
                    inference_config["batch_size"] = 100
                if test_set_name == "qm9":
                    inference_config["batch_size"] = 100
                if test_set_name == "valid":
                    inference_config["batch_size"] = 128
                if test_set_name == "icl":
                    inference_config["batch_size"] = 64

                # Handle validation set path differently
                if test_set_name == "valid":
                    inference_config["test_data_path"] = str(get_data_path("validation_pickle"))
                else:
                    inference_config["test_data_path"] = str(get_data_path(f"{test_set_name}_smi"))
                inference_config["test_set"] = test_set_name
                inference_config["run_name"] = f"new_data_p1_{test_set_name}"

                logger.info(f"Running inference for {test_set_name} with config: {inference_config}")
                run_inference(inference_config=inference_config)

if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="local",
                        choices=["local", "a100", "h100", "all"],
                        help="Device to run on: local (current machine), a100/h100/all (cluster partitions)")
    parser.add_argument("--grid_run_inference", action="store_true")
    parser.add_argument("--test_set", type=str, choices=["clean", "distinct", "corrected"], default=None)
    parser.add_argument("--binned", action="store_true", default=False)
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
        icl=args.icl,
        icl_n=args.icl_n,
        parallel_jobs=args.parallel_jobs
    )
