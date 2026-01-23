import os
import argparse
from datetime import datetime
import time
import random

# Global for NVML state
NVML_AVAILABLE = None


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
    import random
    import torch
    
    random.seed(seed)  # Python random module
    torch.manual_seed(seed)  # PyTorch CPU
    
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        try:
            torch.cuda.manual_seed(seed)  # PyTorch GPU
            torch.cuda.manual_seed_all(seed)  # All GPUs (if using multi-GPU)
            
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def get_gpu_status():
    """Get status of all available GPUs including memory usage."""
    from loguru import logger
    
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
    """Find and lock a free GPU with sufficient memory and low usage.
    
    Returns the GPU index if found and locked, None otherwise.
    """
    from loguru import logger
    global _GPU_LOCK_FILE
    
    if not init_nvml():
        return None
    
    import fcntl
    
    try:
        gpu_status = get_gpu_status()
        if not gpu_status:
            return None
        
        # Sort by free memory (most free first)
        gpu_status.sort(key=lambda x: x["free_memory_gb"], reverse=True)
        
        for gpu in gpu_status:
            idx = gpu["gpu_idx"]
            
            # Check if GPU meets requirements
            if gpu["free_memory_gb"] < min_memory_gb or gpu["usage_percent"] > max_memory_usage_percent:
                continue
            
            # Try to acquire lock for this GPU
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
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from loguru import logger
    from molgen3D.evaluation.utils import (
        log_cuda_memory,
        log_cuda_summary,
        estimate_decoder_flops_per_token,
        detect_peak_flops,
    )
    
    # Configure CUDA settings if available
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
    """Save generation results to pickle and text files."""
    import cloudpickle
    
    with open(os.path.join(results_path, "generation_results.pickle"), 'wb') as results_file_pickle:
        cloudpickle.dump(generations, results_file_pickle, protocol=4)
    
    with open(os.path.join(results_path, "generation_results.txt"), 'w') as results_file_txt:
        results_file_txt.write(f"{stats=}")


def process_batch(model, tokenizer, batch: list[list], gen_config, eos_token_id, binned: bool):
    """Process a batch of molecules and generate conformers."""
    import torch
    import time
    from collections import defaultdict
    from loguru import logger
    from molgen3D.data_processing.smiles_encoder_decoder import (
        strip_smiles,
        decode_cartesian_v2,
        decode_cartesian_binned,
        get_bins_for_coords
    )
    from molgen3D.evaluation.utils import same_molecular_graph, log_cuda_memory, log_mfu
    
    # Create bins for binned decoding (must match encoding bins)
    bins = None
    if binned:
        ranges = [(-13.0, 13.0), (-13.0, 13.0), (-13.0, 13.0)]
        bins = get_bins_for_coords(ranges, bin_size=0.104)
    
    generations = defaultdict(list)
    stats = {"smiles_mismatch": 0, "mol_parse_fail": 0, "no_eos": 0}
    
    # Extract prompts and geom_smiles from batch
    prompts = [item[1] for item in batch]
    geom_smiles_list = [item[0] for item in batch]
    
    tokenized_prompts = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        pad_to_multiple_of=8
    )
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
    decoded_outputs = tokenizer.batch_decode(sequences, skip_special_tokens=False)
    for i, out in enumerate(decoded_outputs):
        out_clean = out.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "")

        # Robust extraction for both standard and ICL prompts
        # 1. Get the target SMILES from the prompt part to be safe
        prompt = prompts[i]
        canonical_smiles = ""
        last_smiles_in_prompt = prompt.rfind("[SMILES]")
        if last_smiles_in_prompt != -1:
            smiles_content_start = last_smiles_in_prompt + len("[SMILES]")
            smiles_end = prompt.find("[/SMILES]", smiles_content_start)
            if smiles_end != -1:
                canonical_smiles = prompt[smiles_content_start:smiles_end]
        
        # 2. Extract the generated conformer from the full output
        # It should be between the LAST [CONFORMER] and the next [/CONFORMER]
        generated_conformer = ""
        last_conformer_start = out_clean.rfind("[CONFORMER]")
        if last_conformer_start != -1:
            conformer_content_start = last_conformer_start + len("[CONFORMER]")
            conformer_end = out_clean.find("[/CONFORMER]", conformer_content_start)
            if conformer_end != -1:
                generated_conformer = out_clean[conformer_content_start:conformer_end]
        
        geom_smiles = geom_smiles_list[i]
        
        if generated_conformer:
            generated_smiles = strip_smiles(generated_conformer)
            if not same_molecular_graph(canonical_smiles, generated_smiles):
                if stats["smiles_mismatch"] < 20: # Log first few mismatches in detail
                    logger.info(f"smiles mismatch: \n{canonical_smiles=}\n{generated_smiles=}\n{generated_conformer=}\nFull output snippet: {out_clean[-500:]}")
                stats["smiles_mismatch"] += 1
            else:
                try:
                    if binned:
                        mol_obj = decode_cartesian_binned(generated_conformer, bins)
                    else:
                        mol_obj = decode_cartesian_v2(generated_conformer)
                    generations[geom_smiles].append(mol_obj)
                except Exception as e:
                    if stats["mol_parse_fail"] < 20:
                        logger.info(f"smiles fails parsing: {e}\n{canonical_smiles=}\n{generated_smiles=}\n{generated_conformer=}")
                    stats["mol_parse_fail"] += 1
        else:
            stats["no_eos"] += 1
            if stats["no_eos"] < 20:
                logger.info(f"no eos: \n{out_clean[:500]=} ... {out_clean[-500:]=}")
    
    return generations, stats


def split_batch_on_geom_size(batch: list[list], max_geom_len: int = 80) -> list[list]:
    """Split batch if any geometry SMILES is too long."""
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
    import torch
    import cloudpickle
    from tqdm import tqdm
    from collections import defaultdict, Counter
    from loguru import logger
    
    torch.set_grad_enabled(False)
    
    # Handle GPU assignment if using CUDA
    device_arg = inference_config.get("device", "cuda")
    target_device = device_arg
    
    if device_arg == "cuda":
        # Add small random delay to reduce race conditions
        time.sleep(random.uniform(0.1, 4.0))
        logger.info("Searching for GPU...")
        free_gpu = find_free_gpu()
        
        if free_gpu is not None:
            # Set CUDA_VISIBLE_DEVICES to isolate this process to the chosen physical GPU.
            # Physical GPU 'free_gpu' becomes logical GPU 'cuda:0' for this process.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpu)
            target_device = "cuda:0"
            try:
                import torch.cuda
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
        device=target_device
    )
    logger.info(f"model loaded: {model.dtype=}, {model.device=}")
    
    # Use [/CONFORMER] as the primary stop token, falling back to <|endoftext|>
    eos_token_id = tokenizer.convert_tokens_to_ids("[/CONFORMER]")
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    
    logger.info(f"Using eos_token_id: {eos_token_id} for generation")
    
    with open(inference_config["test_data_path"], 'rb') as test_data_file:
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
                binned=binned
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
    limit: int = None,
    binned: bool = False,
    icl: bool = False,
    icl_n: int = 5
) -> None:
    """Launch inference jobs via CLI arguments."""
    from loguru import logger
    import submitit
    from molgen3D.config.paths import get_ckpt, get_tokenizer_path, get_data_path, get_base_path
    from molgen3D.config.sampling_config import sampling_configs, gen_num_codes
    
    # Determine which test sets to run
    test_sets_to_run = []
    if test_set:
        test_sets_to_run.append(test_set)
    if xl:
        test_sets_to_run.append("xl")
    if qm9:
        test_sets_to_run.append("qm9")
    if icl:
        test_sets_to_run.append(f"icl_{icl_n}")
    
    if not test_sets_to_run:
        logger.info("No test sets specified. Skipping inference.")
        return
    
    n_gpus = 1
    node = device if device in ["a100", "h100"] else "local"
    executor = None
    
    if device in ["a100", "h100"]:
        executor = submitit.AutoExecutor(folder="outputs/slurm_jobs/conf_gen/job_%j")
    elif device == "local":
        executor = submitit.LocalExecutor(folder="outputs/slurm_jobs/conf_gen/job_%j")
    
    executor.update_parameters(
        name="conf_gen",
        timeout_min=24 * 24 * 60,
        gpus_per_node=n_gpus,
        nodes=1,
        mem_gb=80,
        cpus_per_task=n_gpus * 12,
        slurm_additional_parameters={"partition": node},
    )
    
    # Base configuration template
    base_inference_config = {
        "model_path": get_ckpt("m600_qwen_pre_4seq_binned", "4e"),
        "tokenizer_path": get_tokenizer_path("qwen3_0.6b_binned"),
        "torch_dtype": "bfloat16",
        "batch_size": 128,
        "num_gens": gen_num_codes["2k_per_conf"],
        "gen_config": sampling_configs["top_p_sampling1"],
        "device": "cuda",
        "results_path": get_base_path("gen_results_root"),
        "run_name": "qwen_pre_4seq_binned",
        "limit": limit,
        "binned": binned,
    }
    
    if grid_run_inference:
        jobs = []
        param_grid = [
            ("m600_qwen_pre_4seq_binned", "1e"),
            ("m600_qwen_pre_4seq_binned", "2e"),
            ("m600_qwen_pre_4seq_binned", "3e"),
            ("m600_qwen_pre_4seq_binned", "4e")
        ]
        
        if executor is not None:
            with executor.batch():
                for model_key in param_grid:
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
                        
                        if test_set_name == "icl":
                            grid_config["batch_size"] = 64
                        
                        grid_config["test_data_path"] = get_data_path(f"{test_set_name}_smi")
                        grid_config["test_set"] = test_set_name
                        grid_config["run_name"] = f"{model_key_str}_{test_set_name}"
                        
                        logger.info(f"Submitting job for {grid_config['run_name']}...")
                        job = executor.submit(run_inference, inference_config=grid_config)
                        jobs.append(job)
    else:
        if executor is not None:
            with executor.batch():
                for test_set_name in test_sets_to_run:
                    inference_config = dict(base_inference_config)
                    
                    if test_set_name == "xl":
                        inference_config["batch_size"] = 100
                    if test_set_name == "qm9":
                        inference_config["batch_size"] = 100
                    if test_set_name == "icl":
                        inference_config["batch_size"] = 64
                    
                    inference_config["test_data_path"] = get_data_path(f"{test_set_name}_smi")
                    inference_config["test_set"] = test_set_name
                    inference_config["run_name"] = f"new_data_p1_{test_set_name}"
                    
                    logger.info(f"Running inference for {test_set_name} with config: {inference_config}")
                    job = executor.submit(run_inference, inference_config=inference_config)
        else:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["local", "a100", "h100"], required=True)
    parser.add_argument("--grid_run_inference", action="store_true")
    parser.add_argument("--test_set", type=str, choices=["clean", "distinct", "corrected"], default=None)
    parser.add_argument("--binned", action="store_true", default=False)
    parser.add_argument("--xl", action="store_true")
    parser.add_argument("--qm9", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--icl", action="store_true")
    parser.add_argument("--icl_n", type=int, default=5)
    args = parser.parse_args()
    launch_inference_from_cli(
        device=args.device,
        grid_run_inference=args.grid_run_inference,
        test_set=args.test_set,
        xl=args.xl,
        qm9=args.qm9,
        limit=args.limit,
        binned=args.binned,
        icl=args.icl,
        icl_n=args.icl_n
    )
