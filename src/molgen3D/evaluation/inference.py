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
    results_path = os.path.join(*[inference_config["results_path"], 
                                  datetime.now().strftime('%Y%m%d_%H%M%S') + 
                                  '_' + inference_config["run_name"]])
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
    if icl:
        test_sets_to_run.append(f"icl_{icl_n}")
    
    if not test_sets_to_run:
        logger.info("No test sets specified. Skipping inference.")
        return
    
    n_gpus = 1
    node = device if device in ["a100", "h100", "all"] else "local"
    executor = None
    
    # Set SLURM_CONF if not already set (needed for sbatch to work)
    if device in ["a100", "h100", "all"]:
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
            slurm_additional_parameters={"partition": node},
        )
        # Disable srun to avoid PMIx/MPI plugin issues and srun command not found errors
        if device in ["a100", "h100", "all"]:
            logger.info("Disabling srun to avoid PMIx/MPI plugin issues.")
            executor.update_parameters(slurm_use_srun=False)
    
    # Base configuration template
    base_inference_config = {
        "model_path": get_ckpt("qw600_pre_binned_filtered", "1e"),
        "tokenizer_path": get_tokenizer_path("qwen3_0.6b_binned"),
        "torch_dtype": "bfloat16",
        "batch_size": 256,
        "num_gens": gen_num_codes["2k_per_conf"],
        "gen_config": sampling_configs["top_p_sampling1"],
        "device": "cuda",
        "results_path": get_base_path("gen_results_root"),
        "run_name": "qwen_pre_binned",
        "limit": limit,
        "binned": binned,
    }

    if grid_run_inference:
        param_grid = [
            "/data/chem-project/checkpoints/qwen3_06b/260126-0653-7cf8-qwen3_06b_pre_5e_8e-4_binned_grouped/step-27500-hf",
            # "/home/chem-project/checkpoints/conf_grpo/260205-0417_qwen3_fscore_lr8e-05_d075_b10.0_g2.0_w0_warmup0_tau0_noscale_7cf8model_fbeta/model",
            # "/home/chem-project/checkpoints/conf_grpo/260204-1141_qwen3_fscore_lr8e-05_d075_b2.0_g2.0_w0_warmup0_tau0_noscale_7cf8model_fbeta/model",
            # "/home/chem-project/checkpoints/conf_grpo/260204-1143_qwen3_fscore_lr8e-05_d075_b0.0_g2.0_w0_warmup0_tau0_noscale_7cf8model_valid_count/model",
            # "/home/chem-project/checkpoints/conf_grpo/260204-1143_qwen3_fscore_lr8e-05_d075_b0.0_g2.0_w0_warmup0_tau0_noscale_7cf8model_valid_count/model",
        
        ]
        jobs = []
        if executor is not None:
            with executor.batch():
                for model_path in param_grid:
                    for test_set_name in test_sets_to_run:
                        grid_config = dict(base_inference_config)
                        # Use direct path (model_path is already a full path string)
                        grid_config["model_path"] = model_path
                        
                        # Auto-select tokenizer based on model path
                        if "binned" in model_path or "qwen3" in model_path:
                            grid_config["tokenizer_path"] = get_tokenizer_path("qwen3_0.6b_binned")
                            grid_config["binned"] = True
                        else:
                            grid_config["tokenizer_path"] = get_tokenizer_path("qwen3_0.6b_custom")
                            grid_config["binned"] = False
                        
                        if test_set_name == "xl":
                            grid_config["batch_size"] = 100
                        
                        if test_set_name == "qm9":
                            grid_config["batch_size"] = 100

                        grid_config["test_data_path"] = get_data_path(f"{test_set_name}_smi")
                        grid_config["test_set"] = test_set_name
                        # Create a cleaner run name from the model path
                        # Extract the last meaningful directory name or use basename
                        path_parts = model_path.rstrip('/').split('/')
                        if path_parts[-1] == "model":
                            model_name = path_parts[-2]  # Use parent directory if path ends with /model
                        else:
                            model_name = path_parts[-1]  # Use last directory/file name
                        grid_config["run_name"] = f"{model_name}_{test_set_name}"
                        
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
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["local", "a100", "h100", "all"], default="local")
    parser.add_argument("--grid_run_inference", action="store_true")
    parser.add_argument("--test_set", type=str, choices=["clean", "distinct", "corrected"], default=None)
    parser.add_argument("--binned", action="store_true", default=False)
    parser.add_argument("--xl", action="store_true")
    parser.add_argument("--qm9", action="store_true")
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
        limit=args.limit,
        binned=args.binned,
        icl=args.icl,
        icl_n=args.icl_n,
        parallel_jobs=args.parallel_jobs
    )
