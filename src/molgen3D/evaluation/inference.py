import os
# Reduce CUDA memory fragmentation for large batch inference
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
import torch
import cloudpickle
import random
from transformers.generation.utils import GenerateDecoderOnlyOutput
import yaml
import itertools
import re
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from loguru import logger
from collections import defaultdict, Counter
import submitit
import argparse
from datetime import datetime

# from utils import parse_molecule_with_coordinates
from molgen3D.data_processing.smiles_encoder_decoder import decode_cartesian_v2, strip_smiles
from molgen3D.evaluation.utils import extract_between, same_molecular_graph
from molgen3D.evaluation.qwen_allowlist_logit_processor import (
    QwenAllowlistLogitsProcessor,
    build_templates_for_batch,
    build_precomputed_template,
)
from molgen3D.config.paths import get_ckpt, get_tokenizer_path, get_data_path, get_base_path
from molgen3D.config.sampling_config import sampling_configs, gen_num_codes


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_model_tokenizer(model_path, tokenizer_path, torch_dtype="bfloat16", attention_imp="flash_attention_2", device="auto"):
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), padding_side='left', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path),
                                                 torch_dtype=getattr(torch, torch_dtype),
                                                 attn_implementation=attention_imp,
                                                 device_map=device,
                                                 trust_remote_code=True,
                                                 local_files_only=True).eval()
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Log model info
    logger.info(f"Model loaded: dtype={model.dtype}, device={model.device}")
    logger.info(f"Attention implementation: {attention_imp}")

    # Log GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    return model, tokenizer

def save_results(results_path, generations, stats):
    with open(os.path.join(results_path, "generation_results.pickle"), 'wb') as results_file_pickle:
        cloudpickle.dump(generations, results_file_pickle, protocol=4)
    
    with open(os.path.join(results_path, "generation_results.txt"), 'w') as results_file_txt:
        results_file_txt.write(f"{stats=}")

def _build_template_worker(args):
    """Worker for parallel template building."""
    smi, tokenizer = args
    return build_precomputed_template(smi, tokenizer, use_tokenizer_cache=True)


def process_batch(
    model,
    tokenizer,
    batch: list[list],
    gen_config,
    eos_token_id,
    use_logit_processor: bool = False,
    template_executor: ThreadPoolExecutor | None = None,
):
    """Process a batch of molecules for conformer generation.

    Returns:
        generations: dict mapping geom_smiles to list of mol objects
        stats: dict with error counts
        batch_time: time taken for this batch (seconds)
    """
    batch_start = time.perf_counter()
    generations = defaultdict(list)
    stats = {"smiles_mismatch": 0, "mol_parse_fail": 0, "no_eos": 0, "success": 0}

    # Extract prompts and geom_smiles from batch
    prompts = [item[1] for item in batch]
    geom_smiles_list = [item[0] for item in batch]
    smiles_list = []
    for p in prompts:
        smi = extract_between(p, "[SMILES]", "[/SMILES]")
        if smi is None:
            raise ValueError(f"Prompt is missing SMILES tags: {p}")
        smiles_list.append(smi)

    tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True)
    tokenized_prompts = {k: v.to(model.device, non_blocking=True) for k, v in tokenized_prompts.items()}

    # Optionally build logits processor for constrained generation
    logits_processor = None
    if use_logit_processor:
        # Build templates with tokenizer cache; use parallel if executor provided
        if template_executor is not None and len(smiles_list) > 1:
            templates = list(template_executor.map(
                _build_template_worker,
                [(smi, tokenizer) for smi in smiles_list]
            ))
        else:
            templates = build_templates_for_batch(smiles_list, tokenizer, use_tokenizer_cache=True)

        prompt_lengths = [int(mask.sum().item()) for mask in tokenized_prompts["attention_mask"]]
        logits_processor = LogitsProcessorList(
            [QwenAllowlistLogitsProcessor(templates, prompt_lengths, tokenizer=tokenizer, eos_token_id=eos_token_id)]
        )

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=tokenized_prompts["input_ids"],
            attention_mask=tokenized_prompts["attention_mask"],
            max_new_tokens=4000,
            eos_token_id=eos_token_id,
            generation_config=gen_config,
            logits_processor=logits_processor,
            use_cache=True,
            return_dict_in_generate=False,
        )
    decoded_outputs = tokenizer.batch_decode(outputs)
    for i, out in enumerate(decoded_outputs):
        canonical_smiles = extract_between(out, "[SMILES]", "[/SMILES]")
        generated_conformer = extract_between(out, "[CONFORMER]", "[/CONFORMER]")
        geom_smiles = geom_smiles_list[i]

        if generated_conformer:
            generated_smiles = strip_smiles(generated_conformer)
            if not same_molecular_graph(canonical_smiles, generated_smiles):
                logger.debug(f"smiles mismatch: \n{canonical_smiles=}\n{generated_smiles=}\n{generated_conformer=}")
                stats["smiles_mismatch"] += 1
            else:
                try:
                    mol_obj = decode_cartesian_v2(generated_conformer)
                    generations[geom_smiles].append(mol_obj)
                    stats["success"] += 1
                except Exception:
                    logger.debug(f"smiles fails parsing: \n{canonical_smiles=}\n{generated_smiles=}\n{generated_conformer=}")
                    stats["mol_parse_fail"] += 1
        else:
            stats["no_eos"] += 1
            logger.debug(f"no eos: \n{out[:1000]=}")

    batch_time = time.perf_counter() - batch_start
    return generations, stats, batch_time

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

    model, tokenizer = load_model_tokenizer(model_path=inference_config["model_path"],
                                            tokenizer_path=inference_config["tokenizer_path"],
                                            torch_dtype=inference_config["torch_dtype"],
                                            attention_imp=inference_config.get("attention_impl", "flash_attention_2"),
                                            device=inference_config["device"])
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

    # Sort by SMILES length DESCENDING - process longest first to fail fast on OOM
    # rather than failing after hours when hitting large molecules at the end
    mols_list.sort(key=lambda x: len(x[0]), reverse=True)

    limit = inference_config.get("limit")
    mols_list = mols_list[:limit]

    stats = Counter({"smiles_mismatch": 0, "mol_parse_fail": 0, "no_eos": 0, "success": 0})
    batch_size = int(inference_config["batch_size"])
    generations_all = defaultdict(list)

    # Resolve sampling config (passed as name for pickle compatibility)
    sampling_config_name = inference_config.get("sampling_config", "top_p_sampling1")
    gen_config = sampling_configs[sampling_config_name]
    logger.info(f"Sampling config: {sampling_config_name}")

    use_logit_processor = inference_config.get("use_logit_processor", False)
    logger.info(f"Logit processor enabled: {use_logit_processor}")
    if use_logit_processor:
        logger.info(f"Using QwenAllowlistLogitsProcessor v4.3 (allowlist + smart blocking)")

    # Performance tracking
    total_samples = len(mols_list)
    total_batches = (total_samples + batch_size - 1) // batch_size
    run_start = time.perf_counter()
    batch_times = []
    log_interval = max(1, total_batches // 10)  # Log ~10 times during run

    # Create thread pool for parallel template building (if LP enabled)
    template_executor = ThreadPoolExecutor(max_workers=4) if use_logit_processor else None

    try:
        batch_idx = 0
        for start in tqdm(range(0, len(mols_list), batch_size), desc="generating"):
            batch = mols_list[start:start + batch_size]
            for sub_batch in split_batch_on_geom_size(batch, max_geom_len=80):
                outputs, stats_, batch_time = process_batch(
                    model, tokenizer, sub_batch, gen_config,
                    eos_token_id, use_logit_processor, template_executor
                )
                stats.update(stats_)
                batch_times.append(batch_time)
                for k, v in outputs.items():
                    generations_all[k].extend(v)

            batch_idx += 1

            # Periodic cache clearing to prevent memory fragmentation (every log_interval batches)
            if batch_idx % log_interval == 0:
                torch.cuda.empty_cache()

            # Periodic logging (every log_interval batches)
            if batch_idx % log_interval == 0 or batch_idx == total_batches:
                processed = min(start + batch_size, total_samples)
                elapsed = time.perf_counter() - run_start
                total_processed = stats["success"] + stats["smiles_mismatch"] + stats["mol_parse_fail"] + stats["no_eos"]
                pass_rate = 100 * stats["success"] / total_processed if total_processed > 0 else 0
                avg_batch_time = sum(batch_times[-log_interval:]) / len(batch_times[-log_interval:]) if batch_times else 0
                throughput = processed / elapsed if elapsed > 0 else 0

                logger.info(
                    f"[Progress {batch_idx}/{total_batches}] "
                    f"processed={processed}/{total_samples} | "
                    f"pass_rate={pass_rate:.1f}% ({stats['success']}/{total_processed}) | "
                    f"errors: mismatch={stats['smiles_mismatch']}, parse_fail={stats['mol_parse_fail']}, no_eos={stats['no_eos']} | "
                    f"avg_batch_time={avg_batch_time:.2f}s | throughput={throughput:.1f} samples/s"
                )
    finally:
        if template_executor is not None:
            template_executor.shutdown(wait=False)

    # Final summary
    total_time = time.perf_counter() - run_start
    total_processed = stats["success"] + stats["smiles_mismatch"] + stats["mol_parse_fail"] + stats["no_eos"]
    final_pass_rate = 100 * stats["success"] / total_processed if total_processed > 0 else 0
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0

    logger.info("=" * 60)
    logger.info("INFERENCE COMPLETE")
    logger.info(f"  Total samples: {total_processed}")
    logger.info(f"  Pass rate: {final_pass_rate:.2f}% ({stats['success']}/{total_processed})")
    logger.info(f"  Errors: mismatch={stats['smiles_mismatch']}, parse_fail={stats['mol_parse_fail']}, no_eos={stats['no_eos']}")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"  Avg batch time: {avg_batch_time:.2f}s")
    logger.info(f"  Throughput: {total_processed/total_time:.1f} samples/s")
    logger.info(f"  Unique molecules: {len(generations_all)}")
    logger.info("=" * 60)

    save_results(results_path, dict(generations_all), dict(stats))

    return generations_all, stats


def launch_inference_from_cli(device: str, grid_run_inference: bool, test_set:str = None, xl:bool = False, use_logit_processor: bool = False, attention_impl: str = "flash_attention_2", batch_size: int = 400, sampling_config: str = "top_p_sampling1", qm9:bool = False, limit: int = None) -> None:
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
        executor = submitit.AutoExecutor(folder="~/slurm_jobs/conf_gen/job_%j")
    elif device == "local":
        executor = submitit.LocalExecutor(folder="~/slurm_jobs/conf_gen/job_%j")
    executor.update_parameters(
        name="conf_gen",
        timeout_min=24 * 24 * 60,
        gpus_per_node=n_gpus,
        nodes=1,
        mem_gb=80,
        cpus_per_task=n_gpus * 4,
        slurm_additional_parameters={"partition": node},
    )
    
    # Base configuration template (all paths as strings for pickle compatibility)
    base_inference_config = {
        "model_path": str(get_ckpt("qwen3_06b_pre", "40k")),
        "tokenizer_path": str(get_tokenizer_path("qwen3_0.6b_custom")),
        "torch_dtype": "bfloat16",
        "attention_impl": attention_impl,
        "batch_size": batch_size,
        "num_gens": gen_num_codes["2k_per_conf"],
        "sampling_config": sampling_config,  # Pass name, resolve inside run_inference
        "device": "cuda",
        "results_path": str(get_base_path("gen_results_root")),
        "run_name": "qwen_pre",
        "use_logit_processor": use_logit_processor,
        "limit": limit,
    }

    if grid_run_inference:
        param_grid = {
            # "model_path": [("m380_conf_v2", "4e")],
            "model_path": [("m600_qwen", "3e"), ("m600_qwen", "1e"), ("m600_qwen", "2e")],
            # "model_path": [("m380_conf_v2", "1e")],
        }
        jobs = []
        if executor is not None:
            with executor.batch():
                for model_key in param_grid["model_path"]:
                    for test_set_name in test_sets_to_run:
                        grid_config = dict(base_inference_config)
                        if isinstance(model_key, tuple):
                            grid_config["model_path"] = str(get_ckpt(model_key[0], model_key[1]))
                            model_key_str = f"{model_key[0]}_{model_key[1]}"
                        else:
                            grid_config["model_path"] = str(get_ckpt(model_key))
                            model_key_str = model_key

                        if test_set_name == "xl":
                            grid_config["batch_size"] = 100

                        if test_set_name == "qm9":
                            grid_config["batch_size"] = 100

                        grid_config["test_data_path"] = str(get_data_path(f"{test_set_name}_smi"))
                        grid_config["test_set"] = test_set_name
                        grid_config["run_name"] = f"{model_key_str}_{test_set_name}"
                        
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
                    inference_config["test_data_path"] = str(get_data_path(f"{test_set_name}_smi"))
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
                inference_config["test_data_path"] = str(get_data_path(f"{test_set_name}_smi"))
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
    parser.add_argument("--logit-processor", action="store_true",
                        help="Enable constrained logit processor for SMILES structure enforcement")
    parser.add_argument("--attention", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "sdpa_paged", "eager"],
                        help="Attention implementation (default: flash_attention_2)")
    parser.add_argument("--batch-size", type=int, default=400,
                        help="Batch size for generation (default: 400, xl uses 100 if not specified)")
    parser.add_argument("--sampling-config", type=str, default="top_p_sampling1",
                        choices=list(sampling_configs.keys()),
                        help="Sampling configuration (default: top_p_sampling1)")
    parser.add_argument("--qm9", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    launch_inference_from_cli(
        device=args.device,
        grid_run_inference=args.grid_run_inference,
        test_set=args.test_set,
        xl=args.xl, qm9=args.qm9, limit=args.limit,
        use_logit_processor=getattr(args, 'logit_processor', False),
        attention_impl=args.attention,
        batch_size=args.batch_size,
        sampling_config=args.sampling_config,
    )

    
