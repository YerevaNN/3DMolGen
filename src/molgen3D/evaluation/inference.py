import copy
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import time
import random
import gc
import multiprocessing
from collections import defaultdict, Counter

import torch
import cloudpickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from loguru import logger
import submitit

from molgen3D.data_processing.smiles_encoder_decoder import (
    decode_cartesian_v2,
    strip_smiles,
    decode_cartesian_binned_v2 as decode_cartesian_binned,
    get_bins_for_coords
)
# Use this repo's self-contained FSQ tokenizer model -- the exact module that
# encoded the big-data training set -- as the coordinate decoder at inference.
from molgen3D.data_processing.smiles_encoder_decoder_fsq import MolFSQModel
from molgen3D.evaluation.utils import (
    same_molecular_graph,
    log_cuda_memory,
    log_cuda_summary,
    estimate_decoder_flops_per_token,
    detect_peak_flops,
    log_mfu,
)
from molgen3D.config.paths import get_ckpt, get_tokenizer_path, get_data_path, get_base_path
from molgen3D.config.sampling_config import sampling_configs, gen_num_codes


def _resolve_existing_path(*candidates: str | os.PathLike[str] | None) -> Path:
    """Return the first existing path from the provided candidates."""
    fallback: Path | None = None
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if fallback is None:
            fallback = path
        try:
            if path.exists():
                return path
        except PermissionError:
            continue

    if fallback is None:
        raise FileNotFoundError("No path candidates were provided")
    return fallback


torch.set_grad_enabled(False)

def set_seed(seed=42):
    """Set random seed for reproducibility."""
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


def load_model_tokenizer(
    model_path,
    tokenizer_path,
    torch_dtype="bfloat16",
    attention_imp="sdpa",
    device="auto",
):
    """Load model and tokenizer with appropriate configurations."""
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
    load_kwargs = {
        "dtype": dtype_obj,
        "attn_implementation": attention_imp,
        "trust_remote_code": True,
        "local_files_only": True,
    }

    target_device = device
    if device in ("cuda", "cpu"):
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            **load_kwargs,
        ).eval()
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map=device,
            **load_kwargs,
        ).eval()
        target_device = getattr(model, "device", device)

    if hasattr(model, "lm_head") and hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_weight = model.model.embed_tokens.weight
        head_weight = model.lm_head.weight
        logger.info(
            "HF head status: "
            f"shared_ptr={head_weight.data_ptr() == embed_weight.data_ptr()}, "
            f"equal_by_value={torch.equal(head_weight.detach().cpu(), embed_weight.detach().cpu())}, "
            f"tie_word_embeddings={getattr(model.config, 'tie_word_embeddings', None)}"
        )

    model._flops_per_token = estimate_decoder_flops_per_token(model.config)
    model._peak_device_flops = detect_peak_flops(target_device)

    log_cuda_memory("Post-load")

    disable_torch_compile = os.environ.get("DISABLE_TORCH_COMPILE", "0") == "1"
    if disable_torch_compile:
        logger.info("DISABLE_TORCH_COMPILE=1; using eager model for inference.")
        log_cuda_memory("Post-compile")
    else:
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
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    print(f"{model.dtype=}, {target_device=}")

    return model, tokenizer


def _prepare_generation_config(gen_config, tokenizer, eos_token_id: int):
    prepared = copy.deepcopy(gen_config)
    prepared.eos_token_id = eos_token_id
    prepared.pad_token_id = tokenizer.pad_token_id
    prepared.bos_token_id = tokenizer.bos_token_id
    if hasattr(prepared, "max_length"):
        prepared.max_length = None
    return prepared


def _extract_conformer_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    search_start = 0
    while True:
        start = text.find("[CONFORMER]", search_start)
        if start == -1:
            break
        content_start = start + len("[CONFORMER]")
        end = text.find("[/CONFORMER]", content_start)
        if end == -1:
            break
        candidate = text[content_start:end]
        if candidate:
            candidates.append(candidate)
        search_start = end + len("[/CONFORMER]")
    return candidates

def save_results(results_path, generations, stats):
    """Save generation results to pickle and text files."""
    with open(os.path.join(results_path, "generation_results.pickle"), 'wb') as results_file_pickle:
        cloudpickle.dump(generations, results_file_pickle, protocol=4)
    
    with open(os.path.join(results_path, "generation_results.txt"), 'w') as results_file_txt:
        results_file_txt.write(f"{stats=}")

def process_batch(model, tokenizer, batch: list[list], gen_config, eos_token_id, binned: bool, fsq: bool = False, fsq_decoder = None):
    """Process a batch of molecules and generate conformers."""
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
    prepared_gen_config = _prepare_generation_config(gen_config, tokenizer, eos_token_id)

    start_time = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=tokenized_prompts["input_ids"],
            attention_mask=tokenized_prompts["attention_mask"],
            max_new_tokens=2500,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            generation_config=prepared_gen_config,
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
        
        # Match the original working path: use the last conformer span in the output.
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
                    elif fsq:
                        recon_enriched = fsq_decoder.decode_text(generated_conformer)
                        mol_obj = decode_cartesian_v2(recon_enriched)
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
    """Main inference function for model execution."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    device_arg = inference_config.get("device", "cuda")
    target_device = device_arg
    set_seed(42)
    
    results_path = os.path.join(
        *[inference_config["results_path"],
          datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + inference_config["run_name"]]
    )
    os.makedirs(results_path, exist_ok=True)
    logger.add(os.path.join(results_path, "logs.txt"), rotation="500 MB")
    
    # Redirect stdout/stderr to logger to capture external library messages
    class StdoutRedirect:
        def write(self, message):
            if message and message.strip():
                logger.info(message.rstrip())
        def flush(self):
            pass
    
    class StderrRedirect:
        def write(self, message):
            if message and message.strip():
                logger.error(message.rstrip())
        def flush(self):
            pass
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StdoutRedirect()
    sys.stderr = StderrRedirect()
    
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

    if inference_config.get("unique_only", False):
        mols_list = list(dict.fromkeys(mols_list))
        logger.info(f"unique_only enabled: reduced prompts to {len(mols_list)}")

    limit = inference_config.get("limit")
    mols_list = mols_list[:limit]
    
    stats = Counter({"smiles_mismatch": 0, "mol_parse_fail": 0, "no_eos": 0})
    batch_size = int(inference_config["batch_size"])
    generations_all = defaultdict(list)

    binned = inference_config.get("binned", False)
    if not binned and "binned" in str(inference_config["model_path"]):
        logger.info("Auto-detecting binned=True based on model path")
        binned = True
    
    # FSQ detection and setup
    fsq = inference_config.get("fsq", False)
    if not fsq and "fsq" in str(inference_config["model_path"]).lower():
        logger.info("Auto-detecting fsq=True based on model path")
        fsq = True
        
    fsq_decoder = None
    if fsq:
        # Load the VQ-VAE model for decoding coordinates
        try:
            vq_ckpt = _resolve_existing_path(
                os.environ.get("VQ_CKPT_PATH"),
                Path(__file__).resolve().parents[4] / "checkpoints" / "fsq" / "last-v2.ckpt",
                "/auto/home/filya/fsq_remote/checkpoints/fsq/last-v2.ckpt",
                "/mnt/weka/fgeikyan/fsq/new_checkpoints/full_d1024_v4096_b128_lr0_0001_20260327_131937/last-v2.ckpt",
            )
            logger.info(f"Loading VQ-VAE model from {vq_ckpt}...")

            fsq_decoder = MolFSQModel.load_from_checkpoint(
                str(vq_ckpt), map_location="cpu", device=target_device
            )
            fsq_decoder.eval()
            logger.info(
                f"VQ-VAE model loaded via MolFSQModel.load_from_checkpoint "
                f"(d_model={fsq_decoder.d_model}, levels={fsq_decoder.levels})."
            )
        except Exception as e:
            logger.error(f"Failed to load VQ-VAE model: {e}")
            logger.warning("Proceeding without VQ-VAE. Coordinate reconstruction will fail or be approximate.")
            fsq_decoder = None

    # Show progress for successfully decoded & saved molecules (not total attempted)
    saved_count = 0
    pbar = tqdm(total=len(mols_list), desc="generating", unit="mol")
    for start in range(0, len(mols_list), batch_size):
        batch = mols_list[start:start + batch_size]
        saved_in_batch = 0
        for sub_batch in split_batch_on_geom_size(batch, max_geom_len=80):
            outputs, stats_ = process_batch(
                model,
                tokenizer,
                sub_batch,
                gen_config=inference_config["gen_config"],
                eos_token_id=eos_token_id,
                binned=binned,
                fsq=fsq,
                fsq_decoder=fsq_decoder # Now passing the full MolModel if loaded
            )
            stats.update(stats_)
            for k, v in outputs.items():
                generations_all[k].extend(v)
            # Count successfully decoded / saved molecules from this sub-batch
            try:
                saved_in_sub = sum(len(v) for v in outputs.values())
            except Exception:
                saved_in_sub = 0
            saved_in_batch += saved_in_sub

        saved_count += saved_in_batch
        # Update tqdm by the number of actually saved molecules
        if saved_in_batch:
            pbar.update(saved_in_batch)
        pbar.set_postfix({"saved": saved_count, "errors": stats["mol_parse_fail"] + stats["smiles_mismatch"] + stats["no_eos"]})

        # Periodic checkpoint: flush chunk to disk and clear memory
        if (start // batch_size) % 500 == 0 and start > 0:
            chunk_idx = start // batch_size // 500
            chunk_path = os.path.join(results_path, f"generation_results_chunk_{chunk_idx}.pickle")
            with open(chunk_path, "wb") as _f:
                cloudpickle.dump(dict(generations_all), _f, protocol=4)
            logger.info(f"Chunk {chunk_idx} saved to {chunk_path} ({saved_count} mols so far), clearing accumulator")
            generations_all.clear()
            gc.collect()

        # Aggressive memory cleanup between batches to prevent fragmentation
        try:
            del batch, outputs, stats_
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pbar.close()

    # Merge any flushed chunks with the remaining in-memory results
    import glob as _glob
    chunk_files = sorted(_glob.glob(os.path.join(results_path, "generation_results_chunk_*.pickle")))
    if chunk_files:
        merged = defaultdict(list)
        for chunk_path in chunk_files:
            with open(chunk_path, "rb") as _f:
                chunk = cloudpickle.load(_f)
            for k, v in chunk.items():
                merged[k].extend(v)
        for k, v in generations_all.items():
            merged[k].extend(v)
        save_results(results_path, dict(merged), stats)
        for chunk_path in chunk_files:
            os.remove(chunk_path)
    else:
        save_results(results_path, dict(generations_all), stats)

    return generations_all, stats


def launch_inference_from_cli(
    device: str,
    grid_run_inference: bool = False,
    test_set: str = None,
    xl: bool = False,
    qm9: bool = False,
    limit: int = None,
    unique_only: bool = False,
    binned: bool = False,
    icl: bool = False,
    icl_n: int = 5,
    parallel_jobs: int = 1,
    fsq: bool = False,
    batch_size: int = 2048,
    epoch: str = "4e",
    model_path: str | None = None,
    tokenizer_path: str | None = None,
    test_data_path: str | None = None,
    results_path: str | None = None,
    run_name: str | None = None,
    sampling_config: str = "top_p_sampling1",
) -> None:
    """Launch inference jobs via CLI arguments."""
    if sampling_config not in sampling_configs:
        available = ", ".join(sorted(sampling_configs))
        raise ValueError(f"Unknown sampling_config={sampling_config!r}. Available: {available}")

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
        "tokenizer_path": get_tokenizer_path("qwen3_0.6b_custom"),
        "torch_dtype": "bfloat16",
        "batch_size": batch_size,
        "num_gens": gen_num_codes["2k_per_conf"],
        "gen_config": sampling_configs[sampling_config],
        "sampling_config": sampling_config,
        "device": "cuda",
        "results_path": get_base_path("gen_results_root"),
        "run_name": "qwen_pre_4e_grouped",
        "limit": limit,
        "unique_only": unique_only,
        "binned": binned,
        "fsq": fsq,
    }
    
    # Auto-switch model if FSQ is requested and we are using the default base config
    if fsq:
        if model_path is None:
            base_inference_config["model_path"] = get_ckpt("m1700_qwen_pre_fsq", epoch)
            logger.info(f"FSQ mode enabled: Switched model to {base_inference_config['model_path']}")
        else:
            logger.info("FSQ mode enabled with explicit model_path; skipping alias lookup.")
        if run_name is None:
            base_inference_config["run_name"] = f"qwen17b_pre_fsq_{epoch}"
        if tokenizer_path is None:
            base_inference_config["tokenizer_path"] = get_tokenizer_path("qwen3_1.7b_fsq_4096")

    if model_path:
        base_inference_config["model_path"] = Path(model_path).expanduser().resolve()
    if tokenizer_path:
        base_inference_config["tokenizer_path"] = Path(tokenizer_path).expanduser().resolve()
    if test_data_path:
        base_inference_config["test_data_path"] = Path(test_data_path).expanduser().resolve()
    if results_path:
        base_inference_config["results_path"] = Path(results_path).expanduser().resolve()
    if run_name:
        base_inference_config["run_name"] = run_name

    all_configs = []
    if grid_run_inference:
        param_grid = [
            ("m600_qwen_pre_4e_grouped", "1e"),
            ("m600_qwen_pre_4e_grouped", "2e"),
            ("m600_qwen_pre_4e_grouped", "3e"),
            ("m600_qwen_pre_4e_grouped", "4e"),
            ("m600_qwen_pre", "4e"),
        ]
        if fsq:
            param_grid = [
                ("m1700_qwen_pre_fsq", "2e"),
                ("m1700_qwen_pre_fsq", "3e"),
                ("m1700_qwen_pre_fsq", "4e"),
                ("m1700_qwen_pre_fsq", "5e"),
                ("m1700_qwen_pre_fsq", "6e"),
            ]

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

                if "test_data_path" not in grid_config:
                    grid_config["test_data_path"] = get_data_path(f"{test_set_name}_smi")
                grid_config["test_set"] = test_set_name
                grid_config["run_name"] = f"{model_key_str}_{test_set_name}"
                all_configs.append((grid_config, grid_config["run_name"]))
    else:
        for test_set_name in test_sets_to_run:
            inference_config = dict(base_inference_config)

            if test_set_name == "xl":
                inference_config["batch_size"] = 100
            if test_set_name == "qm9":
                inference_config["batch_size"] = 100
            if test_set_name == "icl":
                inference_config["batch_size"] = 64

            if "test_data_path" not in inference_config:
                inference_config["test_data_path"] = get_data_path(f"{test_set_name}_smi")
            inference_config["test_set"] = test_set_name
            if run_name:
                inference_config["run_name"] = run_name
            else:
                inference_config["run_name"] = f"fsq_{epoch}_{test_set_name}" if fsq else f"new_data_p1_{test_set_name}"
            all_configs.append((inference_config, inference_config["run_name"]))

    if executor is not None:
        with executor.batch():
            for inference_config, run_name in all_configs:
                logger.info(f"Submitting inference for {run_name} with config: {inference_config}")
                executor.submit(run_inference, inference_config=inference_config)
    else:
        if parallel_jobs <= 1 or len(all_configs) == 1:
            logger.info(f"Running {len(all_configs)} inference jobs locally (sequential)")
            for inference_config, run_name in all_configs:
                logger.info(f"Running inference for {run_name}")
                run_inference(inference_config=inference_config)
        else:
            max_workers = min(parallel_jobs, len(all_configs))
            logger.info(f"Running {len(all_configs)} inference jobs locally in parallel (max workers: {max_workers})")

            visible_env = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible_env:
                visible_gpus = [gpu.strip() for gpu in visible_env.split(",") if gpu.strip()]
            else:
                visible_gpus = [str(i) for i in range(torch.cuda.device_count())]
            if not visible_gpus:
                visible_gpus = ["0"]

            ctx = multiprocessing.get_context('spawn')
            processes = []
            parent_cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
            for i, (config, name) in enumerate(all_configs[:max_workers]):
                assigned_gpu = visible_gpus[i % len(visible_gpus)]
                logger.info(f"Assigning {name} to GPU {assigned_gpu}")
                os.environ["CUDA_VISIBLE_DEVICES"] = assigned_gpu
                if i > 0:
                    time.sleep(5)
                p = ctx.Process(target=run_inference, kwargs={'inference_config': config})
                p.start()
                processes.append((p, name, assigned_gpu))

            if parent_cuda_env is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = parent_cuda_env

            for p, run_name, assigned_gpu in processes:
                p.join()
                if p.exitcode == 0:
                    logger.info(f"✓ Completed: {run_name} on GPU {assigned_gpu}")
                else:
                    logger.error(f"✗ Process {run_name} on GPU {assigned_gpu} exited with code {p.exitcode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["local", "a100", "h100"], required=True)
    parser.add_argument("--grid_run_inference", action="store_true")
    parser.add_argument("--test_set", type=str, choices=["clean", "distinct", "corrected"], default=None)
    parser.add_argument("--binned", action="store_true", default=False)
    parser.add_argument("--fsq", action="store_true", default=False)
    parser.add_argument("--xl", action="store_true")
    parser.add_argument("--qm9", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--unique_only", action="store_true")
    parser.add_argument("--icl", action="store_true")
    parser.add_argument("--icl_n", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--parallel_jobs", type=int, default=1, help="Number of parallel inference jobs for local execution")
    parser.add_argument("--epoch", type=str, default="4e", help="FSQ model epoch/step key (e.g. 1e, 2e, 3e, 4e)")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--sampling_config", type=str, default="top_p_sampling1", choices=sorted(sampling_configs))
    args = parser.parse_args()
    launch_inference_from_cli(
        device=args.device,
        grid_run_inference=args.grid_run_inference,
        test_set=args.test_set,
        xl=args.xl,
        qm9=args.qm9,
        limit=args.limit,
        unique_only=args.unique_only,
        binned=args.binned,
        icl=args.icl,
        icl_n=args.icl_n,
        parallel_jobs=args.parallel_jobs,
        fsq=args.fsq,
        batch_size=args.batch_size,
        epoch=args.epoch,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        test_data_path=args.test_data_path,
        results_path=args.results_path,
        run_name=args.run_name,
        sampling_config=args.sampling_config,
    )
