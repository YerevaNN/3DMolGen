from transformers import AutoTokenizer, AutoModelForCausalLM
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
torch.set_grad_enabled(False)

# from utils import parse_molecule_with_coordinates
from molgen3D.data_processing.utils import decode_cartesian_raw
from molgen3D.data_processing.smiles_encoder_decoder import decode_cartesian_v2, strip_smiles
from molgen3D.evaluation.utils import extract_between, same_molecular_graph
from molgen3D.config.paths import get_ckpt, get_tokenizer_path, get_data_path, get_base_path
from molgen3D.config.sampling_config import sampling_configs, gen_num_codes
from molgen3D.training.grpo.rewards import tag_pattern

torch.backends.cudnn.benchmark = True

def set_seed(seed=42):
    random.seed(seed)  # Python random module
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # All GPUs (if using multi-GPU)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model_tokenizer(model_path, tokenizer_path, torch_dtype="bfloat16", attention_imp="flash_attention_2", device="auto"):
    tokenizer  = AutoTokenizer.from_pretrained(str(tokenizer_path), padding_side='left', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path),
                                                 torch_dtype=getattr(torch, torch_dtype),
                                                 attn_implementation=attention_imp,
                                                 device_map=device,
                                                 trust_remote_code=True,
                                                 local_files_only=True).eval()
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
    
    tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True)
    tokenized_prompts = {k: v.to(model.device, non_blocking=True) for k, v in tokenized_prompts.items()}
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=tokenized_prompts["input_ids"], 
            attention_mask=tokenized_prompts["attention_mask"],
            max_new_tokens=4000,
            eos_token_id=eos_token_id, 
            generation_config=gen_config,
            use_cache=True,
            return_dict_in_generate=False,
        )
    decoded_outputs = tokenizer.batch_decode(outputs)
    for i, out in enumerate(decoded_outputs):
        canonical_smiles = extract_between(out, "[SMILES]", "[/SMILES]")
        generated_conformer = extract_between(out, "[CONFORMER]", "[/CONFORMER]")
        geom_smiles = geom_smiles_list[i]
        
        if generated_conformer:
            generated_smiles = tag_pattern.sub('', generated_conformer)     
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
                                            device=inference_config["device"])
    logger.info(f"model loaded: {model.dtype=}, {model.device=}")
    
    eos_token_id = tokenizer.encode("[/CONFORMER]")[0]
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
    
    if inference_config.get("limit"):
        mols_list = mols_list[:inference_config["limit"]]

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

    save_results(results_path, dict(generations_all), stats)

    return generations_all, stats


def launch_inference_from_cli(device: str, grid_run_inference: bool, test_set:str = None, xl:bool = False, qm9:bool = False, limit: int = None) -> None:
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
    
    # Base configuration template
    base_inference_config = {
        "model_path": get_ckpt("m380_conf_v2","2e"),
        "tokenizer_path": get_tokenizer_path("llama3_chem_v1"),
        "torch_dtype": "bfloat16",
        "batch_size": 400,
        "num_gens": gen_num_codes["2k_per_conf"],
        "gen_config": sampling_configs["top_p_sampling1"],
        "device": "cuda",
        "results_path": get_base_path("gen_results_root"),
        "run_name": "new_data_p1",
    }
    if limit:
        base_inference_config["limit"] = limit

    if grid_run_inference:
        param_grid = {
            "model_path": [("m380_conf_v2", "4e")],
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
    parser.add_argument("--device", type=str, choices=["local", "a100", "h100"], required=True)
    parser.add_argument("--grid_run_inference", action="store_true")
    parser.add_argument("--test_set", type=str, choices=["clean", "distinct", "corrected"], default=None)
    parser.add_argument("--xl", action="store_true")
    parser.add_argument("--qm9", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args() 
    launch_inference_from_cli(device=args.device, grid_run_inference=args.grid_run_inference, test_set=args.test_set, xl=args.xl, qm9=args.qm9, limit=args.limit)

    
