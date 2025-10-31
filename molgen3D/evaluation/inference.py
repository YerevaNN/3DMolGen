from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import cloudpickle
import random
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
from molgen3D.config.sampling_config import sampling_configs, gen_num_codes
from molgen3D.data_processing.utils import decode_cartesian_raw
from molgen3D.evaluation.utils import extract_between

torch.backends.cudnn.benchmark = True

def set_seed(seed=42):
    random.seed(seed)  # Python random module
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # All GPUs (if using multi-GPU)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model_tokenizer(model_path, tokenizer_path, torch_dtype, attention_imp="flash_attention_2", device="auto"):
    tokenizer  = AutoTokenizer.from_pretrained(tokenizer_path, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                 torch_dtype=getattr(torch, torch_dtype),
                                                 attn_implementation=attention_imp,
                                                 device_map=device).eval()
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    print(f"{model.dtype=}, {model.device=}")

    return model, tokenizer

def save_results(results_path, generations, stats):
    with open(os.path.join(results_path, "generation_results.pickle"), 'wb') as results_file_pickle:
        cloudpickle.dump(generations, results_file_pickle, protocol=4)
    
    with open(os.path.join(results_path, "generation_results.txt"), 'w') as results_file_txt:
        results_file_txt.write(f"{stats=}")

def process_batch(model, tokenizer, batch: list[list], gen_config, tag_pattern, eos_token_id):
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
            max_new_tokens=2000,
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
            if generated_smiles != canonical_smiles:
                logger.info(f"smiles mismatch: \n{canonical_smiles=}\n{generated_smiles=}\n{generated_conformer=}")
                stats["smiles_mismatch"] += 1
            else:
                try:
                    mol_obj = decode_cartesian_raw(generated_conformer)
                    generations[geom_smiles].append(mol_obj)
                except:
                    logger.info(f"smiles fails parsing: \n{canonical_smiles=}\n{generated_smiles=}\n{generated_conformer=}")
                    stats["mol_parse_fail"] += 1
        else:
            stats["no_eos"] += 1
            logger.info(f"no eos: \n{out[:1000]=}")
    return generations, stats

def run_inference(inference_config: dict):
    results_path = os.path.join(*[inference_config["results_path"], 
                                  datetime.now().strftime('%Y-%m-%d-%H:%M') + 
                                  '_' + inference_config["run_name"]])
    os.makedirs(results_path, exist_ok=True)
    logger.add(os.path.join(results_path, "logs.txt"), rotation="50 MB")
    logger.info(inference_config)

    model, tokenizer = load_model_tokenizer(model_path=inference_config["model_path"],
                                            tokenizer_path=inference_config["tokenizer_path"],
                                            torch_dtype=inference_config["torch_dtype"],
                                            device=inference_config["device"])
    
    tag_pattern = re.compile(r'<[^>]*>')
    eos_token_id = tokenizer.encode("[/CONFORMER]")[0]
    try:
        with open(inference_config["test_data_path"],'rb') as test_data_file:
            test_data = cloudpickle.load(test_data_file)
    except Exception as e:
        logger.error(f"Failed to load test data from {inference_config['test_data_path']}: {e}")
        logger.warning("Test data files appear to be Git LFS pointers. Please run 'git lfs pull' to download actual data.")
        raise RuntimeError(f"Cannot proceed without test data. Please ensure Git LFS is enabled and run 'git lfs pull' to download the data files.")

    mols_list = []
    test_set: str = inference_config.get("test_set", "corrected")
    if test_set in ("corrected", "clean"):
        for geom_smiles, data in test_data.items():
            mols_list.extend([(geom_smiles, f"[SMILES]{data['corrected_smi']}[/SMILES]")] * data["num_confs"] * 2)
    elif test_set == "distinct":
        for geom_smiles, data in test_data.items():
            for sub_smiles, count in data["sub_smiles_counts"].items():
                mols_list.extend([(geom_smiles, f"[SMILES]{sub_smiles}[/SMILES]")] * count * 2)
    print(mols_list[:1000])

    mols_list.sort(key=lambda x: len(x[0]))
    
    stats = Counter({"smiles_mismatch":0, "mol_parse_fail" :0, "no_eos":0})
    batch_size = int(inference_config["batch_size"])
    generations_all = defaultdict(list)

    for start in tqdm(range(0, len(mols_list), batch_size), desc="generating"):
        batch = mols_list[start:start + batch_size]
        if any(len(x) > 80 for x in batch):
            mid = len(batch) // 2
            for sub_batch in (batch[:mid], batch[mid:]):
                if not sub_batch:
                    continue
                outputs, stats_ = process_batch(model, tokenizer, sub_batch, inference_config["gen_config"], tag_pattern, eos_token_id)
                stats.update(stats_)
                for k, v in outputs.items():
                    generations_all[k].extend(v)
        else:
            outputs, stats_ = process_batch(model, tokenizer, batch, inference_config["gen_config"], tag_pattern, eos_token_id)
            stats.update(stats_)
            for k, v in outputs.items():
                generations_all[k].extend(v)

    save_results(results_path, dict(generations_all), stats)

    return generations_all, stats


def launch_inference_from_cli(device: str, grid_search: bool, test_set:str = None) -> None:
    with open("molgen3D/config/paths.yaml", "r") as f:
        paths = yaml.safe_load(f)
    n_gpus = 1
    experiment_name = "conf_gen"
    node = device if device in ["a100", "h100"] else "local"
    executor = None
    if device in ["a100", "h100"]:
        executor = submitit.AutoExecutor(folder="~/slurm_jobs/conf_gen/job_%j")
    elif device == "local":
        executor = submitit.LocalExecutor(folder="~/slurm_jobs/conf_gen/job_%j")
    executor.update_parameters(
        name=experiment_name,
        timeout_min=24 * 24 * 60,
        gpus_per_node=n_gpus,
        nodes=1,
        mem_gb=80,
        cpus_per_task=n_gpus * 4,
        slurm_additional_parameters={"partition": node},
    )
    test_set_key_map = {"clean": "clean_smi", "distinct": "distinct_smi", "corrected": "corrected_smi"}
    selected_test_data_path = paths["test_data_path"][test_set_key_map[test_set]]
    inference_config = {
        "model_path": paths["model_paths"]["m380_1e"],
        "tokenizer_path": paths["qw600_tokenizer_path"],
        "test_data_path": selected_test_data_path,
        "torch_dtype": "bfloat16",
        "batch_size": 400,
        "num_gens": gen_num_codes["2k_per_conf"],
        "gen_config": sampling_configs["top_p_sampling1"],
        "device": "cuda",
        "results_path": paths["results_path"],
        "run_name": "qw600_1e_corrected_smi_noFA2_noCache",
        "test_set": test_set,
    }
    if grid_search:
        param_grid = {
            # "model_path": ["nm380_1e", "m380_1e"],
            # "model_path": ["nm380_1e", "nm380_2e", "nm380_3e", "nm380_4e", "m380_1e", "m380_2e", "m380_3e", "m380_4e",  "m380_1e_1xgrpo_1e_lr5e-5_algnTrue", "m380_1e_1xgrpo_100e_100s"],
            # "model_path": ["nm380_500grpo_alignFalse", "nm380_1000grpo_alignFalse", "nm380_1500grpo_alignFalse", "nm380_2000grpo_alignFalse", "nm380_2230grpo_alignFalse", "m380_500grpo_alignFalse", "m380_1000grpo_alignFalse", "m380_1500grpo_alignFalse", "m380_2000grpo_alignFalse", "m380_2230grpo_alignFalse"],
            # "model_path": ["m380_4e_1xgrpo_aF_b05_const05"],
            # "model_path": ["qw600_1e", "qw600_2e", "qw600_3e", "qw600_4e"],
            "model_path": ["qw600_pre_1e", "qw600_pre_2e", "qw600_pre_3e", "qw600_pre_4e", "qw600_src_1e"],
            # "model_path": ["nm380_1e", "nm380_4e", "m380_1e", "m380_1e_1xgrpo_1e_lr5e-5_algnTrue"],
            # "model_path": ["nm380_1e", "nm380_4e", "m380_1e", "m380_4e"],
            "test_set": ["clean"],
            # "model_path": ["nm380_1e", "nm380_2e", "nm380_3e", "nm380_4e", "nm100_1e", 
            # "nm100_2e", "nm100_3e", "nm100_4e", "nm170_1e", "nm170_2e", "nm170_3e", 
            # "nm170_4e", "m380_1e", "m380_1e_1xgrpo_1e_lr5e-5_algnTrue", "m380_1e_1xgrpo_100e_100s"],
            # "model_path": ["nb1_1e", "nb1_2e", "nb1_3e", "nb1_4e"],
            "gen_config": ["top_p_sampling1"],
        }
        jobs = []
        if executor is not None:
            with executor.batch():
                for model_key, gen_config_key, test_set_value in itertools.product(param_grid["model_path"], param_grid["gen_config"], param_grid["test_set"]):
                    grid_config = dict(inference_config)
                    grid_config["model_path"] = paths["model_paths"][model_key]
                    grid_config["gen_config"] = sampling_configs[gen_config_key]
                    grid_config["batch_size"] = 256 if "1b" in model_key else 256  
                    grid_config["test_set"] = test_set_value
                    grid_config["test_data_path"] = paths["test_data_path"][test_set_key_map[test_set_value]]
                    grid_config["run_name"] = f"{model_key}_{gen_config_key}_{test_set_value}"
                    job = executor.submit(run_inference, inference_config=grid_config)
                    jobs.append(job)
        else:
            for model_key, gen_config_key, test_set_value in itertools.product(param_grid["model_path"], param_grid["gen_config"], param_grid["test_set"]):
                grid_config = dict(inference_config)
                grid_config["model_path"] = paths["model_paths"][model_key]
                grid_config["gen_config"] = sampling_configs[gen_config_key]
                grid_config["batch_size"] = 256 if "1b" in model_key else 256
                grid_config["test_set"] = test_set_value
                grid_config["test_data_path"] = paths["test_data_path"][test_set_key_map[test_set_value]]
                grid_config["run_name"] = f"{model_key}_{gen_config_key}_{test_set_value}"
                run_inference(inference_config=grid_config)
    else:
        if executor is not None:
            executor.submit(run_inference, inference_config=inference_config)
        else:
            run_inference(inference_config=inference_config)

if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["local", "a100", "h100"], required=True)
    parser.add_argument("--grid_search", action="store_true")
    parser.add_argument("--test_set", type=str, choices=["clean", "distinct", "corrected"], default="clean")
    args = parser.parse_args()
    launch_inference_from_cli(device=args.device, grid_search=args.grid_search, test_set=args.test_set)

    
