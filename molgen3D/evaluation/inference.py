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
                                                #  attn_implementation=attention_imp,
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


def process_batch(model, tokenizer, batch, gen_config, tag_pattern):
    generations = defaultdict(list)
    stats = {"smiles_mismatch":0, "mol_parse_fail" :0, "no_eos":0}
    tokenized_prompts = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=tokenized_prompts["input_ids"], 
            attention_mask=tokenized_prompts["attention_mask"],
            max_new_tokens=2000, 
            eos_token_id=128329, 
            generation_config=gen_config,
        )
    decoded_outputs = tokenizer.batch_decode(outputs)
    for out in decoded_outputs:
        canonical_smiles = extract_between(out, "[SMILES]", "[/SMILES]")
        generated_conformer = extract_between(out, "[CONFORMER]", "[/CONFORMER]")
        # logger.info(f"\n{canonical_smiles=}\nlen out {len(out)=} {out=}")
        if generated_conformer:
            generated_smiles = tag_pattern.sub('', generated_conformer)         
            if generated_smiles != canonical_smiles:
                logger.info(f"smiles mismatch: \n{canonical_smiles=}\n{generated_smiles=}\n{out=}")
                stats["smiles_mismatch"] += 1
            else:
                try:
                    mol_obj = decode_cartesian_raw(generated_conformer)
                    generations[canonical_smiles].append(mol_obj)
                except:
                    logger.info(f"smiles fails parsing: \n{canonical_smiles=}\n{generated_smiles=}\n{out=}")
                    stats["mol_parse_fail"] += 1
        else:
            stats["no_eos"] += 1
            logger.info(f"no eos: \n{out=}")
    del tokenized_prompts, outputs, decoded_outputs
    torch.cuda.empty_cache()
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
    
    with open(inference_config["test_data_path"],'rb') as test_data_file:
        test_data = cloudpickle.load(test_data_file)

    num_gens_co = inference_config["num_gens"]
    tag_pattern = re.compile(r'<[^>]*>')
    mols_list, batch, generations = [], [], defaultdict(list)
    stats = Counter({"smiles_mismatch":0, "mol_parse_fail" :0, "no_eos":0})
    for en, sample in enumerate(tqdm(test_data.values())):
        num_gens = (sample["num_confs"] * int(num_gens_co[0])) if type(num_gens_co) == str else num_gens_co
        mols_list.extend([f"[SMILES]{sample['canonical_smiles']}[/SMILES]"] * num_gens)

    # Inference loop
    for i in tqdm(range(0, len(mols_list), inference_config["batch_size"])):
        batch = mols_list[i:i + inference_config["batch_size"]]
        if max([len(b) for b in batch]) > 100:
            mid = len(batch) // 2
            sub_batches = [batch[:mid], batch[mid:]]
        else:
            sub_batches = [batch]
        for sub_batch in sub_batches:
            outputs, stats_ = process_batch(model, tokenizer, sub_batch, inference_config["gen_config"], tag_pattern)
            for k, v in outputs.items():
                generations[k].extend(v)
            stats.update(stats_)
            torch.cuda.empty_cache() 
        # outputs, stats_ = process_batch(model, tokenizer, batch, inference_config["gen_config"], tag_pattern)
        # for k, v in outputs.items():
        #     generations[k].extend(v)
        # stats.update(stats_)
        # torch.cuda.empty_cache()  # Clear GPU memory after each batch

    save_results(results_path, generations, stats)

    return generations, stats


def launch_inference_from_cli(device: str, grid_search: bool) -> None:
    with open("molgen3D/config/paths.yaml", "r") as f:
        paths = yaml.safe_load(f)
    n_gpus = 1
    experiment_name = "conf_gen"
    node = device if device in {"a100", "h100"} else None
    executor = None
    if device in {"a100", "h100"}:
        executor = submitit.AutoExecutor(folder="~/slurm_jobs/conf_gen/job_%j")
        executor.update_parameters(
            name=experiment_name,
            timeout_min=24 * 24 * 60,
            gpus_per_node=n_gpus,
            nodes=1,
            mem_gb=80,
            cpus_per_task=n_gpus * 15,
            slurm_additional_parameters={"partition": node},
        )
    inference_config = {
        "model_path": paths["model_paths"]["m380_1e"],
        "tokenizer_path": paths["tokenizer_path"],
        "test_data_path": paths["test_data_path"],
        "torch_dtype": "bfloat16",
        "batch_size": 256,
        "num_gens": gen_num_codes["2k_per_conf"],
        "gen_config": sampling_configs["top_p_sampling1"],
        "device": "cuda" if device != "local" else "cpu",
        "results_path": paths["results_path"],
        "run_name": "m380_1e_newdataprocess",
    }
    if grid_search:
        param_grid = {
            "model_path": ["m380_1e_1xgrpo", "m380_1e_1xgrpo_6e-5", "m380_1e_1xgrpo_50e_100s", "m380_1e_1xgrpo_100e_100s"],
            "gen_config": ["top_p_sampling1", "top_p_sampling2"],
        }
        jobs = []
        if executor is not None:
            with executor.batch():
                for model_key, gen_config_key in itertools.product(param_grid["model_path"], param_grid["gen_config"]):
                    grid_config = dict(inference_config)
                    grid_config["model_path"] = paths["model_paths"][model_key]
                    grid_config["gen_config"] = sampling_configs[gen_config_key]
                    grid_config["run_name"] = f"{model_key}_{gen_config_key}"
                    job = executor.submit(run_inference, inference_config=grid_config)
                    jobs.append(job)
        else:
            for model_key, gen_config_key in itertools.product(param_grid["model_path"], param_grid["gen_config"]):
                grid_config = dict(inference_config)
                grid_config["model_path"] = paths["model_paths"][model_key]
                grid_config["gen_config"] = sampling_configs[gen_config_key]
                grid_config["run_name"] = f"{model_key}_{gen_config_key}"
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
    args = parser.parse_args()
    launch_inference_from_cli(device=args.device, grid_search=args.grid_search)

    
