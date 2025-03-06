from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import cloudpickle
import random
import yaml
import re
from tqdm import tqdm
from loguru import logger
from collections import defaultdict, Counter
import submitit
import os
from datetime import datetime

# from utils import parse_molecule_with_coordinates
from molgen3D.config.sampling_config import sampling_configs, gen_num_codes
from molgen3D.utils.data_processing_utils import parse_molecule_with_coordinates
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
    tokenizer  = AutoTokenizer.from_pretrained(tokenizer_path, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation=attention_imp, 
                                                 torch_dtype=getattr(torch, torch_dtype), device_map=device)
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    print(f"{model.dtype=}, {model.device=}")

    return model, tokenizer

def save_results(inference_config, generations, stats):
    results_path = inference_config["results_path"]
    results_path = os.path.join(results_path, inference_config["run_name"])
    os.makedirs(results_path, exist_ok=True)

    with open(os.path.join(results_path, "generation_resutls.pickle"), 'wb') as results_file_pickle:
        cloudpickle.dump(generations, results_file_pickle, protocol=4)
    
    with open(os.path.join(results_path, "generation_resutls.txt"), 'w') as results_file_txt:
        results_file_txt.write(f"{stats=}\n{inference_config['gen_config']}")


def process_batch(model, tokenizer, batch, gen_config):
    prompts, geom_smiles_list, canonical_smiles_list, generations = [], [], [], defaultdict(list)
    stats = {"smiles_mismatch":0, "mol_parse_fail" :0, "no_eos":0}

    for sample in batch:
        geom_smiles = sample["geom_smiles"]
        canonical_smiles = sample["canonical_smiles"]
        num_gens =  sample["num_gens"]
        num_prompts = 1 if inference_config["gen_config"].num_beams > 1 else num_gens
        prompts.extend([f"[SMILES]{sample['canonical_smiles']}[/SMILES]"] * num_prompts)
        geom_smiles_list.extend([geom_smiles] * num_gens)
        canonical_smiles_list.extend([canonical_smiles] * num_gens)

    tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(
        tokenized_prompts.input_ids,
        gen_config,
    )
    decoded_outputs = tokenizer.batch_decode(outputs)
    for geom_smiles, canonical_smiles, out in zip(geom_smiles_list, canonical_smiles_list, decoded_outputs):
        if "[/CONFORMER]" in out:
            generated_conformer = out[out.find("[CONFORMER]")+len("[CONFORMER]"):out.find("[/CONFORMER]")]
            generated_smiles = re.sub(r'<[^>]*>', '', generated_conformer) 
            if generated_smiles != canonical_smiles:
                logger.info(f"smiles mismatch: \n{canonical_smiles=}\n{generated_smiles=}")
                stats["smiles_mismatch"] += 1
            else:
                try:
                    mol_obj = parse_molecule_with_coordinates(generated_conformer)
                    generations[geom_smiles].append(mol_obj)
                    # logger.info(f"correct generation: \n{canonical_smiles=}\n{generated_smiles=}")
                except:
                    logger.info(f"smiles fails parsing: \n{canonical_smiles=}\n{generated_smiles=}")
                    stats["mol_parse_fail"] += 1
        else:
            stats["no_eos"] += 1
            logger.info(f"no eos: \n{geom_smiles=}\n{out=}")

    return generations, stats


def run_inference(inference_config: dict):
    model, tokenizer = load_model_tokenizer(model_path=inference_config["model_path"],
                                            tokenizer_path=inference_config["tokenizer_path"],
                                            device=inference_config["device"])
    
    with open(inference_config["test_data_path"],'rb') as test_data_file:
        test_data = cloudpickle.load(test_data_file)

    num_gens_co = inference_config["num_gens"]
    # num_gens_co = int(num_gens_co[0]) if type(num_gens_co) == str else num_gens_co

    batch, generations = [], defaultdict(list)
    batch_size = 0
    accumulate = True
    stats = Counter({"smiles_mismatch":0, "mol_parse_fail" :0, "no_eos":0})
    for sample in tqdm(test_data.values()):
        num_confs = sample["num_confs"] 
        num_gens = (num_confs * int(num_gens_co[0])) if type(num_gens_co) == str else num_gens_co
        sample["num_gens"] = num_gens
        if batch_size + num_gens < inference_config["batch_size"]:
            batch.append(sample)
            batch_size += num_gens
            if inference_config["gen_config"].num_beams > 1:
                inference_config["batch_size"] = num_gens
                inference_config["gen_config"].num_beams = num_gens
                inference_config["gen_config"].num_beam_groups = num_gens
                inference_config["gen_config"].num_return_sequences = num_gens
        else:
            outputs, stats_ = process_batch(model, 
                                            tokenizer, 
                                            batch, 
                                            inference_config["gen_config"])
            generations.update(outputs)
            stats.update(stats_)
            batch = [sample]
            batch_size = num_gens
            if inference_config["gen_config"].num_beams > 1:
                inference_config["batch_size"] = num_gens
                inference_config["gen_config"].num_beams = num_gens
                inference_config["gen_config"].num_beam_groups = num_gens
                inference_config["gen_config"].num_return_sequences = num_gens

    outputs, stats_ = process_batch(model, 
                                    tokenizer, 
                                    batch, 
                                    inference_config["gen_config"])
    generations.update(outputs) 
    stats.update(stats_)

    save_results(inference_config, generations, stats)

    return generations, stats

        

if __name__ == "__main__":
    set_seed(42)    

    with open("molgen3D/config/paths.yaml", "r") as f:
        paths = yaml.safe_load(f)
    results_path = os.path.join(paths["results_path"], f"{datetime.now().strftime('%Y-%m-%d-%H:%M')}")
    
    executor = submitit.AutoExecutor(folder="~/slurm_jobs/conf_gen/job_%j")
    node = "h100"
    # executor = submitit.local.local.LocalExecutor(folder="~/slurm_jobs/conf_gen/job_%j")
    # node = "local"
    n_gpus = 1
    experiment_name = "conf_gen"
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
        "model_path": paths["model_paths"]["cart_4e_path"],
        "tokenizer_path": paths["tokenizer_path"],
        "test_data_path": paths["test_data_path"],
        "batch_size": 100,
        "num_gens": gen_num_codes["1k_per_conf"],
        "gen_config": sampling_configs["min_p_sampling"], 
        "device": "cuda:0",
        "results_path": results_path,
        "run_name": "16e_1k_beam",
    }

    # run locally
    generations, stats = run_inference(inference_config=inference_config)
    
    ## run on slurm
    # job = executor.submit(run_inference, 
    #                       inference_config=inference_config)

    ## run grid search
    # param_grid = {
    #     "model_path": list(paths["model_paths"].keys()),
    #     "num_gens": list(gen_num_codes.keys()),
    #     "gen_config": list(sampling_configs.keys()),
    # }

    # jobs = []
    # with executor.batch():
    #     for model_key, num_gens_key, gen_config_key in itertools.product(*param_grid.values()):
    #         # Update inference config dynamically
    #         inference_config = {
    #             "model_path": paths["model_paths"][model_key],
    #             "tokenizer_path": paths["tokenizer_path"],
    #             "test_data_path": paths["test_data_path"],
    #             "batch_size": 200,
    #             "num_gens": gen_num_codes[num_gens_key],
    #             "gen_config": sampling_configs[gen_config_key], 
    #             "device": "auto",
    #             "results_path": results_path,
    #             "run_name": f"{model_key}_{num_gens_key}_{gen_config_key}",  # Track run details
    #         }

    #         # Submit job
    #         job = executor.submit(run_inference, inference_config)
    #         jobs.append(job)

    
