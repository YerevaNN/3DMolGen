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
    with open(os.path.join(results_path, "generation_resutls.pickle"), 'wb') as results_file_pickle:
        cloudpickle.dump(generations, results_file_pickle, protocol=4)
    
    with open(os.path.join(results_path, "generation_resutls.txt"), 'w') as results_file_txt:
        results_file_txt.write(f"{stats=}")


def process_batch(model, tokenizer, batch, gen_config, tag_pattern):
    generations = defaultdict(list)
    stats = {"smiles_mismatch":0, "mol_parse_fail" :0, "no_eos":0}
    tokenized_prompts = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
    logger.info(f"\n*******\n{len(batch)=} {max(len(b) for b in batch)=} {tokenized_prompts['input_ids'].shape=}")
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
        logger.info(f"\n{canonical_smiles=}\nlen out {len(out)=} {out=}")
        if generated_conformer:
            generated_smiles = tag_pattern.sub('', generated_conformer)         
            if generated_smiles != canonical_smiles:
                logger.info(f"smiles mismatch: \n{canonical_smiles=}\n{generated_smiles=}")
                stats["smiles_mismatch"] += 1
            else:
                try:
                    mol_obj = decode_cartesian_raw(generated_conformer)
                    generations[canonical_smiles].append(mol_obj)
                except:
                    logger.info(f"smiles fails parsing: \n{canonical_smiles=}\n{generated_smiles=}")
                    stats["mol_parse_fail"] += 1
        else:
            stats["no_eos"] += 1
            logger.info(f"no eos: \n{canonical_smiles=}\n{out[out.find('[/SMILES]')+len('[/SMILES]'):]}")
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
        outputs, stats_ = process_batch(model, tokenizer, batch, inference_config["gen_config"], tag_pattern)
        generations.update(outputs)
        stats.update(stats_)
        torch.cuda.empty_cache()  # Clear GPU memory after each batch

    save_results(results_path, generations, stats)

    return generations, stats

        

if __name__ == "__main__":
    set_seed(42)    

    with open("molgen3D/config/paths.yaml", "r") as f:
        paths = yaml.safe_load(f)
    
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
        # slurm_gres="shard:80",
        mem_gb=80,
        cpus_per_task=n_gpus * 15,
        slurm_additional_parameters={"partition": node},
    )

    # inference_config = {
    #     "model_path": paths["model_paths"]["new1b"],
    #     "tokenizer_path": paths["tokenizer_path"],
    #     "test_data_path": paths["test_data_path"],
    #     "torch_dtype": "bfloat16", 
    #     "batch_size": 128,
    #     "num_gens": gen_num_codes["2k_per_conf"], 
    #     "gen_config": sampling_configs["top_p_sampling"], 
    #     "device": "cuda",
    #     "results_path": paths["results_path"],
    #     "run_name": "1b_bs128_bf16_t.8p.8",  # Track run details
    # }

    # run locally
    # generations, stats = run_inference(inference_config=inference_config)
    
    # # run on slurm
    # job = executor.submit(run_inference, 
    #                       inference_config=inference_config)

    # run grid search
    param_grid = {
        # "model_path": ["m380_1e", "m380_2e", "m380_3e", "m380_4e"],
        "model_path": ["b1_1e", "b1_2e", "b1_3e", "b1_4e"],
        # "num_gens": list(gen_num_codes.keys()),
        "gen_config": ["min_p_sampling1", "min_p_sampling2", "min_p_sampling3",
                       "top_p_sampling1", "top_p_sampling2", "top_p_sampling3",],
    }

    jobs = []
    with executor.batch():
        for model_key, gen_config_key in itertools.product(*param_grid.values()):
            # Update inference config dynamically
            inference_config = {
                "model_path": paths["model_paths"][model_key],
                "tokenizer_path": paths["tokenizer_path"],
                "test_data_path": paths["test_data_path"],
                "torch_dtype": "bfloat16",
                "batch_size": 256,
                "num_gens": gen_num_codes["2k_per_conf"],
                "gen_config": sampling_configs[gen_config_key], 
                "device": "cuda",
                "results_path": paths["results_path"],
                "run_name": f"{model_key}_{gen_config_key}",  # Track run details
            }

            # Submit job
            job = executor.submit(run_inference, inference_config)
            jobs.append(job)

    
