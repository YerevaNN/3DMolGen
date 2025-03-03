from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
import rdkit
from torch import bfloat16, float32
import torch
from rdkit import Chem
from posebusters import PoseBusters
import json
import cloudpickle
import random
import re
from tqdm import tqdm
from loguru import logger
from collections import defaultdict, Counter
import submitit
import os
import ast
import itertools
from datetime import datetime


# from utils import parse_molecule_with_coordinates
torch.backends.cudnn.benchmark = True

greedy_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=3000,
    eos_token_id=128329,
)

beam_search_config_b4_g1_d0 = GenerationConfig(
    num_beams=4,  
    num_beam_groups=1,
    diversity_penalty=0,
    max_new_tokens=3000,
    eos_token_id=128329,
)

beam_search_config_b4_g2_d1 = GenerationConfig(
    num_beams=4,  
    num_beam_groups=2,
    diversity_penalty=0.1,
    max_new_tokens=3000,
    eos_token_id=128329,
)

beam_search_config_b4_g2_d2 = GenerationConfig(
    num_beams=4,  
    num_beam_groups=2,
    diversity_penalty=0.2,
    max_new_tokens=3000,
    eos_token_id=128329,
)

beam_search_config_b4_g2_d3 = GenerationConfig(
    num_beams=4,  
    num_beam_groups=2,
    diversity_penalty=0.3,
    max_new_tokens=3000,
    eos_token_id=128329,
)

beam_search_config_b4_g4_d2 = GenerationConfig(
    num_beams=4,  
    num_beam_groups=2,
    diversity_penalty=0.2,
    max_new_tokens=3000,
    eos_token_id=128329,
)

beam_search_config_b4_g4_d3 = GenerationConfig(
    num_beams=4,  
    num_beam_groups=4,
    diversity_penalty=0.3,
    max_new_tokens=3000,
    eos_token_id=128329,
)

beam_search_config_b8_g2_d2 = GenerationConfig(
    num_beams=8,  
    num_beam_groups=2,
    diversity_penalty=0.2,
    max_new_tokens=3000,
    eos_token_id=128329,
)

beam_search_config_b8_g4_d2 = GenerationConfig(
    num_beams=8,  
    num_beam_groups=4,
    diversity_penalty=0.2,
    max_new_tokens=3000,
    eos_token_id=128329,
)

beam_search_config_b8_g8_d2 = GenerationConfig(
    num_beams=8,  
    num_beam_groups=8,
    diversity_penalty=0.2,
    max_new_tokens=3000,
    eos_token_id=128329,
)

top_p_sampling_config = GenerationConfig (
  do_sample=True,
  eos_token_id=128329,
  max_new_tokens=3000,
  temperature=0.8,
  top_p=0.9
)
top_p_sampling_config_p9_t9_k40 = GenerationConfig(
    do_sample=True,
    top_p=0.9,  
    temperature=0.9,
    top_k=40,
    max_new_tokens=3000,
    eos_token_id=128329,
)

top_p_sampling_config_p9_t8_k40 = GenerationConfig(
    do_sample=True,
    top_p=0.9,  
    temperature=0.8,
    top_k=40,
    max_new_tokens=3000,
    eos_token_id=128329,
)

top_p_sampling_config_p9_t1_k40 = GenerationConfig(
    do_sample=True,
    top_p=0.8,  
    temperature=1.0,
    top_k=40,
    max_new_tokens=3000,
    eos_token_id=128329,
)

top_p_sampling_config_p9_t11_k40 = GenerationConfig(
    do_sample=True,
    top_p=0.9,  
    temperature=1.1,
    top_k=40,
    max_new_tokens=3000,
    eos_token_id=128329,
)

min_p_sampling_config_p05= GenerationConfig(
    do_sample=True,
    min_p = 0.05,
    max_new_tokens=3000,
    eos_token_id=128329,
)

min_p_sampling_config_p10= GenerationConfig(
    do_sample=True,
    min_p = 0.1,
    max_new_tokens=3000,
    eos_token_id=128329,
)
min_p_sampling_config_p05_t1= GenerationConfig(
    do_sample=True,
    min_p = 0.05,
    temperature=1.0,
    max_new_tokens=3000,
    eos_token_id=128329,
)
min_p_sampling_config_p05_t11= GenerationConfig(
    do_sample=True,
    min_p = 0.05,
    temperature=1.1,
    max_new_tokens=3000,
    eos_token_id=128329,
)
min_p_sampling_config_p05_t12= GenerationConfig(
    do_sample=True,
    min_p = 0.05,
    temperature=1.2,
    max_new_tokens=3000,
    eos_token_id=128329,
)
min_p_sampling_config_p05_t13= GenerationConfig(
    do_sample=True,
    min_p = 0.05,
    temperature=1.3,
    max_new_tokens=3000,
    eos_token_id=128329,
)

min_p_sampling_config_p10_t1= GenerationConfig(
    do_sample=True,
    min_p = 0.1,
    temperature=1.0,
    max_new_tokens=3000,
    eos_token_id=128329,
)
min_p_sampling_config_p10_t11= GenerationConfig(
    do_sample=True,
    min_p = 0.1,
    temperature=1.1,
    max_new_tokens=3000,
    eos_token_id=128329,
)
min_p_sampling_config_p10_t12= GenerationConfig(
    do_sample=True,
    min_p = 0.1,
    temperature=1.2,
    max_new_tokens=3000,
    eos_token_id=128329,
)
min_p_sampling_config_p10_t13= GenerationConfig(
    do_sample=True,
    min_p = 0.1,
    temperature=1.3,
    max_new_tokens=3000,
    eos_token_id=128329,
)

min_p_sampling_config_p10_t1= GenerationConfig(
    do_sample=True,
    min_p = 0.1,
    temperature=1.0,
    max_new_tokens=3000,
    eos_token_id=128329,
)
min_p_sampling_config_p10_t11= GenerationConfig(
    do_sample=True,
    min_p = 0.1,
    temperature=1.1,
    max_new_tokens=3000,
    eos_token_id=128329,
)
min_p_sampling_config_p10_t12= GenerationConfig(
    do_sample=True,
    min_p = 0.1,
    temperature=1.2,
    max_new_tokens=3000,
    eos_token_id=128329,
)
min_p_sampling_config_p10_t13= GenerationConfig(
    do_sample=True,
    min_p = 0.1,
    temperature=1.3,
    max_new_tokens=3000,
    eos_token_id=128329,
)

min_p_sampling_config_p20_t1= GenerationConfig(
    do_sample=True,
    min_p = 0.2,
    temperature=1.0,
    max_new_tokens=3000,
    eos_token_id=128329,
)
min_p_sampling_config_p20_t11= GenerationConfig(
    do_sample=True,
    min_p = 0.2,
    temperature=1.1,
    max_new_tokens=3000,
    eos_token_id=128329,
)
min_p_sampling_config_p20_t12= GenerationConfig(
    do_sample=True,
    min_p = 0.2,
    temperature=1.2,
    max_new_tokens=3000,
    eos_token_id=128329,
)
min_p_sampling_config_p20_t13= GenerationConfig(
    do_sample=True,
    min_p = 0.2,
    temperature=1.3,
    max_new_tokens=3000,
    eos_token_id=128329,
)


def parse_molecule_with_coordinates(input_str):
    # Extract SMILES by removing coordinate annotations
    extracted_smiles = re.sub(r'<[^>]+>', '', input_str)
    
    # Parse the extracted SMILES
    mol = Chem.AddHs(Chem.MolFromSmiles(extracted_smiles))

    if mol is None:
        raise ValueError("Failed to parse the extracted SMILES.")
    canonical = Chem.MolToSmiles(mol, canonical=True, allHsExplicit=True)
    
    # Retrieve the atom output order from the molecule's properties
    if not mol.HasProp('_smilesAtomOutputOrder'):
        raise ValueError("SMILES atom output order not found.")
    atom_output_order = ast.literal_eval(mol.GetProp('_smilesAtomOutputOrder'))
    
    # Parse coordinates from the input string
    coords = []
    atom_pattern = re.compile(r'\[([^<]+)<([^>]+)>\]')
    for match in atom_pattern.finditer(input_str):
        coord_str = match.group(2)
        coord = list(map(float, coord_str.split(',')))
        coords.append(coord)
    
    # Verify coordinate count matches atom count
    if len(coords) != mol.GetNumAtoms():
        raise ValueError("Mismatch between number of coordinates and atoms.")
    
    # Create conformer and assign coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for input_idx, atom_idx in enumerate(atom_output_order):
        x, y, z = coords[input_idx]
        conf.SetAtomPosition(atom_idx, Chem.rdGeometry.Point3D(x, y, z))
    mol.AddConformer(conf)
    
    return mol

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

    gen_num_codes = {
        # "1x_per_mol": 1,
        # "2x_per_mol": 2,
        # "1k_per_conf": "1k",
        "2k_per_conf": "2k"
    }
    GENERATION_CONFIGS = {
        # "greedy": greedy_config,
        "top_p": top_p_sampling_config,
        # "top_p_p9_t9_k40": top_p_sampling_config_p9_t9_k40,
        # "top_p_p9_t8_k40": top_p_sampling_config_p9_t8_k40,
        # "top_p_p9_t1_k40": top_p_sampling_config_p9_t1_k40,
        "top_p_p9_t11_k40": top_p_sampling_config_p9_t11_k40,
        # "beam_search_b4_g1_d0": beam_search_config_b4_g1_d0,
        # "beam_search_config_b4_g2_d1": beam_search_config_b4_g2_d1,
        # "beam_search_config_b4_g2_d2": beam_search_config_b4_g2_d2,
        # "beam_search_config_b4_g2_d3": beam_search_config_b4_g2_d3,
        # "beam_search_config_b4_g4_d2": beam_search_config_b4_g4_d2,
        # "beam_search_config_b4_g4_d3": beam_search_config_b4_g4_d3,
        # "beam_search_config_b8_g2_d2": beam_search_config_b8_g2_d2,
        # "beam_search_config_b8_g4_d2": beam_search_config_b8_g4_d2,
        # "beam_search_config_b8_g8_d2": beam_search_config_b8_g8_d2,
        "min_p_sampling_config_p05": min_p_sampling_config_p05,
        "min_p_sampling_config_p10": min_p_sampling_config_p10,
        "min_p_sampling_config_p05_t1": min_p_sampling_config_p05_t1,
        "min_p_sampling_config_p05_t11": min_p_sampling_config_p05_t11,
        "min_p_sampling_config_p05_t12": min_p_sampling_config_p05_t12,
        "min_p_sampling_config_p05_t13": min_p_sampling_config_p05_t13,
        "min_p_sampling_config_p10_t1": min_p_sampling_config_p10_t1,
        "min_p_sampling_config_p10_t11": min_p_sampling_config_p10_t11,
        "min_p_sampling_config_p10_t12": min_p_sampling_config_p10_t12,
        "min_p_sampling_config_p10_t13": min_p_sampling_config_p10_t13,
        "min_p_sampling_config_p20_t1": min_p_sampling_config_p20_t1,
        "min_p_sampling_config_p20_t11": min_p_sampling_config_p20_t11,
        "min_p_sampling_config_p20_t12": min_p_sampling_config_p20_t12,
        "min_p_sampling_config_p20_t13": min_p_sampling_config_p20_t13,
    }
    model_paths = {
        # "cart_1e_path": "/nfs/h100/raid/chem/checkpoints/hf/yerevann/Llama-3.2-1B_conformers/c037e75255bc41c19c716939/step-4500",
        # "cart_2e_path": "/nfs/h100/raid/chem/checkpoints/hf/yerevann/Llama-3.2-1B_conformers/d267db61f57d4b428baa604a/step-9000",
        # "cart_4e_path": "/nfs/h100/raid/chem/checkpoints/hf/yerevann/Llama-3.2-1B_conformers/3408e9758572478c80393771/step-18000",
        # "cart_6e_path": "/nfs/h100/raid/chem/checkpoints/hf/yerevann/Llama-3.2-1B_conformers/301b8328481243c6aa8d8003/step-27000",
        "cart_8e_path": "/nfs/h100/raid/chem/checkpoints/hf/yerevann/Llama-3.2-1B_conformers/c13311b27056459eaccf5877/step-36000"

    }
    tokenizer_path = "/auto/home/menuab/code/YNNtitan/torchtitan/tokenizers/Llama-3.2-chem-1B-v1"
    test_data_path = "/auto/home/menuab/code/3DMolGen/drugs_test_inference.pickle"
    results_path = "/auto/home/menuab/code/3DMolGen/gen_results"
    results_path = os.path.join(results_path, f"{datetime.now().strftime('%Y-%m-%d-%H:%M')}")
    
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

    set_seed(42)    
    # inference_config = {
    #     "model_path": model_paths["cart_4e_path"],
    #     "tokenizer_path": tokenizer_path,
    #     "test_data_path": test_data_path,
    #     "batch_size": 100,
    #     "num_gens": gen_num_codes["1k_per_conf"],
    #     "gen_config": GENERATION_CONFIGS["beam_search_config_b4_g2_d1"], 
    #     "device": "cuda:0",
    #     "results_path": results_path,
    #     "run_name": "16e_1k_beam",
    # }

    # generations, stats = run_inference(inference_config=inference_config)
    # job = executor.submit(run_inference, 
    #                       inference_config=inference_config)

    

    param_grid = {
        "model_path": list(model_paths.keys()),
        "num_gens": list(gen_num_codes.keys()),
        "gen_config": list(GENERATION_CONFIGS.keys()),
    }

    jobs = []
    with executor.batch():
        for model_key, num_gens_key, gen_config_key in itertools.product(*param_grid.values()):
            # Update inference config dynamically
            inference_config = {
                "model_path": model_paths[model_key],
                "tokenizer_path": tokenizer_path,
                "test_data_path": test_data_path,
                "batch_size": 200,
                "num_gens": gen_num_codes[num_gens_key],
                "gen_config": GENERATION_CONFIGS[gen_config_key], 
                "device": "auto",
                "results_path": results_path,
                "run_name": f"{model_key}_{num_gens_key}_{gen_config_key}",  # Track run details
            }

            # Submit job
            job = executor.submit(run_inference, inference_config)
            jobs.append(job)
