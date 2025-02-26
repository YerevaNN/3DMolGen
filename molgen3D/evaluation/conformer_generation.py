from transformers import AutoTokenizer, AutoModelForCausalLM
import rdkit
from torch import bfloat16, float32
import torch
from rdkit import Chem
from posebusters import PoseBusters
import json
import cloudpickle
import re
from tqdm import tqdm

from utils import parse_molecule_with_coordinates
torch.backends.cudnn.benchmark = True

cart_1x_path = "/nfs/h100/raid/chem/checkpoints/hf/yerevann/Llama-3.2-1B_conformers/b8f98511589548a093eec1a2/step-4500"
cart_2x_path = "/nfs/h100/raid/chem/checkpoints/hf/yerevann/Llama-3.2-1B_conformers/572934fe0d634f31b3e45925/step-9000"
cart_4x_path = "/nfs/h100/raid/chem/checkpoints/hf/yerevann/Llama-3.2-1B_conformers/4143835e9fdd4a11a7ee1973/step-9000"

test_mols_path = "/mnt/sxtn2/chem/GEOM_data/geom_processed/test_smiles_corrected.csv"
test_smiles_pickle_path = "/auto/home/menuab/code/3DMolGen/drugs_test_inference.pickle"

with open(test_smiles_pickle_path,'rb') as f:
    test_data = cloudpickle.load(f, fix_imports=True)

results_file_pickle = open(f'/auto/home/menuab/code/3DMolGen/evaluation/results/drugs_inference_cart_1xm_2k.pickle','wb') 
results_file_txt = open(f'/auto/home/menuab/code/3DMolGen/evaluation/results/drugs_inference_cart_1xm_2k.txt','w') 

tokenizer  = AutoTokenizer.from_pretrained("/auto/home/menuab/code/YNNtitan/torchtitan/tokenizers/Llama-3.2-chem-1B-v1", padding_side='left')
model = AutoModelForCausalLM.from_pretrained(cart_1x_path, attn_implementation="flash_attention_2", torch_dtype=bfloat16, device_map="cuda:1")
# model = torch.compile(model)
tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = tokenizer.pad_token_id
print(f"{model.dtype=}, {model.device=}")

# test_smiles = [(sample["canonical_smiles"], sample["num_confs"], sample["geom_smiles"]) for sample in test_data.values()]
test_data_values = list(test_data.values())
ref, gen, incs = [], [], []
generations = {}
logs = []
# n= 10 # Batch process multiple molecules at once
batch_size = 1  # Adjust based on GPU memory
for i in tqdm(range(0, len(test_data.keys()), batch_size)):
    # batch = test_data_values[i:i + batch_size]
    batch = [test_data_values[i]]
    n = batch[0]['num_confs'] 
    prompts = [f"[SMILES]{sample['canonical_smiles']}[/SMILES]" for sample in batch]
    # print(prompts)
    tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    # Batch Generation
    outputs = model.generate(
        tokenized_prompts.input_ids,
        max_new_tokens=2000,
        eos_token_id=128329,
        do_sample=True,
        top_p=0.90,
        temperature=0.8,
        num_return_sequences=n,
        # do_sample=False,
        # num_beams=20,              # more beams than outputs desired
        # num_beam_groups=10,        # one group per unique conformer
        # diversity_penalty=3.0,     # adjust between 0.7 and 1.0 as needed
    )

    decoded_outputs = tokenizer.batch_decode(outputs)

    for j, mol_dict in enumerate(batch):
        generated_mols = []
        smiles_mismatch, mol_parse_fail = 0, 0
        canonical_smiles = mol_dict["canonical_smiles"]
        geom_smiles = mol_dict["geom_smiles"]

        for out in decoded_outputs[j * n:(j + 1) * n]:  # Since num_return_sequences=2
            if "[/CONFORMER]" in out:
                generated_conformer = out[out.find("[CONFORMER]")+len("[CONFORMER]"):out.find("[/CONFORMER]")]
                generated_smiles = re.sub(r'<[^>]*>', '', generated_conformer)
                # print(f"{canonical_smiles=}\n{generated_smiles=}")
                if generated_smiles != canonical_smiles:
                    print(f"smiles mismatch: \n{canonical_smiles=}\n{generated_smiles=}")
                    smiles_mismatch += 1
                else:
                    try:
                        mol_obj = parse_molecule_with_coordinates(generated_conformer)
                        generated_mols.append(mol_obj)
                    except:
                        print(f"smiles fails parsing: \n{canonical_smiles=}\n{generated_smiles=}")
                        mol_parse_fail += 1
                
        generations[geom_smiles] = generated_mols
        results_file_txt.write(f"{geom_smiles=} {smiles_mismatch=} {mol_parse_fail=} {canonical_smiles=}\n")

cloudpickle.dump(generations, results_file_pickle, protocol=4)
results_file_txt.close()
results_file_pickle.close()


# ref, gen, incs = [], [], []
# generations = {}
# inc = 0
# for en, mol_dict in enumerate(tqdm(test_smiles)):
#     generated_mols = []
#     smiles_mismatch, mol_parse_fail = 0, 0
#     canonical_smiles = mol_dict["canonical_smiles"]
#     geom_smiles = mol_dict["geom_smiles"]
#     num_generations = mol_dict["num_confs"] * 2
#     print(f"mol num: {en+1} generating {num_generations} conformers for {geom_smiles}")    
#     prompt = f"[SMILES]{canonical_smiles}[/SMILES]"
#     prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
    
#     output = model.generate(prompt,
#                             max_new_tokens=2000,
#                             eos_token_id=128329,
#                             do_sample=True,
#                             top_p=0.90,
#                             temperature=0.8,
#                             num_return_sequences=2)
#     print(f"len prompt toks: {len(prompt[0])}, len gen toks: {len(output[0])-len(prompt[0])}")
#     output = tokenizer.batch_decode(output)
#     # print("raw output: ", output)
#     # print("canonical_smiles: ", canonical_smiles)
#     for out in output:
#         if out.find("[/CONFORMER]"):
#             generated_conformer = out[out.find("[CONFORMER]")+len("[CONFORMER]"):out.find("[/CONFORMER]")]
#             generated_smiles = re.sub(r'<[^>]*>', '', generated_conformer)
#             if generated_smiles != canonical_smiles:
#                 print(f"smiles mismatch: \n{canonical_smiles=}\n{generated_smiles=}")
#                 smiles_mismatch += 1
#             try:
#                 mol_obj = parse_molecule_with_coordinates(generated_conformer)
#                 generated_mols.append(mol_obj)
#             except:
#                 print(f"smiles fails parsing: \n{canonical_smiles=}\n{generated_smiles=}")
#                 mol_parse_fail += 1
        
#     generations[geom_smiles] = generated_mols
#     results_file_txt.write(f"{smiles_mismatch=} {mol_parse_fail=} {canonical_smiles=}\n")
    