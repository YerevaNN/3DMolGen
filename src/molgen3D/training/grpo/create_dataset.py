import json
import csv

def create_prompt_csv_from_jsonl(
    input_path: str,
    output_path: str,
    smiles_key: str = "canonical_smiles"
):
    """
    Reads a JSONL file, extracts unique SMILES strings from a specified key,
    and writes them to a CSV with a single 'prompt' column.

    Args:
        input_path (str): Path to the input JSONL file.
        output_path (str): Path to the output CSV file.
        smiles_key (str): Key in the JSONL objects containing the SMILES string.
    """
    unique_smiles = set()

    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                entry = json.loads(line)
                smiles = entry.get(smiles_key)
                if smiles:
                    unique_smiles.add(smiles)
            except json.JSONDecodeError:
                continue

    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['prompt'])  # Header
        for smi in sorted(unique_smiles):
            writer.writerow(["[SMILES]"+smi+"[/SMILES]"])

input_file = '/auto/home/menuab/cartesianisomeric/DRUGS/train/train_data_1.jsonl' 
output_file = '/auto/home/menuab/cartesianisomeric/grpo/train_data_1.jsonl'

create_prompt_csv_from_jsonl(input_file, output_file, smiles_key="canonical_smiles")

   


