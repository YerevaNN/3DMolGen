import json
import random
from pathlib import Path

def sanitize_smiles_for_filename(smiles: str) -> str:
    """
    Sanitize SMILES string to match the pickle filename format.
    Replaces characters that can't be used in filenames.
    """
    # Based on the pickle files observed, special characters like /, =, \ are replaced with _
    sanitized = smiles
    sanitized = sanitized.replace('/', '_')
    sanitized = sanitized.replace('\\', '_')
    sanitized = sanitized.replace('=', '=')  # Keep = as is based on examples
    # Add more replacements if needed based on actual file naming
    return sanitized

def create_geom_smiles_datasets(
    geom_mapping_path: str,
    pickle_dir: str,
    output_smiles_jsonl: str,
    output_mapping_jsonl: str
):
    """
    Reads the GEOM train smiles mapping, creates:
    1. A JSONL file with unique isomeric SMILES wrapped in tags
    2. A JSONL mapping from isomeric SMILES to pickle file paths

    Args:
        geom_mapping_path: Path to train_geom_to_isomeric_smiles.jsonl
        pickle_dir: Path to the directory containing pickle files
        output_smiles_jsonl: Path for output JSONL with tagged SMILES
        output_mapping_jsonl: Path for output JSONL with SMILES to pickle mapping
    """
    unique_smiles = set()
    smiles_to_geom = {}  # Maps isomeric_smiles to geom_smiles

    # Read the GEOM mapping file
    print(f"Reading GEOM mapping from {geom_mapping_path}")
    with open(geom_mapping_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                entry = json.loads(line)
                geom_smiles = entry.get('geom_smiles')
                isomeric_smiles = entry.get('isomeric_smiles')

                if geom_smiles and isomeric_smiles:
                    unique_smiles.add(isomeric_smiles)
                    # Store the geom_smiles for finding the pickle file
                    if isomeric_smiles not in smiles_to_geom:
                        smiles_to_geom[isomeric_smiles] = geom_smiles
            except json.JSONDecodeError:
                continue

    print(f"Found {len(unique_smiles)} unique isomeric SMILES")

    # Create the SMILES JSONL file with tags (shuffled)
    print(f"Writing shuffled tagged SMILES to {output_smiles_jsonl}")
    smiles_list = list(unique_smiles)
    random.shuffle(smiles_list)
    with open(output_smiles_jsonl, 'w', encoding='utf-8') as outfile:
        for smiles in smiles_list:
            tagged_smiles = f"[SMILES]{smiles}[/SMILES]"
            outfile.write(tagged_smiles + '\n')

    # Create the mapping JSON file (SMILES -> pickle path)
    # Build mapping by iterating through pickle files to avoid filename length issues
    print(f"Creating SMILES to pickle mapping at {output_mapping_jsonl}")
    print("Building pickle filename index...")
    pickle_path = Path(pickle_dir)

    # Create a dictionary mapping geom_smiles to pickle file path
    geom_to_pickle = {}
    for pickle_file in pickle_path.glob('*.pickle'):
        # Extract the SMILES from the filename (remove .pickle extension)
        geom_smiles_from_file = pickle_file.stem
        geom_to_pickle[geom_smiles_from_file] = str(pickle_file)

    print(f"Found {len(geom_to_pickle)} pickle files")

    mapping_count = 0
    missing_count = 0
    smiles_to_pickle_mapping = {}
    missing_examples = []

    for isomeric_smiles in sorted(unique_smiles):
        geom_smiles = smiles_to_geom[isomeric_smiles]

        # Look up the pickle file path using the geom_smiles
        # First try exact match
        if geom_smiles in geom_to_pickle:
            smiles_to_pickle_mapping[isomeric_smiles] = geom_to_pickle[geom_smiles]
            mapping_count += 1
        else:
            # Try with only / replaced by _ (backslash is kept as-is in pickle filenames)
            sanitized_geom = geom_smiles.replace('/', '_')
            if sanitized_geom in geom_to_pickle:
                smiles_to_pickle_mapping[isomeric_smiles] = geom_to_pickle[sanitized_geom]
                mapping_count += 1
            else:
                missing_count += 1
                if len(missing_examples) < 5:
                    missing_examples.append((isomeric_smiles, geom_smiles))

    # Write the mapping as a JSON file
    with open(output_mapping_jsonl, 'w', encoding='utf-8') as outfile:
        json.dump(smiles_to_pickle_mapping, outfile, indent=2)

    print(f"Created {mapping_count} SMILES to pickle mappings")
    if missing_count > 0:
        print(f"\nWarning: {missing_count} SMILES did not have corresponding pickle files")
        print(f"\nMissing examples (first 5):")
        for i, (iso_smi, geom_smi) in enumerate(missing_examples, 1):
            print(f"\n  Example {i}:")
            print(f"    Isomeric SMILES: {iso_smi}")
            print(f"    Geom SMILES:     {geom_smi}")
            # Show what the sanitized version would be
            sanitized = geom_smi.replace('/', '_')
            print(f"    Sanitized:       {sanitized}")
            print(f"    Length:          {len(geom_smi)} chars")

    print("Done!")

if __name__ == "__main__":
    # GEOM dataset paths
    geom_mapping_file = '/nfs/ap/mnt/sxtn2/chem/GEOM_data/geom_processed/geom_cartesian_v3/train_geom_to_isomeric_smiles.jsonl'
    pickle_directory = '/nfs/ap/mnt/sxtn2/chem/GEOM_data/geom_processed/geom_cartesian_v3/processed_pickles/train/'

    # Output paths
    output_smiles_file = '/nfs/ap/mnt/sxtn2/chem/GEOM_data/geom_processed/grpo/train_smiles.jsonl'
    output_mapping_file = '/nfs/ap/mnt/sxtn2/chem/GEOM_data/geom_processed/grpo/train_smiles_to_pickle.json'

    create_geom_smiles_datasets(
        geom_mapping_path=geom_mapping_file,
        pickle_dir=pickle_directory,
        output_smiles_jsonl=output_smiles_file,
        output_mapping_jsonl=output_mapping_file
    )

   


