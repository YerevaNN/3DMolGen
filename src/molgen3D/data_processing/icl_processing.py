import argparse
import cloudpickle
import os
import glob
import random
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from molgen3D.data_processing.smiles_encoder_decoder import encode_cartesian_v2, encode_cartesian_binned

def load_pkl(path):
    with open(path, 'rb') as f:
        return cloudpickle.load(f)

def main():
    parser = argparse.ArgumentParser(description="Add In-Context Learning (ICL) prompts to QM9 data.")
    parser.add_argument("--input_pkl", type=str, default="/auto/home/vover/3DMolGen/data/tmp/qm9_smi.pickle")
    parser.add_argument("--qm9_dir", type=str, default="/nfs/ap/mnt/sxtn2/chem/GEOM_data/torsional_diff_gdrive/extracted/QM9/qm9/")
    parser.add_argument("--split_path", type=str, default="/nfs/ap/mnt/sxtn2/chem/GEOM_data/torsional_diff_gdrive/extracted/QM9/split.npy")
    parser.add_argument("--binned", action="store_true", help="Use binned coordinates")
    parser.add_argument("--bin_size", type=float, default=0.104, help="Bin size for binned coordinates")
    parser.add_argument("--range_k", type=float, default=13.0, help="Range k for binned coordinates ([-k, k])")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of SMILES to sample for ICL")
    parser.add_argument("--coord_filter_range", type=float, default=None, help="Range to filter out conformers (e.g. 13.0)")
    parser.add_argument("--output_pkl", type=str, default="/auto/home/vover/3DMolGen/data/icl_20_smi.pickle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dynamic_n", action="store_true", help="If set, sample a random number of prompts up to n_samples")
    parser.add_argument("--drop_dotted", action="store_true", help="If set, drop SMILES containing dots from input and training data")
    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.input_pkl):
        print(f"Error: Input file {args.input_pkl} not found.")
        return
    
    print(f"Loading input data from {args.input_pkl}...")
    data = load_pkl(args.input_pkl)
    
        
    print(f"Loading splits from {args.split_path}...")
    splits = np.load(args.split_path, allow_pickle=True)
    # splits[0] is typically the training set indices
    train_indices = splits[0]
    
    # Get all training files
    print(f"Identifying training files in {args.qm9_dir}...")
    all_pickle_paths = np.array(sorted(glob.glob(os.path.join(args.qm9_dir, "*.pickle"))))
    
    if len(all_pickle_paths) == 0:
        print(f"Error: No pickle files found in {args.qm9_dir}")
        return
        
    training_files = all_pickle_paths[train_indices]
    
    print(f"Loaded {len(data)} SMILES from input.")
    
    # Drop dotted SMILES
    if args.drop_dotted:
        data = {k: v for k, v in data.items() if '.' not in k}
        print(f"After dropping dotted SMILES: {len(data)} SMILES remaining.")
    
    print(f"Found {len(training_files)} training molecules.")

    ranges = [(-args.range_k, args.range_k)] * 3 if args.range_k else None

    # Process each SMILES in the input
    for smiles_key in tqdm(data, desc="Processing SMILES"):
        entry = data[smiles_key]
        
        # Determine number of prompts
        if args.dynamic_n:
            n_prompts = random.randint(1, args.n_samples)
        else:
            n_prompts = args.n_samples
            
        # Sample training molecules
        # Ensure we don't try to sample more than available
        sampled_files = random.sample(list(training_files), n_prompts)
        
        icl_prompts = []
        for f_path in sampled_files:
            try:
                train_mol_data = load_pkl(f_path)
                
                # Skip training molecules with dots if requested
                if args.drop_dotted and '.' in train_mol_data.get('smiles', ''):
                    continue

                confs = train_mol_data['conformers']
                
                # Filter conformers if needed
                if args.coord_filter_range is not None:
                    valid_confs = []
                    for c in confs:
                        mol_obj = c['rd_mol']
                        pos = mol_obj.GetConformer().GetPositions()
                        if np.all(pos >= -args.coord_filter_range) and np.all(pos <= args.coord_filter_range):
                            valid_confs.append(c)
                    confs = valid_confs
                
                if not confs:
                    continue
                
                # Choose 1 random conformer
                conf = random.choice(confs)
                mol = conf['rd_mol']
                
                # Encode
                if args.binned:
                    enriched, canon_smiles = encode_cartesian_binned(mol, bin_size=args.bin_size, ranges=ranges)
                else:
                    enriched, canon_smiles = encode_cartesian_v2(mol)
                
                # Format: [SMILES]smiles[/SMILES][CONFORMER]enriched_string[/CONFORMER]
                icl_prompts.append(f"[SMILES]{canon_smiles}[/SMILES][CONFORMER]{enriched}[/CONFORMER]\n")
            except Exception as e:
                continue
        
        # Construct the ICL prompt prefix
        icl_prefix = "".join(icl_prompts)  if icl_prompts else ""
        
        # Prepend it to the prompt field
        target_smiles = entry.get('geom_smiles', smiles_key)
        target_part = f"[SMILES]{target_smiles}[/SMILES][CONFORMER]"
        
        entry['icl_n_samples'] = n_prompts
        entry['icl_prefix'] = icl_prefix 
        entry['icl_prompt'] = icl_prefix + target_part
        entry['prompt'] = target_part

    # Save output
    os.makedirs(os.path.dirname(args.output_pkl), exist_ok=True)
    with open(args.output_pkl, 'wb') as f:
        cloudpickle.dump(data, f)
    
    print(f"Saved results with new fields 'icl_prompt' and 'prompt' to {args.output_pkl}")

if __name__ == "__main__":
    main()
