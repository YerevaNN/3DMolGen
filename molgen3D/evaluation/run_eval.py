import pickle
import os
from covmat import CovMatEvaluator, print_covmat_results
from posebustars_check import run_all_posebusters
from loguru import logger as log

with open("/auto/home/menuab/code/3DMolGen/data/geom_drugs_test_set/drugs_test_inference.pickle", 'rb') as f:
    true_mols = pickle.load(f)

gens_directory = "2025-07-15-03:41_m380_1e_1xgrpo_topp1_128bs"
gens_path = os.path.join("/auto/home/menuab/code/3DMolGen/gen_results/", gens_directory)

results_path = os.path.join("/auto/home/menuab/code/3DMolGen/eval_results", gens_directory)
os.makedirs(results_path, exist_ok=True)
results_file = open(os.path.join(results_path, "covmat_results.txt"), "w")

def find_pickle_files(base_dir):
    """Finds all .pickle files recursively and returns their relative paths."""
    pickle_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".pickle"):
                relative_path = os.path.relpath(os.path.join(root, file), base_dir)
                pickle_files.append(relative_path)
    return pickle_files

pickles = find_pickle_files(gens_path)
for pickle_path in sorted(pickles):
    print(os.path.join(gens_path, pickle_path))
    with open(os.path.join(gens_path, pickle_path), 'rb') as file:
        model_preds = pickle.load(file)
    evaluator = CovMatEvaluator(num_workers=2)
    results, rmsd_results, missing = evaluator(ref_data=true_mols, gen_data=model_preds)
    df, summary, fail_smiles, error_smiles = run_all_posebusters(model_preds, config="mol",max_workers=2)
    # print(results)
    log.info("Evaluation finished...")

    # get dataframe of results
    cov_df, matching_metrics = print_covmat_results(results)

    results_file.writelines(f"resutls for {pickle_path}\nnumber of missing mols {len(missing)}\n{missing}\n"\
                            f"{matching_metrics}\n{cov_df.iloc[14]}\n")
results_file.close()
    