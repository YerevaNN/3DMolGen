import os
import pickle
from posebusters import PoseBusters
import pandas as pd
import random
from concurrent.futures import ProcessPoolExecutor
import itertools
import submitit

def _bust_smi(smi, mols, config, full_report):
    try:
        b = PoseBusters(config=config)
        # run checks and collect per‐conformer results
        dfb = b.bust(mols, None, None, full_report=full_report)
        # average per-test metrics
        m = dfb.mean().to_dict()
        # percent of conformers that passed all tests
        m['pass_percentage'] = dfb.all(axis=1).mean() * 100
        m['smiles'] = smi
        m['error'] = ''
        return m
    except Exception as e:
        return {'smiles': smi, 'error': str(e)}

def run_all_posebusters(data, config="mol", full_report=False,
                        max_workers=16, fail_threshold=0.0):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            _bust_smi,
            data.keys(),
            data.values(),
            itertools.repeat(config),
            itertools.repeat(full_report),
        ))
    df = pd.DataFrame(results)
    error_smiles = df.loc[df['error'] != '', 'smiles'].tolist()
    if 'failure_rate' in df.columns:
        bad = df['failure_rate'] > fail_threshold
        fail_smiles = df.loc[bad, 'smiles'].tolist()
    else:
        fail_smiles = []
    summary = df[df['error']==''].mean(numeric_only=True).to_frame().T
    summary.insert(0, 'smiles', 'ALL')

    df.to_csv("posebusters_detailed.csv", index=False)
    summary.to_csv("posebusters_summary.csv", index=False)
    with open("posebusters_failures.txt", "w") as f:
        f.write("\n".join(fail_smiles))
    with open("posebusters_errors.txt", "w") as f:
        f.write("\n".join(error_smiles))

    return df, summary, fail_smiles, error_smiles

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PoseBusters on a dataset.")
    parser.add_argument("data_path", type=str, help="Path to the data file (pickle format).")
    parser.add_argument("--config", type=str, default="mol", help="Configuration for PoseBusters.")
    parser.add_argument("--full_report", action='store_true', help="Generate full report.")
    parser.add_argument("--max_workers", type=int, default=16, help="Number of parallel workers.")
    parser.add_argument("--fail_threshold", type=float, default=0.0,
                        help="Max allowed failure_rate per SMILES.")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of SMILES to randomly sample.")
    parser.add_argument("--use_submitit", action="store_true",
                        help="Submit the job to SLURM via submitit.")
    parser.add_argument("--mem_gb", type=int, default=16,
                        help="Memory (GB) per job.")
    parser.add_argument("--timeout_min", type=int, default=60,
                        help="Timeout (minutes) for job.")

    args = parser.parse_args()

    data = load_data(args.data_path)

    if args.sample_size:
        items = list(data.items())
        data = dict(random.sample(items, k=min(args.sample_size, len(items))))
        print(f"Sampled {len(data)} SMILES for processing.")

    if args.use_submitit:
        executor = submitit.AutoExecutor(folder="~/slurm_jobs/posebusters_%j")
        executor.update_parameters(
            name="posebusters",
            timeout_min=args.timeout_min,
            cpus_per_task=args.max_workers,
            mem_gb=args.mem_gb,
            nodes=1,
            slurm_partition="h100",
        )
        job = executor.submit(
            run_all_posebusters,
            data,
            args.config,
            args.full_report,
            args.max_workers,
            args.fail_threshold
        )
        print(f"Submitted Submitit job with ID {job.job_id}")
        exit(0)

    df, summary, fail_smiles, error_smiles = run_all_posebusters(
        data,
        config=args.config,
        full_report=args.full_report,
        max_workers=args.max_workers,
        fail_threshold=args.fail_threshold
    )

    out_dir = os.path.dirname(os.path.abspath(args.data_path))
    df.to_csv(os.path.join(out_dir, "posebusters_detailed.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "posebusters_summary.csv"), index=False)

    if fail_smiles:
        with open(os.path.join(out_dir, "posebusters_failures.txt"), "w") as f:
            f.write("\n".join(fail_smiles))

    if error_smiles:
        with open(os.path.join(out_dir, "posebusters_errors.txt"), "w") as f:
            f.write("\n".join(error_smiles))

    print(f"Reports saved to {out_dir}")