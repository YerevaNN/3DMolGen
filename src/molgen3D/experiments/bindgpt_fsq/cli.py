"""Single entry point for every stage of the experiment.

    python -m molgen3D.experiments.bindgpt_fsq.cli <stage> [options]

Stages run in the order of the job graph: ``prepare-tokenizers``,
``audit-tokenizers``, ``extract``, ``manifests``, ``build-examples``,
``merge-examples``, ``pack``, ``write-configs``, ``train``,
``generate-unconditional``, ``generate-conditional``, ``metrics-unconditional``,
``metrics-conformer``, ``report``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from molgen3D.experiments.bindgpt_fsq import config
from molgen3D.experiments.bindgpt_fsq.sequences import Arm


def _emit(payload: object) -> None:
    print(json.dumps(payload, indent=1, default=str))


def _data_root(args: argparse.Namespace) -> Path:
    return Path(args.data_root or config.EXPERIMENT_ROOT / "data")


# --------------------------------------------------------------------------- #


def stage_prepare_tokenizers(args: argparse.Namespace) -> None:
    from molgen3D.experiments.bindgpt_fsq import tokenizer_prep

    _emit(tokenizer_prep.prepare_tokenizers(download=not args.offline))


def stage_audit_tokenizers(args: argparse.Namespace) -> None:
    from molgen3D.experiments.bindgpt_fsq import tokenizer_prep

    enriched, smiles = tokenizer_prep.sample_validation_conformers(
        limit=args.conformers, seed=args.seed
    )
    audit_dir = config.EXPERIMENT_ROOT / "tokenizer_audit"
    coord = tokenizer_prep.audit_coord_tokenizer(
        enriched, smiles, max_public_checks=args.public_checks
    )
    (audit_dir / "coord_roundtrip.json").write_text(json.dumps(coord, indent=1))
    fsq = tokenizer_prep.audit_fsq_tokenizer(
        enriched, device=args.device, batch_size=args.batch_size
    )
    (audit_dir / "fsq_roundtrip.json").write_text(json.dumps(fsq, indent=1))
    comparison = tokenizer_prep.write_audit_comparison()
    _emit({"coord": coord, "fsq": fsq, "comparison": comparison})


def stage_extract(args: argparse.Namespace) -> None:
    from molgen3D.experiments.bindgpt_fsq.data import extract_source

    _emit(
        extract_source.extract_split(
            args.split, _data_root(args) / "extracted", limit=args.limit
        )
    )


def stage_manifests(args: argparse.Namespace) -> None:
    from molgen3D.experiments.bindgpt_fsq.data import manifests

    root = _data_root(args) / "extracted"
    splits = {
        "train": manifests.build_split("train", root / "train", workers=args.workers),
        "valid": manifests.build_split("valid", root / "valid", workers=args.workers),
        "benchmark_test": manifests.load_benchmark_test(),
    }
    report = manifests.overlap_report(splits)
    # Match the project's existing SFT data policy: identity intersections are
    # audited and reported, never removed from language-model training.
    applied: Dict[str, int] = {}
    report["training_overlap_policy"] = "audit_only_no_exclusions"

    config.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    summary = manifests.write_molecules_csv(splits, config.MANIFEST_DIR / "molecules.csv")
    graph_rows = manifests.write_graph_index(
        splits, config.MANIFEST_DIR / "graph_index.jsonl"
    )
    report["exclusions_applied"] = applied
    report["split_summary"] = summary
    (config.MANIFEST_DIR / "overlap_report.json").write_text(json.dumps(report, indent=1))
    manifests.write_hashes(
        config.MANIFEST_DIR / "hashes.json",
        extra={"split_summary": summary, "graph_index_rows": graph_rows},
    )
    _emit(
        {
            "split_summary": summary,
            "exclusions_applied": applied,
            "training_overlap_policy": report["training_overlap_policy"],
            "overlap": {
                pair: {key: entry["count"] for key, entry in value.items()}
                for pair, value in report["pairs"].items()
            },
            "fsq_tokenizer_audit": report["fsq_tokenizer_audit"],
        }
    )


def stage_build_examples(args: argparse.Namespace) -> None:
    from molgen3D.experiments.bindgpt_fsq.data import build_examples

    root = _data_root(args)
    _emit(
        build_examples.build_shard(
            root / "extracted",
            args.split,
            args.shard,
            config.MANIFEST_DIR / "graph_index.jsonl",
            root / "examples",
            fsq_device=args.device,
            fsq_batch_size=args.batch_size,
            layout=args.layout,
        )
    )


def stage_merge_examples(args: argparse.Namespace) -> None:
    from molgen3D.experiments.bindgpt_fsq.data import build_examples

    _emit(build_examples.merge_reports(_data_root(args) / "examples", args.split))


def stage_pack(args: argparse.Namespace) -> None:
    from molgen3D.experiments.bindgpt_fsq import dataset, vocabulary
    from molgen3D.experiments.bindgpt_fsq.sequences import UNK

    root = _data_root(args)
    union = vocabulary.build_vocabulary()
    table = union.token_to_id
    reports = {}
    for arm in (Arm.COORD, Arm.FSQ):
        shards = sorted((root / "examples" / args.split).glob(f"{arm.value}_*.jsonl"))
        if not shards:
            raise FileNotFoundError(
                f"no {arm.value} example shards under {root / 'examples' / args.split}"
            )
        reports[arm.value] = dataset.pack_binary(
            shards,
            root / "packed" / args.split,
            table,
            table[UNK],
            arm.value,
        )
    if reports[Arm.COORD.value]["examples"] != reports[Arm.FSQ.value]["examples"]:
        raise RuntimeError(
            "paired example counts differ after packing: "
            f"{reports[Arm.COORD.value]['examples']} vs "
            f"{reports[Arm.FSQ.value]['examples']}"
        )
    _emit(reports)


def stage_write_configs(args: argparse.Namespace) -> None:
    from molgen3D.experiments.bindgpt_fsq.data import manifests

    written = {}
    for arm, run_config in config.default_configs(args.layout).items():
        run_config.training.seed = args.seed
        run_config.decoding.sampling_seed = args.sampling_seed
        run_config.layout = args.layout
        run_config.provenance = {
            "training_code_revision": manifests.git_commit(),
            "evaluation_code_revision": manifests.git_commit(),
            "fsq_code_revision": manifests.git_commit(),
            "slurm_cluster": args.cluster,
        }
        written[arm] = str(run_config.write(config.CONFIG_DIR / f"{arm}_seed{args.seed}.yaml"))
    _emit(written)


def stage_train(args: argparse.Namespace) -> None:
    from molgen3D.experiments.bindgpt_fsq import train as trainer
    from molgen3D.experiments.bindgpt_fsq.config import ExperimentConfig

    run_config = ExperimentConfig.load(args.config)
    root = _data_root(args)
    _emit(
        trainer.train(
            run_config,
            packed_dir=root / "packed" / "train",
            valid_packed_dir=root / "packed" / "valid",
            output_dir=args.output,
        )
    )


def stage_generate_unconditional(args: argparse.Namespace) -> None:
    from molgen3D.experiments.bindgpt_fsq import generate
    from molgen3D.experiments.bindgpt_fsq.config import ExperimentConfig

    run_config = ExperimentConfig.load(args.config)
    sampler = generate.load_sampler(args.checkpoint, run_config, device=args.device)
    destination = (
        Path(args.output)
        if args.output
        else config.GENERATION_DIR
        / "unconditional"
        / run_config.arm
        / f"train_seed_{run_config.training.seed}"
        / f"sample_seed_{args.sampling_seed or run_config.decoding.sampling_seed}"
    )
    _emit(
        generate.generate_unconditional(
            sampler,
            destination,
            target_valid=run_config.unconditional_eval.target_valid_connected,
            max_attempts=run_config.unconditional_eval.max_attempts,
            batch_size=args.batch_size or run_config.unconditional_eval.batch_size,
            seed=args.sampling_seed,
        )
    )


def stage_generate_conditional(args: argparse.Namespace) -> None:
    from molgen3D.experiments.bindgpt_fsq import generate
    from molgen3D.experiments.bindgpt_fsq.config import ExperimentConfig

    run_config = ExperimentConfig.load(args.config)
    sampler = generate.load_sampler(args.checkpoint, run_config, device=args.device)
    molecules = generate.load_test_molecules(args.shard, args.num_shards)
    destination = (
        Path(args.output)
        if args.output
        else config.GENERATION_DIR
        / "conformer"
        / run_config.arm
        / f"train_seed_{run_config.training.seed}"
        / f"sample_seed_{args.sampling_seed or run_config.decoding.sampling_seed}"
    )
    _emit(
        generate.generate_conditional(
            sampler,
            molecules,
            destination,
            generations_per_reference=run_config.conformer_eval.generations_per_reference,
            max_attempt_multiplier=run_config.conformer_eval.max_attempt_multiplier,
            batch_size=args.batch_size or 64,
            seed=args.sampling_seed,
            shard=args.shard,
        )
    )


def stage_metrics_unconditional(args: argparse.Namespace) -> None:
    from molgen3D.experiments.bindgpt_fsq.metrics import unconditional as metrics

    mols, smiles, diagnostics = metrics.load_generated(
        args.attempts, limit=args.population
    )
    generated = metrics.collect_distributions(mols, "generated", parallel=args.parallel)
    output: Dict[str, object] = {"diagnostics": diagnostics}
    for source in ("test", "valid") if args.both_references else (args.reference,):
        reference = metrics.collect_distributions(
            metrics.load_reference(source, limit=args.reference_limit),
            f"geom_{source}",
            parallel=args.parallel,
        )
        output[source] = metrics.compare(reference, generated)
    output["uniqueness_novelty"] = metrics.uniqueness_and_novelty(smiles)

    destination = Path(args.output or config.METRIC_DIR / "unconditional")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{args.tag}.json").write_text(json.dumps(output, indent=1, default=str))
    _emit({key: value for key, value in output.items() if key != "diagnostics"})


def stage_metrics_conformer(args: argparse.Namespace) -> None:
    from molgen3D.experiments.bindgpt_fsq.metrics import conformer as metrics

    paths = sorted(Path(args.generations).glob("attempts_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no attempt shards under {args.generations}")
    generated, diagnostics = metrics.load_generated_conformers(paths)
    result = metrics.evaluate(
        generated,
        threshold=args.threshold,
        use_alignmol=args.use_alignmol,
        num_workers=args.num_workers,
    )
    result["diagnostics"] = diagnostics
    result["stratified"] = metrics.stratify(result["per_molecule"])

    if args.with_posebusters:
        pb = metrics.posebusters(generated, num_workers=args.num_workers or 8)
        pb.pop("_by_smiles", None)
        result["posebusters"] = pb

    destination = Path(args.output or config.METRIC_DIR / "conformer")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{args.tag}.json").write_text(json.dumps(result, indent=1, default=str))
    (destination / f"{args.tag}_per_molecule.json").write_text(
        json.dumps(result["per_molecule"], indent=1, default=str)
    )
    _emit({key: value for key, value in result.items() if key != "per_molecule"})


def stage_metrics_posebusters(args: argparse.Namespace) -> None:
    """PoseBusters validity, as a standalone stage.

    Depends only on the saved conditional generations, so it runs concurrently
    with the COV/MAT job rather than after it.
    """
    from molgen3D.experiments.bindgpt_fsq.metrics import conformer as metrics

    paths = sorted(Path(args.generations).glob("attempts_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no attempt shards under {args.generations}")
    generated, diagnostics = metrics.load_generated_conformers(paths)

    destination = Path(args.output or config.METRIC_DIR / "posebusters")
    destination.mkdir(parents=True, exist_ok=True)

    payload = metrics.posebusters_sharded(
        generated,
        out_dir=destination / f"{args.tag}_shards",
        num_workers=args.num_workers or 8,
        config=args.pb_config,
        molecules_per_shard=args.shard_size,
        shard_indices=(
            None if args.shard_stride <= 1
            else range(args.shard_offset, 10**6, args.shard_stride)
        ),
    )
    payload["diagnostics"] = diagnostics
    if not payload.get("complete") and args.shard_stride > 1:
        # One array task among several: its siblings own the rest.
        _emit(payload)
        return
    if not payload.get("complete"):
        raise RuntimeError(
            f"PoseBusters finished only {payload['shards_done']}/"
            f"{payload['shards_total']} shards; partial aggregate is in "
            f"{destination / (args.tag + '_shards') / 'partial.json'}"
        )
    (destination / f"{args.tag}.json").write_text(
        json.dumps(payload, indent=1, default=str)
    )
    _emit(payload)


def stage_report(args: argparse.Namespace) -> None:
    from molgen3D.experiments.bindgpt_fsq import report as reporter

    _emit(reporter.build(Path(args.output or config.EXPERIMENT_ROOT / "final_report.md")))


# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bindgpt-fsq", description=__doc__)
    parser.add_argument("--data-root", default=None, help="override the data tree root")
    sub = parser.add_subparsers(dest="stage", required=True)

    prepare = sub.add_parser("prepare-tokenizers")
    prepare.add_argument("--offline", action="store_true", help="skip the download")
    prepare.set_defaults(func=stage_prepare_tokenizers)

    audit = sub.add_parser("audit-tokenizers")
    audit.add_argument("--conformers", type=int, default=10_000)
    audit.add_argument("--public-checks", type=int, default=2_000)
    audit.add_argument("--device", default="auto")
    audit.add_argument("--batch-size", type=int, default=64)
    audit.add_argument("--seed", type=int, default=17)
    audit.set_defaults(func=stage_audit_tokenizers)

    extract = sub.add_parser("extract")
    extract.add_argument("--split", choices=["train", "valid"], required=True)
    extract.add_argument("--limit", type=int, default=None)
    extract.set_defaults(func=stage_extract)

    manifest = sub.add_parser("manifests")
    manifest.add_argument("--workers", type=int, default=0)
    manifest.set_defaults(func=stage_manifests)

    build = sub.add_parser("build-examples")
    build.add_argument("--split", choices=["train", "valid"], required=True)
    build.add_argument("--shard", type=int, required=True)
    build.add_argument("--device", default="auto")
    build.add_argument("--batch-size", type=int, default=128)
    build.add_argument("--layout", default="suffix", choices=["suffix", "interleaved"])
    build.set_defaults(func=stage_build_examples)

    merge = sub.add_parser("merge-examples")
    merge.add_argument("--split", choices=["train", "valid"], required=True)
    merge.set_defaults(func=stage_merge_examples)

    pack = sub.add_parser("pack")
    pack.add_argument("--split", choices=["train", "valid"], required=True)
    pack.set_defaults(func=stage_pack)

    configs = sub.add_parser("write-configs")
    configs.add_argument("--seed", type=int, default=17)
    configs.add_argument("--sampling-seed", type=int, default=2_023)
    configs.add_argument("--cluster", default="unknown")
    configs.add_argument("--layout", default="suffix", choices=["suffix", "interleaved"])
    configs.set_defaults(func=stage_write_configs)

    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--output", default=None)
    train.set_defaults(func=stage_train)

    uncond = sub.add_parser("generate-unconditional")
    uncond.add_argument("--config", required=True)
    uncond.add_argument("--checkpoint", required=True)
    uncond.add_argument("--device", default="cuda")
    uncond.add_argument("--batch-size", type=int, default=None)
    uncond.add_argument("--sampling-seed", type=int, default=None)
    uncond.add_argument("--output", default=None)
    uncond.set_defaults(func=stage_generate_unconditional)

    cond = sub.add_parser("generate-conditional")
    cond.add_argument("--config", required=True)
    cond.add_argument("--checkpoint", required=True)
    cond.add_argument("--device", default="cuda")
    cond.add_argument("--batch-size", type=int, default=None)
    cond.add_argument("--sampling-seed", type=int, default=None)
    cond.add_argument("--shard", type=int, default=0)
    cond.add_argument("--num-shards", type=int, default=1)
    cond.add_argument("--output", default=None)
    cond.set_defaults(func=stage_generate_conditional)

    mu = sub.add_parser("metrics-unconditional")
    mu.add_argument("--attempts", required=True)
    mu.add_argument("--tag", required=True)
    mu.add_argument("--population", type=int, default=1_000)
    mu.add_argument("--reference", default="test", choices=["test", "valid"])
    mu.add_argument("--both-references", action="store_true")
    mu.add_argument("--reference-limit", type=int, default=None)
    mu.add_argument("--parallel", action="store_true")
    mu.add_argument("--output", default=None)
    mu.set_defaults(func=stage_metrics_unconditional)

    mc = sub.add_parser("metrics-conformer")
    mc.add_argument("--generations", required=True)
    mc.add_argument("--tag", required=True)
    mc.add_argument("--threshold", type=float, default=0.75)
    mc.add_argument("--use-alignmol", action="store_true")
    mc.add_argument("--num-workers", type=int, default=0)
    mc.add_argument(
        "--with-posebusters",
        action="store_true",
        help="also run PoseBusters here; normally it is its own job (see metrics-posebusters)",
    )
    mc.add_argument("--output", default=None)
    mc.set_defaults(func=stage_metrics_conformer)

    pb = sub.add_parser("metrics-posebusters")
    pb.add_argument("--generations", required=True)
    pb.add_argument("--tag", required=True)
    pb.add_argument("--num-workers", type=int, default=0)
    pb.add_argument("--pb-config", default="mol", choices=["mol", "redock"])
    pb.add_argument("--shard-stride", type=int, default=1,
                    help="number of array tasks sharing the shard list")
    pb.add_argument("--shard-offset", type=int, default=0,
                    help="this task's index within the stride")
    pb.add_argument("--shard-size", type=int, default=25,
                    help="molecules per persisted shard; smaller = less lost on a kill")
    pb.add_argument("--output", default=None)
    pb.set_defaults(func=stage_metrics_posebusters)

    rep = sub.add_parser("report")
    rep.add_argument("--output", default=None)
    rep.set_defaults(func=stage_report)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
