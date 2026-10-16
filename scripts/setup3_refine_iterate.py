#!/usr/bin/env python3
"""Apply another tokenizer encode/decode pass to an existing refinement round.

Round 1 asked what one pass does to Setup 3's interpolants. This asks whether
the effect converges: a round-1 output is itself a tokenizer output, so if the
first pass reached the manifold a second should be close to a no-op, and if it
is not, the first pass did not get there.

Condition names are preserved, so round N's directory is scored by exactly the
same commands as round N-1 and the rounds are compared directory to directory.

Usage:
  setup3_refine_iterate.py --source-dir <round N-1> --output-dir <round N>
                           --fsq-checkpoint <ckpt> [--conditions ...]
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from molgen3D.evaluation.conformer_interpolation import (
    _atomic_json_dump,
    _atomic_pickle_dump,
)
from molgen3D.evaluation.setup3_tokenizer_refinement import refine_molecules

DEFAULT_CONDITIONS = (
    "anchors_rt",
    "latent_rt_all",
    "random_token_rt_all",
    "cartesian_rt_all",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fsq-checkpoint", required=True)
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int, default=218)
    parser.add_argument("--model-batch-size", type=int, default=128)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    (output_dir / "molecules").mkdir(parents=True, exist_ok=True)
    paths = sorted((source_dir / "molecules").glob("*.pickle"))
    if not paths:
        raise RuntimeError(f"No molecule records under {source_dir}")

    _atomic_json_dump(
        {
            "source_dir": str(source_dir),
            "fsq_checkpoint": str(args.fsq_checkpoint),
            "conditions": list(args.conditions),
            "note": "one further encode/decode pass over each named condition",
        },
        output_dir / "config.json",
    )

    from molgen3D.data_processing.smiles_encoder_decoder_fsq import configure_fsq_encoder

    model = configure_fsq_encoder(
        ckpt_path=args.fsq_checkpoint, device=args.device, max_tokens=args.max_tokens
    )

    totals = {"n": 0, "encode_failures": 0, "decode_failures": 0}
    for position, path in enumerate(paths, start=1):
        out_path = output_dir / "molecules" / path.name
        if args.resume and out_path.exists():
            continue
        with path.open("rb") as handle:
            record = pickle.load(handle)
        if record.get("status") != "complete":
            _atomic_pickle_dump(
                {"molecule_key": record.get("molecule_key"), "status": record.get("status")},
                out_path,
            )
            continue
        conformers = {}
        statistics = {}
        for condition in args.conditions:
            mols = record["conformers"].get(condition)
            if not mols:
                continue
            refined, stats = refine_molecules(model, mols, batch_size=args.model_batch_size)
            conformers[condition] = refined
            statistics[condition] = stats
            for name in totals:
                totals[name] += int(stats.get(name, 0))
        _atomic_pickle_dump(
            {
                "molecule_key": record["molecule_key"],
                "status": "complete",
                "k": record.get("k"),
                "n_anchor": record.get("n_anchor"),
                "variant": record.get("variant"),
                "conformers": conformers,
                "statistics": statistics,
            },
            out_path,
        )
        if position % args.log_every == 0 or position == len(paths):
            print(f"refined {position}/{len(paths)}", flush=True)

    _atomic_json_dump({"refine_statistics": totals}, output_dir / "refine_summary.json")
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == "__main__":
    main()
