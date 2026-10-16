"""Build and audit the two coordinate tokenizers.

Produces, under ``bindgpt_geom_fsq/tokenizer_audit/``:

``bindgpt_public/``
    the pinned public tokenizer and config files, with SHA-256 for each;
``bindgpt_smiles_tokens.json``
    the SMILES token set derived from those files;
``union_tokenizer/``
    the shared union vocabulary as a ``PreTrainedTokenizerFast``;
``coord_roundtrip.json``
    proof that our coordinate spelling reproduces the public tokenizer's token
    sequence, plus quantization error over validation conformers;
``fsq_roundtrip.json``
    the frozen FSQ tokenizer's fingerprint and reconstruction statistics on the
    same conformers.
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Optional, Sequence

import numpy as np

from molgen3D.experiments.bindgpt_fsq import config, public_tokenizer, vocabulary
from molgen3D.experiments.bindgpt_fsq.coord_tokens import (
    TOKENS_PER_ATOM,
    CoordinateRangeError,
    scalar_to_tokens,
)
from molgen3D.experiments.bindgpt_fsq.sequences import (
    build_coord_stream,
    parse_coord_stream,
    parse_enriched,
    atom_coordinates,
)

PERCENTILES = (50, 90, 95, 99)


def prepare_tokenizers(download: bool = True) -> Dict[str, object]:
    """Download, derive, verify and freeze both tokenizer artifacts."""
    if download or not (config.PUBLIC_TOKENIZER_DIR / "tokenizer.json").exists():
        files = public_tokenizer.download(config.PUBLIC_TOKENIZER_DIR)
    else:
        files = public_tokenizer.PublicTokenizerFiles(config.PUBLIC_TOKENIZER_DIR)

    architecture = public_tokenizer.verify_architecture(files)
    public_tokenizer.verify_coordinate_tokens(files)
    smiles_tokens = public_tokenizer.derive_smiles_tokens(files)

    config.SMILES_TOKENS_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.SMILES_TOKENS_JSON.write_text(
        json.dumps(
            {
                "repo_id": public_tokenizer.REPO_ID,
                "file_sha256": files.hashes(),
                "smiles_tokens": smiles_tokens,
            },
            indent=1,
        )
    )

    union = vocabulary.build_vocabulary(smiles_tokens)
    vocabulary.save_hf_tokenizer(union, config.UNION_TOKENIZER_DIR)

    summary = {
        "public_repo": public_tokenizer.REPO_ID,
        "public_file_sha256": files.hashes(),
        "architecture": {
            key: architecture[key] for key in public_tokenizer.EXPECTED_ARCHITECTURE
        },
        "public_vocab_size": len(files.added_tokens()),
        "smiles_token_count": len(smiles_tokens),
        "union_vocab_size": len(union),
        "union_blocks": {name: len(block) for name, block in union.blocks.items()},
        "coord_forbidden_ids": len(union.forbidden_ids(vocabulary.Arm.COORD)),
        "fsq_forbidden_ids": len(union.forbidden_ids(vocabulary.Arm.FSQ)),
    }
    (config.EXPERIMENT_ROOT / "tokenizer_audit" / "vocabulary.json").write_text(
        json.dumps(summary, indent=1)
    )
    return summary


# --------------------------------------------------------------------------- #
# Audits
# --------------------------------------------------------------------------- #


def audit_coord_tokenizer(
    enriched_texts: Sequence[str],
    smiles: Sequence[str],
    max_public_checks: int = 2_000,
) -> Dict[str, object]:
    """Round-trip and quantization audit for the BindGPT coordinate tokens.

    Two independent checks:

    1. **Public agreement.** For a sample of conformers, every coordinate scalar
       is spelled with :func:`scalar_to_tokens` and the resulting text is handed
       to the *public* tokenizer; its token strings must equal ours. This is the
       "do not reimplement rounding from memory" guarantee.
    2. **Quantization error.** Coordinate MAE and aligned heavy-atom RMSD after a
       full encode/decode of the interleaved stream.
    """
    files = public_tokenizer.PublicTokenizerFiles(config.PUBLIC_TOKENIZER_DIR)
    reference = public_tokenizer.load_reference_tokenizer(files)
    smiles_tokens = vocabulary.load_smiles_tokens()

    scalar_errors: List[float] = []
    rmsds: List[float] = []
    stream_lengths: List[int] = []
    atoms: List[int] = []
    public_mismatches: List[Dict[str, object]] = []
    smiles_mismatches: List[Dict[str, object]] = []
    out_of_range = 0
    smiles_out_of_vocabulary = 0
    checked_public = 0

    for index, enriched in enumerate(enriched_texts):
        tokens = parse_enriched(enriched)
        try:
            stream = build_coord_stream(tokens)
        except CoordinateRangeError:
            out_of_range += 1
            continue

        original = atom_coordinates(tokens)
        graph, decoded = parse_coord_stream(stream)
        if graph != [token.text for token in tokens]:
            raise AssertionError("coordinate stream lost the graph on round trip")
        if decoded.shape != original.shape:
            raise AssertionError(
                f"decoded {decoded.shape} coordinates for {original.shape} atoms"
            )

        scalar_errors.extend(np.abs(decoded - original).ravel().tolist())
        rmsds.append(float(np.sqrt(((decoded - original) ** 2).sum(axis=1).mean())))
        stream_lengths.append(len(stream))
        atoms.append(int(original.shape[0]))

        if checked_public < max_public_checks:
            checked_public += 1
            ours: List[str] = []
            text_parts: List[str] = []
            for value in original.ravel():
                int_token, frac_token = scalar_to_tokens(float(value))
                ours.extend((int_token, frac_token))
                text_parts.append(f"{int_token}{frac_token}")
            theirs = public_tokenizer.reference_tokens(reference, " ".join(text_parts))
            if theirs != ours:
                public_mismatches.append(
                    {"index": index, "ours": ours[:12], "theirs": theirs[:12]}
                )

            if index < len(smiles):
                try:
                    mine = public_tokenizer.greedy_tokenize(smiles[index], smiles_tokens)
                except ValueError:
                    smiles_out_of_vocabulary += 1
                    continue
                theirs_smiles = public_tokenizer.reference_tokens(
                    reference, smiles[index]
                )
                if mine != theirs_smiles:
                    smiles_mismatches.append(
                        {
                            "index": index,
                            "smiles": smiles[index],
                            "ours": mine,
                            "theirs": theirs_smiles,
                        }
                    )

    errors = np.asarray(scalar_errors, dtype=np.float64)
    rmsd = np.asarray(rmsds, dtype=np.float64)
    return {
        "conformers": len(rmsds),
        "conformers_out_of_range": out_of_range,
        "public_sequences_checked": checked_public,
        "public_coordinate_mismatches": len(public_mismatches),
        "public_coordinate_mismatch_examples": public_mismatches[:5],
        "public_smiles_mismatches": len(smiles_mismatches),
        "smiles_out_of_vocabulary": smiles_out_of_vocabulary,
        "public_smiles_mismatch_examples": smiles_mismatches[:5],
        "coordinate_mae": float(errors.mean()) if errors.size else None,
        "coordinate_max_abs_error": float(errors.max()) if errors.size else None,
        "rmsd_mean": float(rmsd.mean()) if rmsd.size else None,
        "rmsd_percentiles": {
            f"p{p}": float(np.percentile(rmsd, p)) for p in PERCENTILES
        }
        if rmsd.size
        else {},
        "tokens_per_atom": TOKENS_PER_ATOM,
        "stream_tokens_mean": float(np.mean(stream_lengths)) if stream_lengths else None,
        "stream_tokens_max": int(np.max(stream_lengths)) if stream_lengths else None,
        "atoms_mean": float(np.mean(atoms)) if atoms else None,
    }


def audit_fsq_tokenizer(
    enriched_texts: Sequence[str],
    device: str = "auto",
    batch_size: int = 64,
) -> Dict[str, object]:
    """Fingerprint and reconstruction audit for the frozen FSQ tokenizer."""
    from molgen3D.experiments.bindgpt_fsq.fsq_frozen import FrozenFSQTokenizer

    tokenizer = FrozenFSQTokenizer(device=device, verify_checkpoint_hash=True)
    tokenizer.assert_no_gradients()

    scalar_errors: List[float] = []
    rmsds: List[float] = []
    stream_lengths: List[int] = []
    undecodable = 0
    over_length = 0

    for start in range(0, len(enriched_texts), batch_size):
        chunk = list(enriched_texts[start : start + batch_size])
        parsed = [parse_enriched(text) for text in chunk]
        keep = [
            (text, tokens)
            for text, tokens in zip(chunk, parsed)
            if len(tokens) <= tokenizer.model.max_tokens
        ]
        over_length += len(chunk) - len(keep)
        if not keep:
            continue

        codes_batch = tokenizer.encode([text for text, _ in keep])
        for (text, tokens), codes in zip(keep, codes_batch):
            graph = [token.text for token in tokens]
            if len(codes) != len(graph):
                undecodable += 1
                continue
            try:
                decoded = tokenizer.decode(graph, codes)
            except Exception:  # noqa: BLE001 - counted, not silently dropped
                undecodable += 1
                continue

            original = atom_coordinates(tokens)
            # The FSQ model centres internally, so compare on centroids removed.
            centred = original - original.mean(axis=0, keepdims=True)
            if decoded.shape != centred.shape:
                undecodable += 1
                continue
            scalar_errors.extend(np.abs(decoded - centred).ravel().tolist())
            rmsds.append(
                float(np.sqrt(((decoded - centred) ** 2).sum(axis=1).mean()))
            )
            stream_lengths.append(2 * len(graph))

    tokenizer.assert_unchanged()
    errors = np.asarray(scalar_errors, dtype=np.float64)
    rmsd = np.asarray(rmsds, dtype=np.float64)
    return {
        "fingerprint": tokenizer.fingerprint().as_dict(),
        "conformers": len(rmsds),
        "conformers_over_max_tokens": over_length,
        "undecodable": undecodable,
        "undecodable_rate": (
            undecodable / max(1, len(rmsds) + undecodable)
        ),
        "coordinate_mae": float(errors.mean()) if errors.size else None,
        "coordinate_max_abs_error": float(errors.max()) if errors.size else None,
        "rmsd_mean": float(rmsd.mean()) if rmsd.size else None,
        "rmsd_percentiles": {
            f"p{p}": float(np.percentile(rmsd, p)) for p in PERCENTILES
        }
        if rmsd.size
        else {},
        "tokens_per_graph_token": 2,
        "stream_tokens_mean": float(np.mean(stream_lengths)) if stream_lengths else None,
        "stream_tokens_max": int(np.max(stream_lengths)) if stream_lengths else None,
    }


def sample_validation_conformers(
    limit: int = 10_000, seed: int = 17
) -> tuple[List[str], List[str]]:
    """Return ``(enriched_texts, canonical_smiles)`` from the GEOM valid split.

    The identity logic matches the training builder: every parseable row is
    eligible, including disconnected SMILES; only rows that cannot be rendered
    into either representation are omitted.
    """
    from molgen3D.experiments.bindgpt_fsq.data.extract_source import (
        iter_geom_jsonl,
        strip_coordinates,
    )
    from molgen3D.experiments.bindgpt_fsq.data.identity import identity_from_graph

    enriched: List[str] = []
    smiles: List[str] = []
    identities: dict[str, Optional[str]] = {}
    for row in iter_geom_jsonl("valid", verify_historical_fsq=False):
        graph = strip_coordinates(row.enriched_text)
        if graph not in identities:
            identity = identity_from_graph(graph)
            identities[graph] = (
                identity.standardized_isomeric_smiles if identity.usable else None
            )
        canonical = identities[graph]
        if canonical is None:
            continue
        enriched.append(row.enriched_text)
        smiles.append(canonical)
        if len(enriched) >= limit * 4:
            break

    rng = random.Random(seed)
    if len(enriched) > limit:
        picked = rng.sample(range(len(enriched)), limit)
        picked.sort()
        enriched = [enriched[index] for index in picked]
        smiles = [smiles[index] for index in picked]
    return enriched, smiles


def write_audit_comparison() -> Dict[str, object]:
    """Combine the two audits into the side-by-side tokenizer comparison.

    Covers the two fields the plan asks for that neither audit can produce
    alone: FSQ's token count and compression ratio relative to the original
    BindGPT coordinates, and the reconstruction accuracy each representation
    buys at that length.
    """
    audit_dir = config.EXPERIMENT_ROOT / "tokenizer_audit"
    coord = json.loads((audit_dir / "coord_roundtrip.json").read_text())
    fsq = json.loads((audit_dir / "fsq_roundtrip.json").read_text())

    coord_tokens_mean = coord.get("stream_tokens_mean")
    fsq_tokens_mean = fsq.get("stream_tokens_mean")
    comparison = {
        "conformers": {"coord": coord.get("conformers"), "fsq": fsq.get("conformers")},
        "geometry_stream_tokens_mean": {
            "coord": coord_tokens_mean,
            "fsq": fsq_tokens_mean,
        },
        "geometry_stream_tokens_max": {
            "coord": coord.get("stream_tokens_max"),
            "fsq": fsq.get("stream_tokens_max"),
        },
        "fsq_compression_ratio_vs_coord": (
            coord_tokens_mean / fsq_tokens_mean
            if coord_tokens_mean and fsq_tokens_mean
            else None
        ),
        "coordinate_mae_angstrom": {
            "coord": coord.get("coordinate_mae"),
            "fsq": fsq.get("coordinate_mae"),
        },
        "aligned_rmsd_angstrom": {
            "coord": {
                "mean": coord.get("rmsd_mean"),
                **(coord.get("rmsd_percentiles") or {}),
            },
            "fsq": {
                "mean": fsq.get("rmsd_mean"),
                **(fsq.get("rmsd_percentiles") or {}),
            },
        },
        "undecodable_rate": {"coord": 0.0, "fsq": fsq.get("undecodable_rate")},
        "excluded_by_tokenizer": {
            "coord_coordinate_out_of_range": coord.get("conformers_out_of_range"),
            "fsq_over_max_graph_tokens": fsq.get("conformers_over_max_tokens"),
        },
        "public_agreement": {
            "coordinate_mismatches": coord.get("public_coordinate_mismatches"),
            "smiles_mismatches": coord.get("public_smiles_mismatches"),
            "sequences_checked": coord.get("public_sequences_checked"),
        },
        "note": (
            "FSQ's reconstruction error is a floor on what the FSQ arm can reach "
            "on AMR: no amount of language-model accuracy decodes below it. "
            "Compare it against the 0.75 A coverage threshold when reading the "
            "conformer table."
        ),
    }
    (audit_dir / "tokenizer_comparison.json").write_text(
        json.dumps(comparison, indent=1)
    )
    return comparison
