"""Frozen configuration for the BindGPT-Coord vs BindGPT-FSQ experiment.

Every scientific setting lives here or in the YAML this module reads and writes.
Shell scripts carry only partition, walltime and launcher flags.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from molgen3D.config.paths import REPO_ROOT

#: Root of the experiment's artifact tree.
EXPERIMENT_ROOT = Path(
    os.environ.get("BINDGPT_FSQ_ROOT", REPO_ROOT / "bindgpt_geom_fsq")
)

PUBLIC_TOKENIZER_DIR = EXPERIMENT_ROOT / "tokenizer_audit" / "bindgpt_public"
SMILES_TOKENS_JSON = EXPERIMENT_ROOT / "tokenizer_audit" / "bindgpt_smiles_tokens.json"
UNION_TOKENIZER_DIR = EXPERIMENT_ROOT / "tokenizer_audit" / "union_tokenizer"

MANIFEST_DIR = EXPERIMENT_ROOT / "manifests"
CONFIG_DIR = EXPERIMENT_ROOT / "configs"
CHECKPOINT_DIR = EXPERIMENT_ROOT / "checkpoints"
GENERATION_DIR = EXPERIMENT_ROOT / "generations"
METRIC_DIR = EXPERIMENT_ROOT / "metrics"
LOG_DIR = EXPERIMENT_ROOT / "logs"
ENVIRONMENT_DIR = EXPERIMENT_ROOT / "environment"

# --------------------------------------------------------------------------- #
# Immutable data provenance
# --------------------------------------------------------------------------- #

#: The pinned frozen FSQ coordinate tokenizer. ``last-v2.ckpt`` is the file the
#: project's own preprocessing and inference paths use; it is epoch 0 end,
#: ``global_step=185757``, ``d_model=1024``, ``levels=[2]*12`` (4,096 codes),
#: 202,011,535 parameters.
FSQ_CHECKPOINT = Path(
    "/mnt/weka/fgeikyan/fsq/new_checkpoints/"
    "full_d1024_v4096_b128_lr0_0001_20260327_131937/last-v2.ckpt"
)

#: Exact decimal-coordinate corpus from which the historical FSQ SFT data was
#: produced.  The six train shards align row-for-row with
#: ``GEOM_HISTORICAL_FSQ_ROOT / 'train'`` and contain the full historical
#: 5,745,689-example training population.
GEOM_DECIMAL_ROOT = Path("/data/vtarasov/geom_cartesian_v3/processed_strings")
GEOM_HISTORICAL_FSQ_ROOT = Path(
    "/mnt/weka/fgeikyan/fsq/geom_fsq_4096/processed_strings"
)

EXPECTED_GEOM_TRAIN_ROWS = 5_745_689
EXPECTED_GEOM_VALID_ROWS = 716_818
STANDARD_SPLIT_TRAIN_ROWS = EXPECTED_GEOM_TRAIN_ROWS

#: The frozen 1,000-molecule GEOM-DRUGS benchmark test set (GeoMol /
#: Torsional-Diffusion split) with all reference conformers.
BENCHMARK_TEST_PICKLE = Path(
    "/mnt/weka/fgeikyan/tokenizer/distinct_smi_filtered_13.pickle"
)
EXPECTED_TEST_MOLECULES = 1_000


# --------------------------------------------------------------------------- #
# Training and decoding
# --------------------------------------------------------------------------- #


@dataclass
class TrainingSettings:
    """The BindGPT GEOM supervised-fine-tuning recipe, applied from scratch."""

    epochs: int = 1
    optimizer: str = "AdamW"
    learning_rate: float = 5e-4
    warmup_steps: int = 100
    schedule: str = "cosine"
    weight_decay: float = 1e-2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    dtype: str = "bfloat16"
    #: Enabled (non-padding) tokens per optimizer step, BindGPT's stated target.
    target_tokens_per_step: int = 1_600_000
    #: Examples per optimizer step. Fixed and identical across arms so the
    #: comparison is matched by molecule-conformer examples, never by tokens.
    global_example_batch_size: int = 2_048
    micro_batch_size: int = 32
    #: Length-bucketing granularity, in tokens.
    bucket_width: int = 64
    max_sequence_length: int = 2_048
    seed: int = 17
    checkpoint_fractions: List[float] = field(
        default_factory=lambda: [0.25, 0.50, 0.75, 1.00]
    )
    smiles_randomization: bool = False
    random_rotations: bool = False


@dataclass
class DecodingSettings:
    """Primary decoding settings, identical for both arms."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    do_sample: bool = True
    num_beams: int = 1
    #: BindGPT's public example caps total length at 400 tokens. The coord arm
    #: needs more than that under the paired format, so the cap is expressed as
    #: a multiple of the prompt-implied requirement and reported per arm.
    max_new_tokens: int = 1_536
    sampling_seed: int = 2_023


@dataclass
class ConformerEvalSettings:
    """Evaluation 2: the regular GEOM-DRUGS conformer protocol."""

    #: Headline threshold only. No extra grid, no curve, no AUC.
    threshold: float = 0.75
    generations_per_reference: int = 2
    max_attempt_multiplier: int = 4
    use_alignmol: bool = False
    failure_budget: float = 0.01


@dataclass
class UnconditionalEvalSettings:
    """Evaluation 1: BindGPT/MolDiff-style unconditional generation."""

    target_valid_connected: int = 1_000
    max_attempts: int = 20_000
    batch_size: int = 256
    #: Frozen before generation: the 1,000-molecule benchmark test set is the
    #: primary reference distribution, validation the secondary table.
    primary_reference: str = "test"


@dataclass
class ExperimentConfig:
    """One arm's complete, immutable run description."""

    arm: str
    run_id: str
    training: TrainingSettings = field(default_factory=TrainingSettings)
    decoding: DecodingSettings = field(default_factory=DecodingSettings)
    conformer_eval: ConformerEvalSettings = field(default_factory=ConformerEvalSettings)
    unconditional_eval: UnconditionalEvalSettings = field(
        default_factory=UnconditionalEvalSettings
    )
    fsq_checkpoint: str = str(FSQ_CHECKPOINT)
    #: "suffix" is published BindGPT's layout -- SMILES, separator, geometry
    #: only. "interleaved" repeats the graph with geometry woven in and exists
    #: only to reproduce the first run.
    layout: str = "suffix"
    provenance: Dict[str, str] = field(default_factory=dict)

    @property
    def checkpoint_dir(self) -> Path:
        return CHECKPOINT_DIR / self.arm / f"seed_{self.training.seed}"

    def to_yaml(self) -> str:
        return yaml.safe_dump(asdict(self), sort_keys=False)

    def write(self, path: Optional[Path] = None) -> Path:
        path = path or CONFIG_DIR / f"{self.run_id}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml())
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        payload = yaml.safe_load(Path(path).read_text())
        return cls(
            arm=payload["arm"],
            run_id=payload["run_id"],
            training=TrainingSettings(**payload["training"]),
            decoding=DecodingSettings(**payload["decoding"]),
            conformer_eval=ConformerEvalSettings(**payload["conformer_eval"]),
            unconditional_eval=UnconditionalEvalSettings(
                **payload["unconditional_eval"]
            ),
            fsq_checkpoint=payload["fsq_checkpoint"],
            layout=payload.get("layout", "interleaved"),
            provenance=payload.get("provenance", {}),
        )


def default_configs(layout: str = "suffix") -> Dict[str, ExperimentConfig]:
    """Return the two headline run configurations."""
    tag = "" if layout == "interleaved" else "_suffix"
    return {
        "coord": ExperimentConfig(
            arm="coord", run_id=f"bindgpt_cartesian_geom_scratch{tag}", layout=layout
        ),
        "fsq": ExperimentConfig(
            arm="fsq", run_id=f"bindgpt_coordtoken_geom_scratch{tag}", layout=layout
        ),
    }
