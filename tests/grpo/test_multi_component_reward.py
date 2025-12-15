from types import SimpleNamespace

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from molgen3D.training.grpo import multi_component_reward as reward_module
from molgen3D.training.grpo.config import AdvancedRewardConfig, AdvancedRewardWeights
from molgen3D.training.grpo.multi_component_reward import MultiComponentRewardCalculator
from molgen3D.training.grpo.stats import RunStatistics


def _build_config(num_generations: int = 3, enable_diversity: bool = True, match_partial: bool = False):
    weights = AdvancedRewardWeights(
        precision=0.4,
        coverage=0.3,
        match=0.2,
        validity=0.1,
        diversity=0.05,
    )
    advanced = AdvancedRewardConfig(
        delta=0.75,
        precision_scale=0.45,
        coverage_scale=0.6,
        diversity_scale=1.2,
        normalization_epsilon=1e-6,
        enable_diversity=enable_diversity,
        enable_posebusters=True,
        posebusters_config="mol",
        match_partial_credit=match_partial,
        max_reference_conformers=30,
        weights=weights,
    )
    grpo = SimpleNamespace(
        num_generations=num_generations,
        max_ground_truths=30,
        advanced_reward=advanced,
    )
    return SimpleNamespace(grpo=grpo)


def _clone_conformer(mol: Chem.Mol, conf_id: int) -> Chem.Mol:
    clone = Chem.Mol(mol)
    conformer = Chem.Conformer(mol.GetConformer(conf_id))
    clone.RemoveAllConformers()
    clone.AddConformer(conformer, assignId=True)
    return clone


def _embed_conformers(smiles: str, num_confs: int, seed: int = 7) -> list[Chem.Mol]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    for conf_id in range(num_confs):
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id)
    return [_clone_conformer(mol, conf_id) for conf_id in range(num_confs)]


def _translate_mol(mol: Chem.Mol, offset: tuple[float, float, float]) -> Chem.Mol:
    translated = Chem.Mol(mol)
    conf = translated.GetConformer()
    for atom_idx in range(conf.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        conf.SetAtomPosition(
            atom_idx,
            Point3D(pos.x + offset[0], pos.y + offset[1], pos.z + offset[2]),
        )
    return translated


@pytest.fixture
def stats(tmp_path):
    return RunStatistics(output_dir=str(tmp_path))


@pytest.fixture(autouse=True)
def disable_wandb(monkeypatch):
    monkeypatch.setattr(reward_module.wandb, "run", None)


def test_multi_component_pipeline(monkeypatch, stats):
    config = _build_config(num_generations=3)
    calculator = MultiComponentRewardCalculator(
        config=config,
        stats=stats,
        tokenizer=None,
        posebusters_checker=lambda mol: True,
    )

    # Generate reference and candidate conformers
    references = _embed_conformers("CCO", num_confs=3, seed=21)
    generated = [
        references[0],  # perfect match
        _translate_mol(references[1], (0.1, 0.0, 0.0)),  # close match
        _translate_mol(references[2], (2.0, 2.0, 2.0)),  # outside delta threshold
    ]

    # Patch decoding and ground-truth loading to use synthetic molecules
    gen_iter = iter(generated)
    monkeypatch.setattr(reward_module, "_decode_generated_conformer", lambda _: next(gen_iter))
    monkeypatch.setattr(reward_module, "load_ground_truths", lambda *args, **kwargs: references)

    prompts = ["[SMILES]CCO[/SMILES]"] * 3
    completions = ["[CONFORMER]placeholder[/CONFORMER]"] * 3

    rewards = calculator(prompts, completions)

    assert len(rewards) == 3
    assert stats.processed_prompts == 3
    assert stats.distinct_prompts == 1
    assert stats.coverage_claims == 2  # first two conformers cover unique references
    assert pytest.approx(stats.precision_rewards[0], rel=1e-3) == 1.0
    assert stats.precision_rewards[2] < stats.precision_rewards[0]
    assert rewards[0] > rewards[2]
    assert stats.posebusters_successes == 3
    assert stats.failed_ground_truth == 0


def test_diversity_scores(monkeypatch, stats):
    config = _build_config(num_generations=2, enable_diversity=True)
    calculator = MultiComponentRewardCalculator(
        config=config,
        stats=stats,
        tokenizer=None,
        posebusters_checker=lambda mol: True,
    )

    mols = _embed_conformers("CCC", num_confs=2, seed=14)
    mols[1] = _translate_mol(mols[1], (1.5, 0.0, 0.0))

    scores = calculator._diversity_scores(mols)
    assert scores[0] == 0.0
    assert scores[1] > scores[0]
    assert 0.0 <= scores[1] <= 1.0


def test_normalization_zero_std(stats):
    config = _build_config(num_generations=2, enable_diversity=False)
    calculator = MultiComponentRewardCalculator(
        config=config,
        stats=stats,
        tokenizer=None,
        posebusters_checker=lambda mol: True,
    )

    precision = [0.5, 0.5]
    coverage = [0.5, 0.5]
    match = [0.5, 0.5]
    validity = [1.0, 1.0]
    diversity = [0.0, 0.0]

    scores = calculator._combine_components(precision, coverage, match, validity, diversity)
    # All normalized components collapse to zero; validity remains
    expected = config.grpo.advanced_reward.weights.validity
    assert all(pytest.approx(score, rel=1e-6) == expected for score in scores)


def test_match_partial_credit(stats):
    config = _build_config(num_generations=2, match_partial=True)
    calculator = MultiComponentRewardCalculator(
        config=config,
        stats=stats,
        tokenizer=None,
        posebusters_checker=lambda mol: True,
    )

    target = "CCO"
    mol_exact = Chem.MolFromSmiles(target)
    mol_partial = Chem.MolFromSmiles("CC")

    scores = calculator._match_scores(
        generated=[mol_exact, mol_partial],
        canonical_smiles=target,
    )
    assert scores[0] == 1.0
    assert 0.0 < scores[1] < 1.0
