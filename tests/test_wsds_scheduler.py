import math

import pytest
import torch

from molgen3D.training.pretraining.helpers.wsds_scheduler import (
    WSDSSchedule,
    WSDSSchedulerConfig,
)


def _build_schedule(cfg: WSDSSchedulerConfig) -> WSDSSchedule:
    optimizer = torch.optim.SGD(
        [torch.zeros(1, requires_grad=True)], lr=cfg.base_lr or 1.0
    )
    return WSDSSchedule(
        optimizer=optimizer,
        checkpoints=cfg.validate(),
        warmup_steps=cfg.warmup_steps,
        lr_max=cfg.lr_max,
        lr_min=cfg.lr_min,
        decay_frac=cfg.decay_frac,
        decay_steps=cfg.decay_steps,
        base_lr=cfg.base_lr,
    )


def test_wsds_config_validation_errors() -> None:
    bad_cases = [
        (
            dict(
                warmup_steps=-1,
                checkpoints=[10],
                lr_max=1.0,
                lr_min=0.0,
                decay_frac=0.1,
            ),
            "warmup_steps must be non-negative",
        ),
        (
            dict(
                warmup_steps=1,
                checkpoints=[10],
                lr_max=0.0,
                lr_min=0.0,
                decay_frac=0.1,
            ),
            "lr_max must be positive",
        ),
        (
            dict(
                warmup_steps=1,
                checkpoints=[10],
                lr_max=0.5,
                lr_min=1.0,
                decay_frac=0.1,
            ),
            "lr_min must be between 0 and lr_max",
        ),
        (
            dict(
                warmup_steps=1,
                checkpoints=[10],
                lr_max=1.0,
                lr_min=0.0,
                decay_steps=0,
                decay_frac=None,
            ),
            "decay_steps must be positive",
        ),
        (
            dict(
                warmup_steps=1,
                checkpoints=[10],
                lr_max=1.0,
                lr_min=0.0,
                decay_frac=1.5,
            ),
            "decay_frac must be within",
        ),
        (
            dict(
                warmup_steps=1,
                checkpoints=[10],
                lr_max=1.0,
                lr_min=0.0,
                decay_frac=None,
                decay_steps=None,
            ),
            "Set decay_steps or decay_frac",
        ),
    ]

    for kwargs, message in bad_cases:
        cfg = WSDSSchedulerConfig(enable=True, **kwargs)
        with pytest.raises(ValueError, match=message):
            cfg.validate()


def test_wsds_schedule_warmup_stable_decay_reset_fraction() -> None:
    cfg = WSDSSchedulerConfig(
        enable=True,
        warmup_steps=100,
        checkpoints=[1250, 2500, 5000, 10000],
        lr_max=1.0,
        lr_min=0.0,
        decay_frac=0.1,
        base_lr=1.0,
    )
    sched = _build_schedule(cfg)

    assert math.isclose(sched.get_lr(0), 0.0, rel_tol=1e-9)
    assert math.isclose(sched.get_lr(50), 0.5, rel_tol=1e-9)
    assert math.isclose(sched.get_lr(99), 0.99, rel_tol=1e-9)
    assert math.isclose(sched.get_lr(100), cfg.lr_max, rel_tol=1e-9)

    expected_start = {
        1250: 1125,
        2500: 2250,
        5000: 4500,
        10000: 9000,
    }
    for target, start in expected_start.items():
        assert math.isclose(sched.get_lr(start - 1), cfg.lr_max, rel_tol=1e-9)
        assert math.isclose(sched.get_lr(start), cfg.lr_max, rel_tol=1e-9)
        mid = start + (target - start) // 2
        expected_mid = cfg.lr_max - (cfg.lr_max - cfg.lr_min) * (
            (mid - start) / (target - start)
        )
        assert math.isclose(sched.get_lr(mid), expected_mid, rel_tol=1e-9)
        assert math.isclose(sched.get_lr(target), cfg.lr_min, rel_tol=1e-9)
        if target == cfg.checkpoints[-1]:
            assert math.isclose(sched.get_lr(target + 1), cfg.lr_min, rel_tol=1e-9)
        else:
            assert math.isclose(sched.get_lr(target + 1), cfg.lr_max, rel_tol=1e-9)

    assert math.isclose(sched.get_lr(11000), cfg.lr_min, rel_tol=1e-9)


def test_wsds_schedule_fixed_decay_length_priority() -> None:
    cfg = WSDSSchedulerConfig(
        enable=True,
        warmup_steps=100,
        checkpoints=[1250, 2500, 5000, 10000],
        lr_max=2.0,
        lr_min=0.2,
        decay_steps=100,
        decay_frac=0.5,  # should be ignored
        base_lr=0.5,
    )
    sched = _build_schedule(cfg)

    expected_start = {
        1250: 1150,
        2500: 2400,
        5000: 4900,
        10000: 9900,
    }

    for target, start in expected_start.items():
        assert math.isclose(sched.get_lr(start - 1), cfg.lr_max, rel_tol=1e-9)
        assert math.isclose(sched.get_lr(start), cfg.lr_max, rel_tol=1e-9)
        assert math.isclose(sched.get_lr(target), cfg.lr_min, rel_tol=1e-9)
        if target == cfg.checkpoints[-1]:
            assert math.isclose(sched.get_lr(target + 1), cfg.lr_min, rel_tol=1e-9)
        else:
            assert math.isclose(sched.get_lr(target + 1), cfg.lr_max, rel_tol=1e-9)


def test_wsds_schedule_callable_matches_factor() -> None:
    cfg = WSDSSchedulerConfig(
        enable=True,
        warmup_steps=10,
        checkpoints=[20],
        lr_max=1.0,
        lr_min=0.1,
        decay_steps=5,
        base_lr=0.01,
    )
    sched = _build_schedule(cfg)

    for step in [0, 5, 10, 15, 20]:
        lr_value = sched.get_lr(step)
        factor = sched(step)
        assert math.isclose(
            factor * cfg.base_lr, lr_value, rel_tol=1e-9
        ), f"step={step} mismatch"

