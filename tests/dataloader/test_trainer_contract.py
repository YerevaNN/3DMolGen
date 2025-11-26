import inspect

import torchtitan.train as titan_train
from molgen3D.training.pretraining.torchtitan_model import qwen3_custom


def test_forward_backward_step_uses_loss_fn_directly():
    """Invariant: Titan feeds raw logits into CE and backprops the same scalar."""
    src = inspect.getsource(titan_train.Trainer.forward_backward_step)
    assert "loss = self.loss_fn(pred, labels)" in src
    assert "loss.backward()" in src
    call_line = "pred = model_parts[0](inputs"
    assert call_line in src
    assert "labels=labels" not in src.split("loss = self.loss_fn")[0]


def test_train_step_accumulates_and_logs_same_loss():
    """Invariant: loss logged as global_avg_loss originates from the same scalar."""
    src = inspect.getsource(titan_train.Trainer.train_step)
    assert "accumulated_losses.append(loss.detach())" in src
    assert "loss = torch.sum(torch.stack(accumulated_losses))" in src
    assert "self.metrics_processor.log(" in src
    assert "global_avg_loss" in src and "global_max_loss" in src


def test_metrics_use_expected_reduction_ops():
    """Invariant: Titan computes global avg/max via dp/cp reductions."""
    src = inspect.getsource(titan_train.Trainer.train_step)
    assert "dist_utils.dist_mean" in src
    assert "dist_utils.dist_max" in src
    assert "dist_utils.dist_sum" in src


def test_qwen_wrapper_does_not_apply_softmax():
    """Invariant: model wrapper leaves logits unnormalized."""
    src = inspect.getsource(qwen3_custom._parallelize_with_resize)
    assert ".softmax" not in src
    assert "F.softmax" not in src
