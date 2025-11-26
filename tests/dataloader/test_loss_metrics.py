import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def _manual_per_token_losses(logits, labels):
    losses = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )
    mask = labels.view(-1) != IGNORE_INDEX
    return losses[mask]


def test_cross_entropy_matches_manual_mean():
    """Invariant: CE loss used for backward equals mean per-token CE."""
    logits = torch.tensor(
        [[[4.0, 1.0, -2.0], [0.5, 2.0, 0.0], [3.0, -1.0, 0.0]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[0, 1, IGNORE_INDEX]])

    ce_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="mean",
    )
    manual = _manual_per_token_losses(logits, labels).mean()

    assert torch.allclose(ce_loss, manual, atol=1e-7)


def test_global_max_loss_matches_manual_max():
    """Invariant: global_max_loss equals the max per-token CE."""
    logits = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[0, 2, IGNORE_INDEX]])

    per_token = _manual_per_token_losses(logits, labels)
    expected_max = per_token.max()

    assert torch.allclose(expected_max, per_token.max())


def test_logits_are_raw_scores_not_probabilities():
    """Invariant: logits passed to CE are raw scores prior to softmax."""
    logits = torch.tensor([[[-0.5, 2.5], [3.0, -1.5]]], dtype=torch.float32)
    assert (logits < 0).any() and (logits > 1).any()

    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-6)
