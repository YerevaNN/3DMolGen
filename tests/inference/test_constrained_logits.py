import torch

from molgen3D.evaluation.constrained_logits import (
    ConformerConstraintLogitsProcessor,
    build_sequence_template,
)


class DummyTokenizer:
    """
    Minimal tokenizer stub with a fixed vocab.

    Only the `encode(text, add_special_tokens=False)` method is used by the code under test.
    """

    def __init__(self):
        # Reserve 0 for pad to mimic real tokenizers
        self.vocab = {
            "[CONFORMER]": 1,
            "[/CONFORMER]": 5,
            "[C]": 2,
            "C": 2,  # allow bare C as well
            "=": 3,
            "1": 4,
            ">": 6,
            "<": 7,
            "[SMILES]": 8,
            "X": 9,
            "8": 10,
        }

    def encode(self, text, add_special_tokens=False):
        if text not in self.vocab:
            raise ValueError(f"Unknown token in dummy tokenizer: {text}")
        return [self.vocab[text]]

    def get_vocab(self):
        return self.vocab


def test_template_builds_atoms_and_coords():
    tok = DummyTokenizer()
    tmpl = build_sequence_template("CC", tok)

    labels = [seg.label for seg in tmpl.segments]
    kinds = [seg.kind for seg in tmpl.segments]

    assert labels[0] == "[CONFORMER]"
    # Atom, coord, atom, coord
    assert labels[1] == "[C]"
    assert labels[2] == "<coords>"
    assert labels[3] == "[C]"
    assert labels[4] == "<coords>"
    assert labels[-1] == "[/CONFORMER]"

    assert kinds == ["fixed", "fixed", "coord", "fixed", "coord", "fixed"]


def test_logits_processor_masks_fixed_tokens_and_skips_coords():
    tok = DummyTokenizer()
    tmpl = build_sequence_template("CC", tok)

    prompt_len = 2  # pretend prompt already has two tokens
    proc = ConformerConstraintLogitsProcessor([tmpl], [prompt_len], tokenizer=tok)

    vocab_size = max(tok.vocab.values()) + 1
    lt_id = tok.encode("<", add_special_tokens=False)[0]
    smi_id = tok.encode("[SMILES]", add_special_tokens=False)[0]
    free_id = tok.encode("X", add_special_tokens=False)[0]
    digit_id = tok.encode("8", add_special_tokens=False)[0]

    def call_with_ids(ids):
        input_ids = torch.tensor([ids], dtype=torch.long)
        scores = torch.zeros(1, vocab_size)
        out = proc(input_ids, scores.clone())
        return out

    # Step 0: before any generation, expect [CONFORMER] (id=1)
    masked = call_with_ids([99, 99])
    assert torch.isinf(masked).sum() == vocab_size - 1
    assert masked[0, 1] == 0

    # Step 1: emit [CONFORMER]
    masked = call_with_ids([99, 99, 1])
    # Now expect first atom [C] (id=2)
    assert masked[0, 2] == 0
    assert torch.isinf(masked).sum() == vocab_size - 1

    # Step 2: emit atom and enter coord block -> force '<'
    masked = call_with_ids([99, 99, 1, 2])
    assert torch.isinf(masked).sum() == vocab_size - 1
    assert masked[0, lt_id] == 0

    # Step 3: emit '<' then coords are limited to numeric-ish tokens and '>'
    masked = call_with_ids([99, 99, 1, 2, lt_id])
    # allowed set contains digits and '>'
    assert masked[0, digit_id] == 0

    # Step 4: still inside coord until '>' arrives
    masked = call_with_ids([99, 99, 1, 2, lt_id, digit_id])
    # forbidden token masked, allowed digit remains
    assert masked[0, smi_id] == -torch.inf
    assert masked[0, free_id] == -torch.inf  # non-numeric token masked
    assert masked[0, digit_id] == 0

    # Step 5: close coord with '>' (id=6); expect next atom masked
    masked = call_with_ids([99, 99, 1, 2, lt_id, digit_id, 6])
    assert masked[0, 2] == 0  # second [C]
    assert torch.isinf(masked).sum() == vocab_size - 1
