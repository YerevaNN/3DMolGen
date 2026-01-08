"""
Tests for the Qwen Allowlist Logit Processor (v4.3).

These tests verify basic correctness and behavior of the logit processor.
Real end-to-end testing is done via the smoke runner (scripts/logit_processor/run_logit_processor_smoke.py).
"""
from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer

from molgen3D.config.paths import get_tokenizer_path
from molgen3D.evaluation.qwen_logit_processor import (
    COORD_PLACEHOLDER,
    COORD_VALID_CHARS,
    PrecomputedTemplate,
    QwenAllowlistLogitsProcessor,
    build_precomputed_template,
    build_reference_skeleton,
    _get_allowed_coord_token_ids,
    _get_comma_dash_token_ids,
    _get_comma_only_token_ids,
    _is_valid_coord_token,
)


@pytest.fixture(scope="module")
def qwen_tokenizer():
    """Load the Qwen tokenizer once for all tests in this module."""
    tokenizer_path = get_tokenizer_path("qwen3_0.6b_custom")
    return AutoTokenizer.from_pretrained(tokenizer_path)


class TestBuildReferenceSkeleton:
    """Tests for build_reference_skeleton function."""

    def test_simple_smiles(self, qwen_tokenizer):
        """Test skeleton building for simple SMILES."""
        skeleton = build_reference_skeleton("CC")

        # Should have two atoms with coordinate placeholders
        assert "[C]" in skeleton or "[CH3]" in skeleton or "[CH4]" in skeleton
        assert f"<{COORD_PLACEHOLDER}>" in skeleton
        # Should have two coordinate blocks (one per atom)
        assert skeleton.count(f"<{COORD_PLACEHOLDER}>") == 2

    def test_double_bond_smiles(self, qwen_tokenizer):
        """Test skeleton building for SMILES with double bond."""
        skeleton = build_reference_skeleton("C=C")

        # Should preserve the double bond
        assert "=" in skeleton
        # Should have two coordinate blocks
        assert skeleton.count(f"<{COORD_PLACEHOLDER}>") == 2

    def test_ring_smiles(self, qwen_tokenizer):
        """Test skeleton building for cyclic SMILES."""
        skeleton = build_reference_skeleton("C1CCCCC1")  # cyclohexane

        # Should have ring notation
        assert "1" in skeleton
        # Should have 6 coordinate blocks (one per carbon)
        assert skeleton.count(f"<{COORD_PLACEHOLDER}>") == 6


class TestBuildPrecomputedTemplate:
    """Tests for build_precomputed_template function."""

    def test_template_structure(self, qwen_tokenizer):
        """Test that template has correct structure."""
        template = build_precomputed_template("CC", qwen_tokenizer)

        assert isinstance(template, PrecomputedTemplate)
        assert isinstance(template.ref_ids, torch.Tensor)
        assert isinstance(template.is_free, torch.Tensor)
        assert isinstance(template.seq_len, int)
        assert template.seq_len == len(template.ref_ids)
        assert template.seq_len == len(template.is_free)

        # Should have smart blocking masks
        assert template.block_comma_dash is not None
        assert template.is_first_free is not None

    def test_free_positions_exist(self, qwen_tokenizer):
        """Test that template has FREE positions for coordinates."""
        template = build_precomputed_template("CC", qwen_tokenizer)

        # Should have some FREE positions (for coordinate content)
        assert template.is_free.sum() > 0

        # Should have some COPY positions (for structural tokens)
        assert (~template.is_free).sum() > 0

    def test_conformer_tags_included(self, qwen_tokenizer):
        """Test that [CONFORMER] tags are included by default."""
        template = build_precomputed_template("CC", qwen_tokenizer, include_tags=True)

        # Decode the template to check for tags
        decoded = qwen_tokenizer.decode(template.ref_ids.tolist())
        assert "[CONFORMER]" in decoded
        assert "[/CONFORMER]" in decoded

    def test_conformer_tags_excluded(self, qwen_tokenizer):
        """Test that [CONFORMER] tags can be excluded."""
        template = build_precomputed_template("CC", qwen_tokenizer, include_tags=False)

        # Decode the template to check for tags
        decoded = qwen_tokenizer.decode(template.ref_ids.tolist())
        assert "[CONFORMER]" not in decoded
        assert "[/CONFORMER]" not in decoded


class TestCoordTokenValidation:
    """Tests for coordinate token validation."""

    def test_valid_single_chars(self):
        """Test that single valid characters are accepted."""
        for char in "0123456789.,-":
            assert _is_valid_coord_token(char), f"'{char}' should be valid"

    def test_invalid_chars(self):
        """Test that invalid characters are rejected."""
        for char in "abcXYZ<>[]":
            assert not _is_valid_coord_token(char), f"'{char}' should be invalid"

    def test_valid_coord_strings(self):
        """Test valid coordinate-like strings."""
        valid_strings = ["0", "123", "1.234", "-0.5", "1,2,3", "0.000,0.000,0.000"]
        for s in valid_strings:
            assert _is_valid_coord_token(s), f"'{s}' should be valid"

    def test_invalid_patterns(self):
        """Test invalid character patterns."""
        # Double punctuation
        assert not _is_valid_coord_token("..")
        assert not _is_valid_coord_token(",,")
        assert not _is_valid_coord_token("--")

        # Invalid pairs
        assert not _is_valid_coord_token(".,")
        assert not _is_valid_coord_token(",.")
        assert not _is_valid_coord_token(".-")
        assert not _is_valid_coord_token("-,")

    def test_empty_string(self):
        """Test that empty string is rejected."""
        assert not _is_valid_coord_token("")


class TestAllowedTokenIds:
    """Tests for allowed token ID caching."""

    def test_allowed_ids_not_empty(self, qwen_tokenizer):
        """Test that we find allowed coordinate tokens."""
        allowed_ids = _get_allowed_coord_token_ids(qwen_tokenizer)

        assert len(allowed_ids) > 0
        # Should include digit tokens (Qwen uses single-char tokens for digits)
        # At minimum we expect digits 0-9, period, comma, minus

    def test_comma_dash_ids_found(self, qwen_tokenizer):
        """Test that comma/dash tokens are found for blocking."""
        comma_dash_ids = _get_comma_dash_token_ids(qwen_tokenizer)

        assert len(comma_dash_ids) > 0

    def test_comma_only_ids_found(self, qwen_tokenizer):
        """Test that comma-only tokens are found for first-position blocking."""
        comma_only_ids = _get_comma_only_token_ids(qwen_tokenizer)

        assert len(comma_only_ids) > 0


class TestQwenAllowlistLogitsProcessor:
    """Tests for the main logit processor."""

    def test_processor_initialization(self, qwen_tokenizer):
        """Test processor can be initialized."""
        template = build_precomputed_template("CC", qwen_tokenizer)
        prompt_length = 10  # arbitrary prompt length

        processor = QwenAllowlistLogitsProcessor(
            templates=[template],
            prompt_lengths=[prompt_length],
            tokenizer=qwen_tokenizer,
            eos_token_id=qwen_tokenizer.eos_token_id,
        )

        assert processor.templates == [template]
        assert processor.prompt_lengths == [prompt_length]

    def test_processor_forces_copy_tokens(self, qwen_tokenizer):
        """Test that processor forces COPY tokens at COPY positions."""
        template = build_precomputed_template("CC", qwen_tokenizer)
        prompt_length = 5

        processor = QwenAllowlistLogitsProcessor(
            templates=[template],
            prompt_lengths=[prompt_length],
            tokenizer=qwen_tokenizer,
            eos_token_id=qwen_tokenizer.eos_token_id,
        )

        vocab_size = len(qwen_tokenizer)

        # Find a COPY position (where is_free is False), skip position 0 to test non-trivial case
        copy_pos = None
        for i, is_free in enumerate(template.is_free.tolist()):
            if not is_free and i > 0:
                copy_pos = i
                break

        assert copy_pos is not None, "Should have at least one COPY position after position 0"

        # First call to initialize _prev_lens
        input_ids_init = torch.zeros(1, prompt_length, dtype=torch.long)
        scores_init = torch.zeros(1, vocab_size)
        _ = processor(input_ids_init, scores_init)

        # Now test at the target COPY position
        cur_len = prompt_length + copy_pos
        input_ids = torch.zeros(1, cur_len, dtype=torch.long)
        scores = torch.zeros(1, vocab_size)

        masked_scores = processor(input_ids, scores.clone())

        # Only the reference token should be allowed (score = 0)
        expected_token = template.ref_ids[copy_pos].item()
        assert masked_scores[0, expected_token] == 0.0

        # All other tokens should be masked (-inf)
        assert torch.isinf(masked_scores).sum() == vocab_size - 1

    def test_processor_allows_coord_tokens_at_free_positions(self, qwen_tokenizer):
        """Test that processor allows coordinate tokens at FREE positions."""
        template = build_precomputed_template("CC", qwen_tokenizer)
        prompt_length = 5

        processor = QwenAllowlistLogitsProcessor(
            templates=[template],
            prompt_lengths=[prompt_length],
            tokenizer=qwen_tokenizer,
            eos_token_id=qwen_tokenizer.eos_token_id,
        )

        vocab_size = len(qwen_tokenizer)

        # Find a "middle" FREE position that is NOT first and NOT last-before-close
        # (to avoid smart blocking logic)
        free_pos = None
        for i, is_free in enumerate(template.is_free.tolist()):
            if is_free:
                is_first = template.is_first_free[i].item() if template.is_first_free is not None else False
                is_last = template.block_comma_dash[i].item() if template.block_comma_dash is not None else False
                if not is_first and not is_last:
                    free_pos = i
                    break

        if free_pos is None:
            pytest.skip("No suitable FREE position found for this test")

        # Simulate generation: first call sets _prev_lens, then advance to target position
        # The processor tracks position as: pos = cur_len - _prev_lens[b]
        # First call with prompt_length tokens sets _prev_lens[0] = prompt_length
        input_ids_init = torch.zeros(1, prompt_length, dtype=torch.long)
        scores_init = torch.zeros(1, vocab_size)
        _ = processor(input_ids_init, scores_init)  # Initialize _prev_lens

        # Now call at the target position
        cur_len = prompt_length + free_pos
        input_ids = torch.zeros(1, cur_len, dtype=torch.long)
        scores = torch.zeros(1, vocab_size)
        masked_scores = processor(input_ids, scores.clone())

        # Allowed coordinate tokens should NOT be masked (except comma-only if in comma_only_mask)
        # Check single-character digit tokens which should always be allowed at middle FREE positions
        digit_tokens = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # 0-9 in Qwen
        for tok_id in digit_tokens:
            if tok_id < vocab_size:
                assert not torch.isinf(masked_scores[0, tok_id]), f"Digit token {tok_id} should be allowed"

    def test_processor_forces_eos_past_sequence(self, qwen_tokenizer):
        """Test that processor forces EOS when past expected sequence length."""
        template = build_precomputed_template("CC", qwen_tokenizer)
        prompt_length = 5

        processor = QwenAllowlistLogitsProcessor(
            templates=[template],
            prompt_lengths=[prompt_length],
            tokenizer=qwen_tokenizer,
            eos_token_id=qwen_tokenizer.eos_token_id,
        )

        vocab_size = len(qwen_tokenizer)

        # First call to initialize _prev_lens
        input_ids_init = torch.zeros(1, prompt_length, dtype=torch.long)
        scores_init = torch.zeros(1, vocab_size)
        _ = processor(input_ids_init, scores_init)

        # Simulate being past the sequence length
        cur_len = prompt_length + template.seq_len + 10
        input_ids = torch.zeros(1, cur_len, dtype=torch.long)
        scores = torch.zeros(1, vocab_size)

        masked_scores = processor(input_ids, scores.clone())

        # Only EOS should be allowed
        assert masked_scores[0, qwen_tokenizer.eos_token_id] == 0.0
        assert torch.isinf(masked_scores).sum() == vocab_size - 1


class TestSmartBlocking:
    """Tests for smart blocking features (first/last FREE position blocking)."""

    def test_first_free_blocks_comma(self, qwen_tokenizer):
        """Test that first FREE position blocks comma tokens."""
        template = build_precomputed_template("CC", qwen_tokenizer)
        prompt_length = 5

        processor = QwenAllowlistLogitsProcessor(
            templates=[template],
            prompt_lengths=[prompt_length],
            tokenizer=qwen_tokenizer,
            eos_token_id=qwen_tokenizer.eos_token_id,
        )

        vocab_size = len(qwen_tokenizer)

        # Find a first FREE position
        first_free_pos = None
        for i, is_free in enumerate(template.is_free.tolist()):
            if is_free and template.is_first_free is not None and template.is_first_free[i].item():
                first_free_pos = i
                break

        if first_free_pos is None:
            pytest.skip("No first FREE position found")

        # First call to initialize _prev_lens
        input_ids_init = torch.zeros(1, prompt_length, dtype=torch.long)
        scores_init = torch.zeros(1, vocab_size)
        _ = processor(input_ids_init, scores_init)

        cur_len = prompt_length + first_free_pos
        input_ids = torch.zeros(1, cur_len, dtype=torch.long)
        scores = torch.zeros(1, vocab_size)

        masked_scores = processor(input_ids, scores.clone())

        # Comma-only tokens should be blocked at first FREE position
        comma_only_ids = _get_comma_only_token_ids(qwen_tokenizer)
        for tok_id in comma_only_ids:
            if tok_id < vocab_size:
                assert torch.isinf(masked_scores[0, tok_id]), f"Comma token {tok_id} should be blocked at first FREE"

    def test_last_free_blocks_comma_and_dash(self, qwen_tokenizer):
        """Test that last FREE position (before '>') blocks comma and dash tokens."""
        template = build_precomputed_template("CC", qwen_tokenizer)
        prompt_length = 5

        processor = QwenAllowlistLogitsProcessor(
            templates=[template],
            prompt_lengths=[prompt_length],
            tokenizer=qwen_tokenizer,
            eos_token_id=qwen_tokenizer.eos_token_id,
        )

        vocab_size = len(qwen_tokenizer)

        # Find a position where block_comma_dash is True
        last_free_pos = None
        for i, is_free in enumerate(template.is_free.tolist()):
            if is_free and template.block_comma_dash is not None and template.block_comma_dash[i].item():
                last_free_pos = i
                break

        if last_free_pos is None:
            pytest.skip("No last FREE position found")

        # First call to initialize _prev_lens
        input_ids_init = torch.zeros(1, prompt_length, dtype=torch.long)
        scores_init = torch.zeros(1, vocab_size)
        _ = processor(input_ids_init, scores_init)

        cur_len = prompt_length + last_free_pos
        input_ids = torch.zeros(1, cur_len, dtype=torch.long)
        scores = torch.zeros(1, vocab_size)

        masked_scores = processor(input_ids, scores.clone())

        # Comma/dash tokens should be blocked at last FREE position
        comma_dash_ids = _get_comma_dash_token_ids(qwen_tokenizer)
        for tok_id in comma_dash_ids:
            if tok_id < vocab_size:
                assert torch.isinf(masked_scores[0, tok_id]), f"Comma/dash token {tok_id} should be blocked at last FREE"


class TestBatchProcessing:
    """Tests for batch processing."""

    def test_batch_different_smiles(self, qwen_tokenizer):
        """Test processor handles batch with different SMILES."""
        smiles_list = ["CC", "C=O", "CCO"]
        templates = [build_precomputed_template(smi, qwen_tokenizer) for smi in smiles_list]
        prompt_lengths = [5, 6, 7]  # Different prompt lengths

        processor = QwenAllowlistLogitsProcessor(
            templates=templates,
            prompt_lengths=prompt_lengths,
            tokenizer=qwen_tokenizer,
            eos_token_id=qwen_tokenizer.eos_token_id,
        )

        vocab_size = len(qwen_tokenizer)
        batch_size = 3

        # Use max prompt length for input_ids
        cur_len = max(prompt_lengths)
        input_ids = torch.zeros(batch_size, cur_len, dtype=torch.long)
        scores = torch.zeros(batch_size, vocab_size)

        # Should not raise
        masked_scores = processor(input_ids, scores.clone())

        assert masked_scores.shape == (batch_size, vocab_size)
