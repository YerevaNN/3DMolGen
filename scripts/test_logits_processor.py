"""Test script for ConformerControlLogitsProcessor and ConformerCountStoppingCriteria."""

import sys
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.molgen3D.training.grpo.logits_constraints import (
    ConformerControlLogitsProcessor,
    ConformerCountStoppingCriteria,
)
from src.molgen3D.config.paths import get_ckpt, get_tokenizer_path


def test_logits_processor(
    model_path: str,
    tokenizer_path: str,
    target_k: int = 8,
    test_smiles: str = "CCO",
    device: str = "cuda",
):
    """Test the logits processor with a simple example."""
    
    logger.info(f"Loading model from {model_path}")
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        padding_side="left",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    ).eval()
    
    logger.info(f"Model loaded: {model.dtype}, {model.device}")
    
    # Setup tags and token IDs
    conf_tags = ["[CONFORMER]", "[/CONFORMER]"]
    mol_tags = ["[SMILES]", "[/SMILES]"]
    
    conformer_start_ids = tokenizer.encode(conf_tags[0], add_special_tokens=False)
    conformer_end_ids = tokenizer.encode(conf_tags[1], add_special_tokens=False)
    smiles_start_ids = tokenizer.encode(mol_tags[0], add_special_tokens=False)
    smiles_end_ids = tokenizer.encode(mol_tags[1], add_special_tokens=False)
    
    logger.info(f"Token IDs:")
    logger.info(f"  [CONFORMER]: {conformer_start_ids}")
    logger.info(f"  [/CONFORMER]: {conformer_end_ids}")
    logger.info(f"  [SMILES]: {smiles_start_ids}")
    logger.info(f"  [/SMILES]: {smiles_end_ids}")
    logger.info(f"  EOS: {tokenizer.eos_token_id}")
    logger.info(f"  PAD: {tokenizer.pad_token_id}")
    
    # Setup banned tokens (SMILES start/end, EOS, PAD)
    banned_ids = set(smiles_start_ids)
    banned_ids.update(smiles_end_ids)  # Ban closing SMILES tag too!
    if tokenizer.eos_token_id is not None:
        banned_ids.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        banned_ids.add(tokenizer.pad_token_id)
    
    # Don't ban conformer tokens
    for tok in conformer_start_ids:
        banned_ids.discard(tok)
    for tok in conformer_end_ids:
        banned_ids.discard(tok)
    
    logger.info(f"Banned token IDs: {banned_ids}")
    
    # Create processor and stopping criteria
    processor = ConformerControlLogitsProcessor(
        conformer_start_ids=conformer_start_ids,
        conformer_end_ids=conformer_end_ids,
        banned_token_ids=banned_ids,
        target_k=target_k,
        force_hard=True,
    )
    
    stopper = ConformerCountStoppingCriteria(
        conformer_end_ids=conformer_end_ids,
        target_k=target_k,
    )
    
    # Test 1: Basic generation
    logger.info(f"\n{'='*80}")
    logger.info(f"TEST 1: Basic generation with target_k={target_k}")
    logger.info(f"{'='*80}")
    
    prompt = f"[SMILES]{test_smiles}[/SMILES]"
    logger.info(f"Prompt: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    logger.info(f"Input IDs shape: {inputs.input_ids.shape}")
    
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=2500,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            logits_processor=[processor],
            stopping_criteria=[stopper],
        )
    
    generated = tokenizer.decode(out[0], skip_special_tokens=False)
    logger.info(f"\nGenerated output length: {len(out[0])} tokens")
    logger.info(f"\nGenerated output:\n{generated}")
    
    # Count conformers
    conformer_count = generated.count("[/CONFORMER]")
    logger.info(f"\nConformer count: {conformer_count} (target: {target_k})")
    
    # Check if SMILES token was generated (should not be)
    smiles_in_output = "[SMILES]" in generated[len(prompt):]
    logger.info(f"SMILES token in generated part: {smiles_in_output} (should be False)")
    
    # Test 2: Batch generation
    logger.info(f"\n{'='*80}")
    logger.info(f"TEST 2: Batch generation with 2 different SMILES")
    logger.info(f"{'='*80}")
    
    batch_prompts = [
        f"[SMILES]CCO[/SMILES]",
        f"[SMILES]CC[/SMILES]",
    ]
    
    logger.info(f"Prompts: {batch_prompts}")
    
    inputs_batch = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        pad_to_multiple_of=8
    ).to(model.device)
    
    # Create fresh processor for batch test
    processor_batch = ConformerControlLogitsProcessor(
        conformer_start_ids=conformer_start_ids,
        conformer_end_ids=conformer_end_ids,
        banned_token_ids=banned_ids,
        target_k=target_k,
        force_hard=True,
    )
    
    stopper_batch = ConformerCountStoppingCriteria(
        conformer_end_ids=conformer_end_ids,
        target_k=target_k,
    )
    
    with torch.inference_mode():
        out_batch = model.generate(
            **inputs_batch,
            max_new_tokens=2500,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            logits_processor=[processor_batch],
            stopping_criteria=[stopper_batch],
        )
    
    for idx, seq in enumerate(out_batch):
        generated_seq = tokenizer.decode(seq, skip_special_tokens=False)
        conformer_count = generated_seq.count("[/CONFORMER]")
        logger.info(f"\nSequence {idx}: {conformer_count} conformers (target: {target_k})")
        logger.info(f"Length: {len(seq)} tokens")
        # Show first 500 chars
        logger.info(f"Output (truncated): {generated_seq[:500]}...")
    
    # Test 3: Different target_k values
    logger.info(f"\n{'='*80}")
    logger.info(f"TEST 3: Testing different target_k values")
    logger.info(f"{'='*80}")
    
    for test_k in [2, 4, 6]:
        logger.info(f"\nTesting target_k={test_k}")
        
        processor_k = ConformerControlLogitsProcessor(
            conformer_start_ids=conformer_start_ids,
            conformer_end_ids=conformer_end_ids,
            banned_token_ids=banned_ids,
            target_k=test_k,
            force_hard=True,
        )
        
        stopper_k = ConformerCountStoppingCriteria(
            conformer_end_ids=conformer_end_ids,
            target_k=test_k,
        )
        
        inputs_k = tokenizer(f"[SMILES]{test_smiles}[/SMILES]", return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            out_k = model.generate(
                **inputs_k,
                max_new_tokens=2500,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                logits_processor=[processor_k],
                stopping_criteria=[stopper_k],
            )
        
        generated_k = tokenizer.decode(out_k[0], skip_special_tokens=False)
        conformer_count_k = generated_k.count("[/CONFORMER]")
        logger.info(f"  Generated {conformer_count_k} conformers (target: {test_k})")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"ALL TESTS COMPLETE")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test conformer logits processor")
    parser.add_argument("--model_key", type=str, default="m600_qwen_pre_4seq_binned")
    parser.add_argument("--checkpoint", type=str, default="4e")
    parser.add_argument("--tokenizer_key", type=str, default="qwen3_0.6b_custom")
    parser.add_argument("--target_k", type=int, default=8)
    parser.add_argument("--test_smiles", type=str, default="CCO")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    model_path = get_ckpt(args.model_key, args.checkpoint)
    tokenizer_path = get_tokenizer_path(args.tokenizer_key)
    
    test_logits_processor(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        target_k=args.target_k,
        test_smiles=args.test_smiles,
        device=args.device,
    )
