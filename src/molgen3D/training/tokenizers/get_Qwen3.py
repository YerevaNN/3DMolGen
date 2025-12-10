#!/usr/bin/env python
import os
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# ----- CONFIG -----

# Hugging Face model identifier
HF_MODEL_NAME = "Qwen/Qwen3-0.6B-Base"

# Where to save the untouched original tokenizer snapshot
ORIG_TOK_DIR = os.path.join("./src/molgen3D/training/tokenizers", "Qwen3_tokenizer_original")

# Where to save the tokenizer with the 4 extra tokens
CUSTOM_TOK_DIR = os.path.join("./src/molgen3D/training/tokenizers", "Qwen3_tokenizer_custom")

# Where to save the model
MODEL_DIR = os.path.join("/nfs/h100/raid/chem/checkpoints/yerevann/qwen3_06b/", "Qwen3-0.6B-Base")

# Your 4 new tokens as *normal* vocab items
NEW_TOKENS = ["[SMILES]", "[CONFORMER]", "[/SMILES]", "[/CONFORMER]"]

# ----- SCRIPT -----

def main():
    load_model_and_tokenizer()
    sanity_checks()

def load_model_and_tokenizer():
    print(f"Loading model and tokenizer from Hugging Face: {HF_MODEL_NAME}")
    
    # Load tokenizer and model from Hugging Face
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    print(f"Tokenizer is_fast: {tokenizer.is_fast}")
    
    print("Downloading model...")
    # Try AutoModelForCausalLM first (has LM head for weight tying check)
    # Fallback to AutoModel if needed
    try:
        model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME)
        print("Loaded as AutoModelForCausalLM (includes LM head)")
    except Exception as e:
        print(f"Could not load as AutoModelForCausalLM: {e}")
        print("Falling back to AutoModel...")
        model = AutoModel.from_pretrained(HF_MODEL_NAME)
        print("Loaded as AutoModel")
    
    # Print original vocab and embedding shapes
    orig_vocab_size = len(tokenizer)
    print(f"\n=== Original Sizes ===")
    print(f"Original vocab size: {orig_vocab_size}")
    
    # Get embedding layer shape
    if hasattr(model, 'embed_tokens'):
        embedding = model.embed_tokens
    elif hasattr(model, 'wte'):  # Some models use 'wte' for word token embeddings
        embedding = model.wte
    elif hasattr(model, 'embeddings') and hasattr(model.embeddings, 'word_embeddings'):
        embedding = model.embeddings.word_embeddings
    else:
        # Try to find embedding layer
        embedding = None
        for name, module in model.named_modules():
            if 'embed' in name.lower() and hasattr(module, 'weight'):
                embedding = module
                print(f"Found embedding layer: {name}")
                break
    
    if embedding is not None:
        orig_embed_shape = embedding.weight.shape
        print(f"Original embedding shape: {orig_embed_shape}")
        print(f"  - Vocab size: {orig_embed_shape[0]}")
        print(f"  - Embedding dim: {orig_embed_shape[1]}")
    else:
        print("Warning: Could not find embedding layer in model")
        orig_embed_shape = None
    
    # Check if LM head is tied to embeddings
    print(f"\n=== LM Head Weight Tying Check ===")
    
    # Method 1: Check model configuration (most reliable)
    config_tied = None
    if hasattr(model, 'config') and hasattr(model.config, 'tie_word_embeddings'):
        config_tied = model.config.tie_word_embeddings
        print(f"Model config tie_word_embeddings: {config_tied}")
    
    # Method 2: Use standardized methods to get embeddings and LM head
    try:
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
        
        if input_embeddings is None:
            print("Warning: get_input_embeddings() returned None")
            input_embeddings = embedding
        if output_embeddings is None:
            print("Warning: Model does not have an output LM head")
            output_embeddings = None
    except AttributeError:
        print("Model does not support get_input_embeddings() or get_output_embeddings()")
        input_embeddings = embedding
        output_embeddings = None
        # Fallback to manual detection
        if hasattr(model, 'lm_head'):
            output_embeddings = model.lm_head
            print(f"Found LM head via fallback: model.lm_head")
        elif hasattr(model, 'output_projection'):
            output_embeddings = model.output_projection
            print(f"Found LM head via fallback: model.output_projection")
        else:
            # Try to find LM head layer
            for name, module in model.named_modules():
                if 'lm_head' in name.lower() or ('output' in name.lower() and 'embed' in name.lower()):
                    if hasattr(module, 'weight') and module.weight is not None:
                        output_embeddings = module
                        print(f"Found LM head via fallback: {name}")
                        break
    
    # Perform weight tying check if we have both embeddings and LM head
    if input_embeddings is not None and output_embeddings is not None:
        embed_weight = input_embeddings.weight
        lm_head_weight = output_embeddings.weight
        
        print(f"Input embedding weight shape: {embed_weight.shape}")
        print(f"Output LM head weight shape: {lm_head_weight.shape}")
        
        # Multiple checks for weight tying
        is_tied_object = embed_weight is lm_head_weight
        is_tied_memory = False
        is_tied_values = False
        torch_available = False
        
        try:
            import torch
            torch_available = True
            # Check if they share the same memory address (most reliable for detecting tied weights)
            is_tied_memory = embed_weight.data_ptr() == lm_head_weight.data_ptr()
            
            # Check if they have the same values (less reliable, could be copies)
            if embed_weight.shape == lm_head_weight.shape:
                is_tied_values = torch.equal(embed_weight, lm_head_weight)
        except (AttributeError, ImportError) as e:
            print(f"Could not perform memory/value comparison: {e}")
        
        # Final determination: weights are tied if any method detects it
        is_tied = is_tied_object or is_tied_memory
        
        print(f"\nWeight tying detection results:")
        print(f"  - Same object (is): {is_tied_object}")
        if torch_available:
            print(f"  - Same memory address (data_ptr): {is_tied_memory}")
            print(f"  - Same values (torch.equal): {is_tied_values}")
        print(f"  - Config setting: {config_tied}")
        
        # Final verdict
        if is_tied:
            print(f"\n✓ Weight tying DETECTED - embedding and LM head share weights")
            if config_tied is False:
                print("  Note: Config says tie_word_embeddings=False, but weights are actually tied!")
        else:
            print(f"\n✗ No weight tying - embedding and LM head have separate weights")
            if config_tied is True:
                print("  Note: Config says tie_word_embeddings=True, but weights are not actually tied!")
            elif config_tied is None:
                print("  Note: Could not determine config setting")
        
    elif input_embeddings is None:
        print("Cannot check weight tying - input embedding layer not found")
    elif output_embeddings is None:
        print("Cannot check weight tying - output LM head not found")
        print("  (This is normal for encoder-only models like BERT)")

    # 1) Save a snapshot of the original tokenizer
    os.makedirs(ORIG_TOK_DIR, exist_ok=True)
    print(f"\nSaving original tokenizer to: {ORIG_TOK_DIR}")
    tokenizer.save_pretrained(ORIG_TOK_DIR)

    # 2) Add new tokens as *normal* tokens (not special tokens)
    print(f"\nAdding new tokens: {NEW_TOKENS}")
    num_added = tokenizer.add_tokens(NEW_TOKENS, special_tokens=False)
    print(f"Number of tokens actually added: {num_added}")

    new_vocab_size = len(tokenizer)
    print(f"New vocab size: {new_vocab_size}")

    # 3) Save the modified tokenizer
    os.makedirs(CUSTOM_TOK_DIR, exist_ok=True)
    print(f"\nSaving custom tokenizer with extra tokens to: {CUSTOM_TOK_DIR}")
    tokenizer.save_pretrained(CUSTOM_TOK_DIR)
    
    # Store metadata for sanity_checks to read
    metadata_file = os.path.join(CUSTOM_TOK_DIR, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump({
            "orig_vocab_size": orig_vocab_size,
            "num_added": num_added,
            "new_vocab_size": new_vocab_size,
            "orig_embed_shape": list(orig_embed_shape) if orig_embed_shape is not None else None
        }, f, indent=2)

    # 4) Save the model (unchanged, no resizing)
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Saving model to: {MODEL_DIR}")
    model.save_pretrained(MODEL_DIR)

def sanity_checks():
    # ----- SANITY CHECKS -----
    print("\n=== Sanity Checks ===")
    
    # Load metadata if available (from previous run of load_model_and_tokenizer)
    metadata_file = os.path.join(CUSTOM_TOK_DIR, "metadata.json")
    orig_vocab_size = None
    num_added = None
    new_vocab_size = None
    orig_embed_shape = None
    
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            orig_vocab_size = metadata.get("orig_vocab_size")
            num_added = metadata.get("num_added")
            new_vocab_size = metadata.get("new_vocab_size")
            orig_embed_shape = tuple(metadata["orig_embed_shape"]) if metadata.get("orig_embed_shape") else None
            print(f"Loaded metadata: orig_vocab_size={orig_vocab_size}, num_added={num_added}, new_vocab_size={new_vocab_size}")
    else:
        print("Warning: metadata.json not found. Some checks will be skipped.")

    # 1) Reload both original and custom tokenizers and verify
    print("\n=== Reloading and Verifying ===")
    print("Reloading original tokenizer...")
    if not os.path.exists(ORIG_TOK_DIR):
        print(f"Warning: Original tokenizer directory not found: {ORIG_TOK_DIR}")
        print("Skipping original tokenizer checks.")
    else:
        orig_tok = AutoTokenizer.from_pretrained(ORIG_TOK_DIR)
        print(f"Original reloaded vocab size: {len(orig_tok)} (is_fast={orig_tok.is_fast})")
        if orig_vocab_size is not None:
            assert len(orig_tok) == orig_vocab_size, f"Reloaded original vocab size mismatch: expected {orig_vocab_size}, got {len(orig_tok)}"

    print("Reloading custom tokenizer...")
    if not os.path.exists(CUSTOM_TOK_DIR):
        print(f"Error: Custom tokenizer directory not found: {CUSTOM_TOK_DIR}")
        print("Please run load_model_and_tokenizer() first.")
        return
    
    custom_tok = AutoTokenizer.from_pretrained(CUSTOM_TOK_DIR)
    print(f"Custom reloaded vocab size: {len(custom_tok)} (is_fast={custom_tok.is_fast})")
    
    # 2) Verify vocab size increased by num_added
    if orig_vocab_size is not None and num_added is not None:
        expected_new_size = orig_vocab_size + num_added
        assert len(custom_tok) == expected_new_size, (
            f"Vocab size mismatch: expected {expected_new_size}, got {len(custom_tok)}"
        )
        if new_vocab_size is not None:
            assert len(custom_tok) == new_vocab_size, (
                f"Vocab size mismatch: expected {new_vocab_size}, got {len(custom_tok)}"
            )

    # 3) New tokens have valid (non-UNK) IDs
    unk_id = custom_tok.unk_token_id
    for tok in NEW_TOKENS:
        tid = custom_tok.convert_tokens_to_ids(tok)
        print(f"Token {tok!r} -> id {tid}")
        assert tid is not None and tid != unk_id, (
            f"Token {tok!r} did not get a valid id (got {tid}, unk={unk_id})"
        )

    # 4) Reload model and verify embedding size (should be unchanged)
    print("\nReloading model...")
    if not os.path.exists(MODEL_DIR):
        print(f"Warning: Model directory not found: {MODEL_DIR}")
        print("Skipping model embedding checks.")
    else:
        try:
            reloaded_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
        except Exception:
            reloaded_model = AutoModel.from_pretrained(MODEL_DIR)
        
        if hasattr(reloaded_model, 'embed_tokens'):
            reloaded_embedding = reloaded_model.embed_tokens
        elif hasattr(reloaded_model, 'wte'):
            reloaded_embedding = reloaded_model.wte
        elif hasattr(reloaded_model, 'embeddings') and hasattr(reloaded_model.embeddings, 'word_embeddings'):
            reloaded_embedding = reloaded_model.embeddings.word_embeddings
        else:
            reloaded_embedding = None
        
        if reloaded_embedding is not None:
            reloaded_embed_shape = reloaded_embedding.weight.shape
            print(f"Reloaded model embedding shape: {reloaded_embed_shape}")
            print(f"  - Vocab size: {reloaded_embed_shape[0]}")
            print(f"  - Embedding dim: {reloaded_embed_shape[1]}")
            # Verify embedding shape matches original (model was not resized)
            if orig_embed_shape is not None:
                assert reloaded_embed_shape == orig_embed_shape, (
                    f"Embedding shape changed: {orig_embed_shape} -> {reloaded_embed_shape}"
                )
                print(f"  ✓ Embedding shape matches original")

    print("\nAll sanity checks passed ✅")

if __name__ == "__main__":
    main()