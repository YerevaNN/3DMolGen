import os
from transformers import AutoTokenizer

# Paths
tokenizer_dir = "/auto/home/vover/3DMolGen/src/molgen3D/training/tokenizers/Qwen3_tokenizer_original"
new_tokenizer_binned = "/auto/home/vover/3DMolGen/src/molgen3D/training/tokenizers/Qwen3_tokenizer_binned"

# Ensure output directory exists
os.makedirs(new_tokenizer_binned, exist_ok=True)

print(f"Loading original tokenizer from {tokenizer_dir}...")
# trust_remote_code=True might be needed for some Qwen tokenizers if they use custom code
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

# Generate new tokens "000" to "249"
new_tokens = [f"{i:03d}" for i in range(250)]

special_tokens = ["[SMILES]", "[CONFORMER]", "[/SMILES]", "[/CONFORMER]"]
special_tokens_dict = {
    "additional_special_tokens": special_tokens+new_tokens
}

num_added = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Successfully added {num_added} tokens (special and numerical).")

tokenizer.save_pretrained(new_tokenizer_binned)
print(f"Successfully saved tokenizer to {new_tokenizer_binned}.")
