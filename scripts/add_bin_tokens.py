import os
from transformers import AutoTokenizer
from molgen3D.config.paths import get_base_path, get_tokenizer_path

# Paths
tokenizer_dir = get_tokenizer_path("qwen3_0.6b_origin")
new_tokenizer_binned = get_base_path("tokenizers_root") / "Qwen3_tokenizer_binned_v2"

# Ensure output directory exists
os.makedirs(new_tokenizer_binned, exist_ok=True)

print(f"Loading original tokenizer from {tokenizer_dir}...")
# trust_remote_code=True might be needed for some Qwen tokenizers if they use custom code
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

# Generate new tokens "000" to "249"
new_tokens = [f"{i:03d}" for i in range(250)]

special_tokens = ["[SMILES]", "[CONFORMER]", "[/SMILES]", "[/CONFORMER]"]
special_tokens_dict = {
    "additional_special_tokens": special_tokens
}


num_added = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Successfully added {num_added} tokens (special).")

tokenizer.add_tokens(new_tokens)
print(f"Successfully added {len(new_tokens)} tokens (numerical).")

print(f"Total number of tokens: {len(tokenizer)}")

tokenizer.save_pretrained(new_tokenizer_binned)
print(f"Successfully saved tokenizer to {new_tokenizer_binned}.")
