from transformers import AutoTokenizer

SPECIAL_TOKENS = ["[SMILES]", "[CONFORMER]", "[/SMILES]", "[/CONFORMER]"]

# add special tokens to the tokenizer 

tokenizer = AutoTokenizer.from_pretrained("/auto/home/vover/3DMolGen/src/molgen3D/training/tokenizers/Qwen3_tokenizer_original", local_files_only=True)

tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

tokenizer.save_pretrained("/auto/home/vover/3DMolGen/src/molgen3D/training/tokenizers/Qwen3_tokenizer_custom_v2")