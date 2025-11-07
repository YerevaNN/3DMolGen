from transformers import GenerationConfig

greedy_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=2000,
    eos_token_id=128329,
)

beam_search_config = GenerationConfig(
    num_beams=8,  
    num_beam_groups=4,
    max_new_tokens=2000,
    eos_token_id=128329,
)

top_p_sampling_config = GenerationConfig(
    do_sample=True,
    top_p=0.9,  
    temperature=0.8,
    max_new_tokens=2000,
    eos_token_id=128329,
)

GENERATION_CONFIGS = {
    "greedy": greedy_config,
    "beam_search": beam_search_config,
    "top_p": top_p_sampling_config
}
