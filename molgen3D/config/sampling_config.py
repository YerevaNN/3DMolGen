from transformers import GenerationConfig

greedy_config = GenerationConfig(
    do_sample=False,
)

beam_search_config = GenerationConfig(
    num_beams=4,  
    num_beam_groups=1,
    diversity_penalty=0,
)

top_p_sampling_config = GenerationConfig (
  do_sample=True,
  temperature=0.8,
  top_p=0.9
)

min_p_sampling_config= GenerationConfig(
    do_sample=True,
    min_p = 0.05,
)

sampling_configs = {
    "greedy": greedy_config,
    "beam_search": beam_search_config,
    "top_p_sampling": top_p_sampling_config,
    "min_p_sampling": min_p_sampling_config,
}

gen_num_codes = {
    "1x_per_mol": 1,
    "2x_per_mol": 2,
    "1k_per_conf": "1k",
    "2k_per_conf": "2k"
}