from transformers import GenerationConfig

greedy_config = GenerationConfig(
    do_sample=False,
)

beam_search_config = GenerationConfig(
    num_beams=4,  
    num_beam_groups=1,
    diversity_penalty=0,
)

top_p_sampling_config1 = GenerationConfig (
  do_sample=True,
  temperature=1.0,
  top_p=0.8
)

top_p_sampling_config2 = GenerationConfig (
  do_sample=True,
  temperature=1.2,
  top_p=0.8
)

top_p_sampling_config3 = GenerationConfig (
  do_sample=True,
  temperature=1.0,
  top_p=0.6
)

min_p_sampling_config1= GenerationConfig(
    do_sample=True,
    temperature=1.0,
    min_p = 0.05,
)

min_p_sampling_config2= GenerationConfig(
    do_sample=True,
    temperature=1.2,
    min_p = 0.05,
)

min_p_sampling_config3= GenerationConfig(
    do_sample=True,
    temperature=1.0,
    min_p = 0.1,
)

sampling_configs = {
    "greedy": greedy_config,
    "beam_search": beam_search_config,
    "top_p_sampling1": top_p_sampling_config1,
    "top_p_sampling2": top_p_sampling_config2,
    "top_p_sampling3": top_p_sampling_config3,
    "min_p_sampling1": min_p_sampling_config1,
    "min_p_sampling2": min_p_sampling_config2,
    "min_p_sampling3": min_p_sampling_config3,
}

gen_num_codes = {
    "1x_per_mol": 1,
    "2x_per_mol": 2,
    "1k_per_conf": "1k",
    "2k_per_conf": "2k"
}