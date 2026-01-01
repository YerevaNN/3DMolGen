from transformers import GenerationConfig

greedy_config = GenerationConfig(
    do_sample=False,
)

top_p_low_temperature_config = GenerationConfig(
    do_sample=True,
    temperature=0.8,
    top_p=0.8,
)

beam_search_config = GenerationConfig(
    num_beams=4,  
    num_beam_groups=1,
    diversity_penalty=0,
)

top_p_sampling_config1 = GenerationConfig (
  do_sample=True,
  temperature=1.0,
  top_p=1.0
)

top_p_sampling_config2 = GenerationConfig (
  do_sample=True,
  temperature=1.0,
  top_p=0.8
)

top_p_sampling_config3 = GenerationConfig (
  do_sample=True,
  temperature=1.0,
  top_p=0.6
)

top_p_sampling_config4 = GenerationConfig (
  do_sample=True,
  temperature=1.0,
  top_p=0.4
)

top_p_sampling_config5 = GenerationConfig (
  do_sample=True,
  temperature=1.0,
  top_p=1.0
)

top_p_sampling_config6 = GenerationConfig (
  do_sample=True,
  temperature=1.0,
  top_p=0.8
)

top_p_sampling_config7 = GenerationConfig (
  do_sample=True,
  temperature=1.0,
  top_p=0.6
)

top_p_sampling_config8 = GenerationConfig (
  do_sample=True,
  temperature=1.0,
  top_p=0.4
)

top_p_sampling_config9 = GenerationConfig (
  do_sample=True,
  temperature=1.2,
  top_p=1.0
)

top_p_sampling_config10 = GenerationConfig (
  do_sample=True,
  temperature=1.2,
  top_p=0.8
)

top_p_sampling_config11 = GenerationConfig (
  do_sample=True,
  temperature=1.2,
  top_p=0.6
)

top_p_sampling_config12 = GenerationConfig (
  do_sample=True,
  temperature=1.2,
  top_p=0.4
)

top_p_sampling_config13 = GenerationConfig (
  do_sample=True,
  temperature=1.4,
  top_p=1.0
)

top_p_sampling_config14 = GenerationConfig (
  do_sample=True,
  temperature=1.4,
  top_p=0.8
)

top_p_sampling_config15 = GenerationConfig (
  do_sample=True,
  temperature=1.4,
  top_p=0.6
)

top_p_sampling_config16 = GenerationConfig (
  do_sample=True,
  temperature=1.4,
  top_p=0.4
)

min_p_sampling_config1= GenerationConfig(
    do_sample=True,
    temperature=1.4,
    min_p = 0.1,
)

min_p_sampling_config2= GenerationConfig(
    do_sample=True,
    temperature=1.4,
    min_p = 0.075,
)

min_p_sampling_config3= GenerationConfig(
    do_sample=True,
    temperature=1.4,
    min_p = 0.05,
)
min_p_sampling_config4= GenerationConfig(
    do_sample=True,
    temperature=1.2,
    min_p = 0.1,
)

min_p_sampling_config5= GenerationConfig(
    do_sample=True,
    temperature=1.2,
    min_p = 0.075,
)

min_p_sampling_config6= GenerationConfig(
    do_sample=True,
    temperature=1.2,
    min_p = 0.05,
)
min_p_sampling_config7= GenerationConfig(
    do_sample=True,
    temperature=1.0,
    min_p = 0.1,
)

min_p_sampling_config8= GenerationConfig(
    do_sample=True,
    temperature=1.0,
    min_p = 0.075,
)

min_p_sampling_config9= GenerationConfig(
    do_sample=True,
    temperature=1.0,
    min_p = 0.05,
)

min_p_sampling_config10= GenerationConfig(
    do_sample=True,
    temperature=0.8,
    min_p = 0.1,
)

min_p_sampling_config11= GenerationConfig(
    do_sample=True,
    temperature=0.8,
    min_p = 0.075,
)

min_p_sampling_config12= GenerationConfig(
    do_sample=True,
    temperature=0.8,
    min_p = 0.05,
)
sampling_configs = {
    "greedy": greedy_config,
    "top_p_low_temperature": top_p_low_temperature_config,
    "beam_search": beam_search_config,
    "top_p_sampling1": top_p_sampling_config1,
    "top_p_sampling2": top_p_sampling_config2,
    "top_p_sampling3": top_p_sampling_config3,
    "top_p_sampling4": top_p_sampling_config4,
    "top_p_sampling5": top_p_sampling_config5,
    "top_p_sampling6": top_p_sampling_config6,
    "top_p_sampling7": top_p_sampling_config7,
    "top_p_sampling8": top_p_sampling_config8,
    "top_p_sampling9": top_p_sampling_config9,
    "top_p_sampling10": top_p_sampling_config10,
    "top_p_sampling11": top_p_sampling_config11,
    "top_p_sampling12": top_p_sampling_config12,
    "top_p_sampling13": top_p_sampling_config13,
    "top_p_sampling14": top_p_sampling_config14,
    "top_p_sampling15": top_p_sampling_config15,
    "top_p_sampling16": top_p_sampling_config16,    
    "min_p_sampling1": min_p_sampling_config1,
    "min_p_sampling2": min_p_sampling_config2,
    "min_p_sampling3": min_p_sampling_config3,
    "min_p_sampling4": min_p_sampling_config4,
    "min_p_sampling5": min_p_sampling_config5,
    "min_p_sampling6": min_p_sampling_config6,
    "min_p_sampling7": min_p_sampling_config7,
    "min_p_sampling8": min_p_sampling_config8,
    "min_p_sampling9": min_p_sampling_config9,
    "min_p_sampling10": min_p_sampling_config10,
    "min_p_sampling11": min_p_sampling_config11,
    "min_p_sampling12": min_p_sampling_config12,
}

gen_num_codes = {
    "1x_per_mol": 1,
    "2x_per_mol": 2,
    "1k_per_conf": "1k",
    "2k_per_conf": "2k"
}