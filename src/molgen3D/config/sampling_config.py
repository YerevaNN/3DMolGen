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

# HP Sweep configs - top_p with temperature variations
top_p_sweep1 = GenerationConfig(
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
)

top_p_sweep2 = GenerationConfig(
    do_sample=True,
    temperature=1.0,
    top_p=0.9,
)

top_p_sweep3 = GenerationConfig(
    do_sample=True,
    temperature=1.2,
    top_p=0.9,
)

# HP Sweep configs - min_p with temperature variations
min_p_sweep1 = GenerationConfig(
    do_sample=True,
    temperature=0.8,
    min_p=0.1,
)

min_p_sweep2 = GenerationConfig(
    do_sample=True,
    temperature=1.0,
    min_p=0.1,
)

min_p_sweep3 = GenerationConfig(
    do_sample=True,
    temperature=1.2,
    min_p=0.1,
)

# HP Sweep configs - top_k with temperature variations
top_k_sweep1 = GenerationConfig(
    do_sample=True,
    temperature=0.8,
    top_k=50,
)

top_k_sweep2 = GenerationConfig(
    do_sample=True,
    temperature=1.0,
    top_k=50,
)

top_k_sweep3 = GenerationConfig(
    do_sample=True,
    temperature=1.2,
    top_k=50,
)

# HP Sweep Round 2 - Fixed temperature (1.0), vary parameters
# top_p variations
top_p_r2_1 = GenerationConfig(
    do_sample=True,
    temperature=1.0,
    top_p=0.8,
)

top_p_r2_2 = GenerationConfig(
    do_sample=True,
    temperature=1.0,
    top_p=0.9,
)

top_p_r2_3 = GenerationConfig(
    do_sample=True,
    temperature=1.0,
    top_p=0.95,
)

# min_p variations
min_p_r2_1 = GenerationConfig(
    do_sample=True,
    temperature=1.0,
    min_p=0.05,
)

min_p_r2_2 = GenerationConfig(
    do_sample=True,
    temperature=1.0,
    min_p=0.1,
)

min_p_r2_3 = GenerationConfig(
    do_sample=True,
    temperature=1.0,
    min_p=0.15,
)

# top_k variations
top_k_r2_1 = GenerationConfig(
    do_sample=True,
    temperature=1.0,
    top_k=20,
)

top_k_r2_2 = GenerationConfig(
    do_sample=True,
    temperature=1.0,
    top_k=50,
)

top_k_r2_3 = GenerationConfig(
    do_sample=True,
    temperature=1.0,
    top_k=100,
)

# HP Sweep Round 3 - Extended temperature range with top_k variations
# top_k=30 variations
top_k_r3_30_t06 = GenerationConfig(
    do_sample=True,
    temperature=0.6,
    top_k=30,
)

top_k_r3_30_t07 = GenerationConfig(
    do_sample=True,
    temperature=0.7,
    top_k=30,
)

top_k_r3_30_t13 = GenerationConfig(
    do_sample=True,
    temperature=1.3,
    top_k=30,
)

top_k_r3_30_t15 = GenerationConfig(
    do_sample=True,
    temperature=1.5,
    top_k=30,
)

# top_k=50 variations
top_k_r3_50_t06 = GenerationConfig(
    do_sample=True,
    temperature=0.6,
    top_k=50,
)

top_k_r3_50_t07 = GenerationConfig(
    do_sample=True,
    temperature=0.7,
    top_k=50,
)

top_k_r3_50_t13 = GenerationConfig(
    do_sample=True,
    temperature=1.3,
    top_k=50,
)

top_k_r3_50_t15 = GenerationConfig(
    do_sample=True,
    temperature=1.5,
    top_k=50,
)

# top_k=70 variations
top_k_r3_70_t06 = GenerationConfig(
    do_sample=True,
    temperature=0.6,
    top_k=70,
)

top_k_r3_70_t07 = GenerationConfig(
    do_sample=True,
    temperature=0.7,
    top_k=70,
)

top_k_r3_70_t13 = GenerationConfig(
    do_sample=True,
    temperature=1.3,
    top_k=70,
)

top_k_r3_70_t15 = GenerationConfig(
    do_sample=True,
    temperature=1.5,
    top_k=70,
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
    # HP Sweep configs
    "top_p_sweep1": top_p_sweep1,
    "top_p_sweep2": top_p_sweep2,
    "top_p_sweep3": top_p_sweep3,
    "min_p_sweep1": min_p_sweep1,
    "min_p_sweep2": min_p_sweep2,
    "min_p_sweep3": min_p_sweep3,
    "top_k_sweep1": top_k_sweep1,
    "top_k_sweep2": top_k_sweep2,
    "top_k_sweep3": top_k_sweep3,
    # HP Sweep Round 2 configs
    "top_p_r2_1": top_p_r2_1,
    "top_p_r2_2": top_p_r2_2,
    "top_p_r2_3": top_p_r2_3,
    "min_p_r2_1": min_p_r2_1,
    "min_p_r2_2": min_p_r2_2,
    "min_p_r2_3": min_p_r2_3,
    "top_k_r2_1": top_k_r2_1,
    "top_k_r2_2": top_k_r2_2,
    "top_k_r2_3": top_k_r2_3,
    # HP Sweep Round 3 configs - extended temperature range
    "top_k_r3_30_t06": top_k_r3_30_t06,
    "top_k_r3_30_t07": top_k_r3_30_t07,
    "top_k_r3_30_t13": top_k_r3_30_t13,
    "top_k_r3_30_t15": top_k_r3_30_t15,
    "top_k_r3_50_t06": top_k_r3_50_t06,
    "top_k_r3_50_t07": top_k_r3_50_t07,
    "top_k_r3_50_t13": top_k_r3_50_t13,
    "top_k_r3_50_t15": top_k_r3_50_t15,
    "top_k_r3_70_t06": top_k_r3_70_t06,
    "top_k_r3_70_t07": top_k_r3_70_t07,
    "top_k_r3_70_t13": top_k_r3_70_t13,
    "top_k_r3_70_t15": top_k_r3_70_t15,
}

gen_num_codes = {
    "1x_per_mol": 1,
    "2x_per_mol": 2,
    "1k_per_conf": "1k",
    "2k_per_conf": "2k"
}