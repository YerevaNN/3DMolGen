from typing import List, Optional
from dataclasses import dataclass, field
import yaml


@dataclass
class ModelConfig:
    checkpoint_path: str
    tokenizer_path: str
    mol_tags: List[str]
    conf_tags: List[str]
    pad_token: str
    dtype: str


@dataclass
class GenerationConfig:
    max_completion_length: int
    temperature: float
    do_sample: bool
    repetition_penalty: float
    num_return_sequences: int


@dataclass
class ProcessingConfig:
    eos_token_id: int


@dataclass
class PoseBustersGateConfig:
    mode: str = "off"
    max_workers: int = 0
    chunk_size: int = 100
    energy_num_threads: int = 1


@dataclass
class GRPOConfig:
    # Required parameters (no defaults)
    output_base_dir: str
    learning_rate: float
    temperature: float
    num_generations: int
    grad_acc_steps: int
    per_device_batch_size: int
    scheduler: str
    adam_beta1: float
    adam_beta2: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    beta: float
    seed: int
    max_ground_truths: int
    checkpoint_base_dir: str

    # Legacy reward parameters (with defaults for backward compatibility)
    reward_weight_rmsd: float = 0.85
    reward_weight_match: float = 0.15
    rmsd_const: float = 0.75

    # Strategy selection
    reward_strategy: str = "legacy"
    advanced_reward: "AdvancedRewardConfig" = field(default_factory=lambda: AdvancedRewardConfig())

    # Optional parameters (with defaults)
    num_iterations: int = 1
    max_steps: Optional[int] = None
    num_epochs: Optional[int] = None
    importance_sampling_level: str = "sequence"
    epsilon_low: float = 3e-4
    epsilon_high: float = 4e-4
    steps_per_generation: int = 4

    # V2 reward parameters (optional, with defaults matching spec)
    coverage_delta: float = 0.75
    coverage_scale: float = 0.5
    enable_diversity: bool = True
    diversity_scale: float = 1.0
    precision_scale: float = 0.5
    enable_posebusters: bool = True
    match_partial_credit: bool = False
    weight_precision: float = 0.4
    weight_coverage: float = 0.3
    weight_match: float = 0.2
    weight_validity: float = 0.1
    weight_diversity: float = 0.05

    # V3 reward parameters (GEOM-Drugs aligned)
    delta: float = 0.75        # RMSD threshold (Å)
    sigma: float = 0.25        # Quality scale (Å)
    rho: float = 0.75          # Smooth coverage kernel scale (Å)
    lambda_qual: float = 1.0   # Weight for quality term
    lambda_smcov: float = 1.0  # Weight for smooth coverage term
    lambda_match: float = 1.0  # Weight for matching term
    r_floor: float = -1.0      # Reward for invalid samples
    hard_rmsd_gate: bool = True  # Drop PoseBusters-valid but RMSD-invalid rollouts
    profile_rewards: bool = False
    log_distance_samples_per_group: int = 32
    enable_pairwise_rmsd_logging: bool = False
    pairwise_rmsd_log_every: int = 50
    log_every_steps: int = 1
    posebusters: PoseBustersGateConfig = field(default_factory=PoseBustersGateConfig)

    # Runtime parameters (set during execution)
    output_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None


@dataclass
class DatasetConfig:
    dataset_path: str
    smiles_mapping_path: str  # Path to the SMILES mapping JSON file


@dataclass
class RunConfig:
    name: str
    log_level: str = "INFO"

@dataclass
class DeviceConfig:
    device_type: str  # local, a100, h100
    num_gpus: int


@dataclass
class DataLoaderConfig:
    # DataLoader parameters for memory balancing
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = False

@dataclass
class TrainerConfig:
    # Checkpointing and saving
    save_strategy: str = "steps"
    save_steps: int = 800
    save_total_limit: int = 8
    save_on_each_node: bool = False
    save_safetensors: bool = True
    
    # Logging
    log_on_each_node: bool = False
    logging_steps: int = 1
    
    # Training specific
    use_liger_loss: bool = False
    loss_type: str = "grpo"
    
    # Model loading
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"


@dataclass
class AdvancedRewardWeights:
    precision: float = 0.4
    coverage: float = 0.3
    match: float = 0.2
    validity: float = 0.1
    diversity: float = 0.05


@dataclass
class AdvancedRewardConfig:
    delta: float = 0.75
    precision_scale: float = 0.5
    coverage_scale: float = 0.5
    diversity_scale: float = 1.0
    normalization_epsilon: float = 1e-6
    enable_diversity: bool = True
    enable_posebusters: bool = True
    posebusters_config: str = "mol"
    match_partial_credit: bool = False
    max_reference_conformers: Optional[int] = None
    weights: AdvancedRewardWeights = field(default_factory=AdvancedRewardWeights)


@dataclass
class Config:
    model: ModelConfig
    generation: GenerationConfig
    processing: ProcessingConfig
    grpo: GRPOConfig
    dataset: DatasetConfig
    run: RunConfig
    device: DeviceConfig
    trainer: TrainerConfig
    dataloader: DataLoaderConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from a YAML file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            Config: A Config instance with all parameters loaded from the YAML file
        """
        with open(yaml_path, 'r', encoding='utf-8', errors='replace') as f:
            config_dict = yaml.safe_load(f)

        grpo_dict_raw = dict(config_dict['grpo'])
        reward_strategy = grpo_dict_raw.get('reward_strategy', 'legacy')
        grpo_dict_raw['reward_strategy'] = reward_strategy

        # Backward compatibility: map 'epsilon' to 'epsilon_low' if present
        if 'epsilon' in grpo_dict_raw and 'epsilon_low' not in grpo_dict_raw:
            grpo_dict_raw['epsilon_low'] = grpo_dict_raw.pop('epsilon')

        advanced_reward_dict = grpo_dict_raw.get('advanced_reward', {})
        if advanced_reward_dict is None:
            advanced_reward = AdvancedRewardConfig()
        else:
            weights_dict = advanced_reward_dict.get('weights', {})
            weights = (
                AdvancedRewardWeights(**weights_dict)
                if weights_dict else AdvancedRewardWeights()
            )
            advanced_kwargs = {
                key: value
                for key, value in advanced_reward_dict.items()
                if key != 'weights'
            }
            advanced_reward = AdvancedRewardConfig(weights=weights, **advanced_kwargs)
        grpo_dict_raw['advanced_reward'] = advanced_reward

        posebusters_raw = grpo_dict_raw.get('posebusters')
        if isinstance(posebusters_raw, PoseBustersGateConfig):
            posebusters_config = posebusters_raw
        elif posebusters_raw is None:
            posebusters_config = PoseBustersGateConfig()
        elif isinstance(posebusters_raw, dict):
            posebusters_config = PoseBustersGateConfig(**posebusters_raw)
        elif isinstance(posebusters_raw, str):
            posebusters_config = PoseBustersGateConfig(mode=posebusters_raw)
        else:
            raise TypeError("grpo.posebusters must be a dict, string, or PoseBustersGateConfig instance")
        grpo_dict_raw['posebusters'] = posebusters_config
        
        return cls(
            model=ModelConfig(**config_dict['model']),
            generation=GenerationConfig(**config_dict['generation']),
            processing=ProcessingConfig(**config_dict['processing']),
            grpo=GRPOConfig(**grpo_dict_raw),
            dataset=DatasetConfig(**config_dict['dataset']),
            run=RunConfig(**config_dict['run']),
            device=DeviceConfig(**config_dict['device']),
            trainer=TrainerConfig(**config_dict.get('trainer', {})),
            dataloader=DataLoaderConfig(**config_dict.get('dataloader', {}))
        )
