from typing import List, Dict, Any
from dataclasses import dataclass, field
from typing import Optional
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
    reward_weight_rmsd: float
    reward_weight_match: float
    rmsd_const: float
    max_ground_truths: int
    checkpoint_base_dir: str
    # Optional parameters (with defaults)
    num_iterations: int = 1
    max_steps: Optional[int] = None
    num_epochs: Optional[int] = None
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
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict['model']),
            generation=GenerationConfig(**config_dict['generation']),
            processing=ProcessingConfig(**config_dict['processing']),
            grpo=GRPOConfig(**config_dict['grpo']),
            dataset=DatasetConfig(**config_dict['dataset']),
            run=RunConfig(**config_dict['run']),
            device=DeviceConfig(**config_dict['device']),
            trainer=TrainerConfig(**config_dict.get('trainer', {})),
            dataloader=DataLoaderConfig(**config_dict.get('dataloader', {}))
        )