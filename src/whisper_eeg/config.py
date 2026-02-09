import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

@dataclass
class DataConfig:
    data_dir: str
    max_eeg_length: int
    max_text_length: int
    vocab_size: int
    gaussian_noise: float = 0.0

@dataclass
class ModelConfig:
    use_whisper: bool
    whisper_model_name: str
    eeg_input_dim: int
    hidden_dim: int
    encoder_layers: int
    decoder_layers: int
    num_heads: int
    dropout: float = 0.1
    freeze_decoder: bool = False
    max_generation_length: int = 25

@dataclass
class TrainingConfig:
    batch_size: int
    num_workers: int
    learning_rate: float
    num_epochs: int
    device: str
    checkpoint_dir: str
    weight_decay: float = 0.0
    ctc_weight: float = 0.3 # Loss = Whisper + ctc_weight * CTC

@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
            
        # Handle potential missing keys for backward compatibility
        model_cfg = cfg_dict['model']
        if 'use_whisper' not in model_cfg:
            model_cfg['use_whisper'] = False
            model_cfg['whisper_model_name'] = "openai/whisper-tiny"
            
        if 'freeze_decoder' not in model_cfg:
            model_cfg['freeze_decoder'] = False
            
        if 'max_generation_length' not in model_cfg:
            model_cfg['max_generation_length'] = 100
        
        return cls(
            data=DataConfig(**cfg_dict['data']),
            model=ModelConfig(**model_cfg),
            training=TrainingConfig(**cfg_dict['training'])
        )

    def save(self, path: str):
        cfg_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__
        }
        with open(path, 'w') as f:
            yaml.dump(cfg_dict, f)
