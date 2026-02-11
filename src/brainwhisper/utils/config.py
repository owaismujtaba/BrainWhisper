"""Configuration management utilities"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration container with dot notation access"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __repr__(self):
        return f"Config({self.__dict__})"
    
    def to_dict(self):
        """Convert back to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Config object with dot notation access
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)


def save_config(config: Config, save_path: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Config object
        save_path: Path to save config
    """
    with open(save_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
