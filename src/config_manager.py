"""
Configuration management for the face swapping project.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for face detection models."""
    predictor_path: str = "models/shape_predictor_68_face_landmarks.dat"
    confidence_threshold: float = 0.5
    max_faces: int = 1


@dataclass
class ProcessingConfig:
    """Configuration for image processing parameters."""
    resize_width: int = 500
    clone_method: int = 1  # cv2.NORMAL_CLONE
    border_mode: int = 4  # cv2.BORDER_REFLECT_101
    quality: int = 95


@dataclass
class UIConfig:
    """Configuration for user interface."""
    theme: str = "light"
    show_landmarks: bool = False
    show_triangles: bool = False
    auto_save: bool = True


@dataclass
class AppConfig:
    """Main application configuration."""
    model: ModelConfig
    processing: ProcessingConfig
    ui: UIConfig
    log_level: str = "INFO"
    output_dir: str = "output"
    temp_dir: str = "temp"


class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config_path = config_path or Path("config/config.yaml")
        self.config = self._load_default_config()
        
        if self.config_path.exists():
            self.load_config()
    
    def _load_default_config(self) -> AppConfig:
        """Load default configuration."""
        return AppConfig(
            model=ModelConfig(),
            processing=ProcessingConfig(),
            ui=UIConfig()
        )
    
    def load_config(self) -> None:
        """Load configuration from file."""
        try:
            if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                self._load_yaml_config()
            elif self.config_path.suffix.lower() == '.json':
                self._load_json_config()
            else:
                logger.warning(f"Unsupported config file format: {self.config_path.suffix}")
                
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            logger.info("Using default configuration")
    
    def _load_yaml_config(self) -> None:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        self.config = self._dict_to_config(config_data)
        logger.info(f"Loaded YAML config from: {self.config_path}")
    
    def _load_json_config(self) -> None:
        """Load configuration from JSON file."""
        with open(self.config_path, 'r') as f:
            config_data = json.load(f)
        
        self.config = self._dict_to_config(config_data)
        logger.info(f"Loaded JSON config from: {self.config_path}")
    
    def _dict_to_config(self, config_data: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to AppConfig object."""
        return AppConfig(
            model=ModelConfig(**config_data.get('model', {})),
            processing=ProcessingConfig(**config_data.get('processing', {})),
            ui=UIConfig(**config_data.get('ui', {})),
            log_level=config_data.get('log_level', 'INFO'),
            output_dir=config_data.get('output_dir', 'output'),
            temp_dir=config_data.get('temp_dir', 'temp')
        )
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            config_dict = asdict(self.config)
            
            if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                with open(self.config_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif self.config_path.suffix.lower() == '.json':
                with open(self.config_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to: {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get_config(self) -> AppConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown config parameter: {key}")


# Global configuration instance
config_manager = ConfigManager()
