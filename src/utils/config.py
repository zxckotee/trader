"""
Configuration management for the trading ML project.
"""

import os
from pathlib import Path
from typing import Dict, Any, List
import json


class Config:
    """
    Configuration class for the trading ML project.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_default_config()
        
        if config_path and Path(config_path).exists():
            self.load_from_file(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            # Data settings
            'data': {
                'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'SOLUSDT'],
                'timeframes': ['5m', '30m', '1h', '1d', '1w'],
                'start_date': '2019-01-01',
                'data_dir': './data',
                'sequence_length': 100,
                'prediction_horizons': {
                    '5m': 12,    # 1 hour ahead
                    '30m': 8,    # 4 hours ahead
                    '1h': 6,     # 6 hours ahead
                    '1d': 7,     # 1 week ahead
                    '1w': 4      # 1 month ahead
                }
            },
            
            # Model settings
            'model': {
                'input_dim': 50,  # Will be determined by preprocessor
                'hidden_dim': 256,
                'num_layers': 4,
                'num_heads': 8,
                'dropout': 0.1,
                'use_gating': True
            },
            
            # Training settings
            'training': {
                'batch_size': 32,
                'learning_rate': 1e-4,
                'num_epochs': 100,
                'weight_decay': 0.01,
                'grad_clip': 1.0,
                'scheduler': 'cosine',
                'min_lr': 1e-6,
                'early_stopping': True,
                'early_stopping_patience': 20,
                'num_workers': 'auto',
                'train_split': 0.8,
                'val_split': 0.1,
                'test_split': 0.1,
                'device': 'auto',
                'force_cpu': False
            },
            
            # Loss weights
            'loss': {
                'price_weight': 1.0,
                'direction_weight': 0.5,
                'volatility_weight': 0.3,
                'diversity_weight': 0.1
            },
            
            # LoRA settings
            'lora': {
                'enabled': True,
                'r': 16,
                'lora_alpha': 32,
                'lora_dropout': 0.1,
                'target_modules': [
                    "experts.*.transformer.layers.*.self_attn.q_proj",
                    "experts.*.transformer.layers.*.self_attn.v_proj",
                    "experts.*.transformer.layers.*.linear1",
                    "experts.*.transformer.layers.*.linear2"
                ]
            },
            
            # Preprocessing settings
            'preprocessing': {
                'scaler_type': 'standard',
                'fill_method': 'ffill'
            },
            
            # Inference settings
            'inference': {
                'lookback_hours': 24,
                'prediction_interval': 300,  # 5 minutes
                'confidence_threshold': 0.6
            },
            
            # Paths
            'paths': {
                'data_dir': './data',
                'model_dir': './models',
                'logs_dir': './logs',
                'output_dir': './output'
            },
            
            # API settings
            'api': {
                'bybit_base_url': 'https://api.bybit.com',
                'rate_limit_delay': 0.01,
                'max_retries': 3
            }
        }
    
    def load_from_file(self, config_path: str) -> None:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            file_config = json.load(f)
        
        # Deep merge configurations
        self.config = self._deep_merge(self.config, file_config)
    
    def save_to_file(self, config_path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            config_path: Path to save configuration
        """
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            update: Dictionary to merge
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables."""
        # API credentials
        if os.getenv('BYBIT_API_KEY'):
            self.set('api.bybit_api_key', os.getenv('BYBIT_API_KEY'))
        if os.getenv('BYBIT_SECRET'):
            self.set('api.bybit_secret', os.getenv('BYBIT_SECRET'))
        if os.getenv('BYBIT_TESTNET'):
            self.set('api.bybit_testnet', os.getenv('BYBIT_TESTNET').lower() == 'true')
        
        # Paths
        if os.getenv('DATA_DIR'):
            self.set('paths.data_dir', os.getenv('DATA_DIR'))
        if os.getenv('MODEL_DIR'):
            self.set('paths.model_dir', os.getenv('MODEL_DIR'))
        if os.getenv('LOGS_DIR'):
            self.set('paths.logs_dir', os.getenv('LOGS_DIR'))
    
    def create_directories(self) -> None:
        """Create necessary directories."""
        dirs_to_create = [
            self.get('paths.data_dir'),
            self.get('paths.model_dir'),
            self.get('paths.logs_dir'),
            self.get('paths.output_dir')
        ]
        
        for dir_path in dirs_to_create:
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> List[str]:
        """
        Validate configuration.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check required fields
        required_fields = [
            'data.symbols',
            'data.timeframes',
            'model.input_dim',
            'training.batch_size',
            'training.learning_rate'
        ]
        
        for field in required_fields:
            if self.get(field) is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate data splits
        train_split = self.get('training.train_split', 0.8)
        val_split = self.get('training.val_split', 0.1)
        test_split = self.get('training.test_split', 0.1)
        
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            errors.append("Data splits must sum to 1.0")
        
        # Validate timeframes
        valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w']
        timeframes = self.get('data.timeframes', [])
        
        for tf in timeframes:
            if tf not in valid_timeframes:
                errors.append(f"Invalid timeframe: {tf}")
        
        return errors
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting."""
        self.set(key, value)


# Global configuration instance
config = Config()


def load_config(config_path: str = None) -> Config:
    """
    Load configuration from file or environment.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration instance
    """
    global config
    config = Config(config_path)
    config.update_from_env()
    config.create_directories()
    
    # Validate configuration
    errors = config.validate()
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
    
    return config


def get_config() -> Config:
    """Get global configuration instance."""
    return config
