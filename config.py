"""
Centralized Configuration System for NES-CARLA Training

This module provides a unified configuration system for all training modes,
making hyperparameters easy to tune and experiments reproducible.

Features:
- Type-safe configuration classes with validation
- Easy serialization/deserialization for experiment tracking
- Default values optimized through experimentation
- Clear documentation of all hyperparameters

Author: Manuel Di Lullo
Date: 2025
"""

import json
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, Literal
from pathlib import Path


@dataclass
class ModelConfig:
    """
    Configuration for neural network models.
    
    Attributes:
        model_type: Type of model ('linear', 'temporal')
        input_size: Number of input features
        memory_size: Number of timesteps to remember (temporal only)
        decay_factor: Memory decay rate [0,1] (temporal only)
    """
    model_type: Literal['linear', 'temporal'] = 'temporal'
    input_size: int = 10  # Updated: 10 for comprehensive state (speed, dist_goal, angle_goal, dist_wp, angle_wp, collision, lane_invasion, steer, throttle, brake)
    memory_size: int = 5
    decay_factor: float = 0.85
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.input_size < 1:
            raise ValueError(f"input_size must be >= 1, got {self.input_size}")
        if self.memory_size < 1:
            raise ValueError(f"memory_size must be >= 1, got {self.memory_size}")
        if not 0 <= self.decay_factor <= 1:
            raise ValueError(f"decay_factor must be in [0,1], got {self.decay_factor}")
    
    def get_param_size(self) -> int:
        """Calculate total parameter size based on model type."""
        if self.model_type == 'linear':
            # Linear model: 3 weight vectors (steering, throttle, brake)
            return self.input_size * 3
        else:  # temporal
            # Temporal model: (input + memory_elements) * 2 controls
            memory_elements = self.memory_size * 3
            extended_size = self.input_size + memory_elements
            return extended_size * 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary."""
        return cls(**config_dict)


@dataclass
class NESConfig:
    """
    Configuration for Natural Evolution Strategies optimizer.
    
    Attributes:
        population_size: Number of individuals per generation
        sigma: Exploration noise (standard deviation)
        learning_rate: Parameter update step size
        min_sigma: Minimum sigma to maintain exploration
        sigma_decay: Decay rate for sigma per generation
    """
    population_size: int = 50
    sigma: float = 0.1
    learning_rate: float = 0.01
    min_sigma: float = 0.05
    sigma_decay: float = 0.995
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.population_size < 2:
            raise ValueError(f"population_size must be >= 2, got {self.population_size}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if not 0 <= self.min_sigma <= self.sigma:
            raise ValueError(f"min_sigma must be in [0, sigma], got {self.min_sigma}")
        if not 0 < self.sigma_decay <= 1:
            raise ValueError(f"sigma_decay must be in (0,1], got {self.sigma_decay}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'NESConfig':
        """Create NESConfig from dictionary."""
        return cls(**config_dict)


@dataclass
class TrainingConfig:
    """
    Configuration for training process.
    
    Attributes:
        training_type: Type of training ('nes', 'temporal', 'batch')
        num_generations: Total number of training generations
        max_steps: Maximum steps per episode
        min_route_distance: Minimum route distance in meters
        max_route_distance: Maximum route distance in meters
        completion_bonus: Reward for completing route
        save_interval: Save checkpoint every N generations
        print_interval: Print stats every N generations
        visualize_waypoints: Draw waypoints in simulator
        batch_size: Number of vehicles for batch training
        num_npc_vehicles: Number of NPC vehicles to spawn
        num_npc_pedestrians: Number of NPC pedestrians to spawn
    """
    training_type: Literal['nes', 'temporal', 'batch'] = 'temporal'
    num_generations: int = 300
    max_steps: int = 600
    min_route_distance: float = 50.0
    max_route_distance: float = 150.0
    completion_bonus: float = 500.0
    save_interval: int = 10
    print_interval: int = 1
    visualize_waypoints: bool = False
    batch_size: int = 4
    num_npc_vehicles: int = 0
    num_npc_pedestrians: int = 0
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.num_generations < 1:
            raise ValueError(f"num_generations must be >= 1, got {self.num_generations}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.min_route_distance < 0:
            raise ValueError(f"min_route_distance must be >= 0, got {self.min_route_distance}")
        if self.max_route_distance < self.min_route_distance:
            raise ValueError(f"max_route_distance must be >= min_route_distance")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_npc_vehicles < 0:
            raise ValueError(f"num_npc_vehicles must be >= 0, got {self.num_npc_vehicles}")
        if self.num_npc_pedestrians < 0:
            raise ValueError(f"num_npc_pedestrians must be >= 0, got {self.num_npc_pedestrians}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create TrainingConfig from dictionary."""
        return cls(**config_dict)


@dataclass
class Config:
    """
    Complete configuration bundle for training.
    
    Combines model, NES, and training configurations into a single
    object for easy management and serialization.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    nes: NESConfig = field(default_factory=NESConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert complete config to dictionary."""
        return {
            'model': self.model.to_dict(),
            'nes': self.nes.to_dict(),
            'training': self.training.to_dict()
        }
    
    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Configuration saved to {filepath}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        return cls(
            model=ModelConfig.from_dict(config_dict.get('model', {})),
            nes=NESConfig.from_dict(config_dict.get('nes', {})),
            training=TrainingConfig.from_dict(config_dict.get('training', {}))
        )
    
    @classmethod
    def from_json(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def print_summary(self) -> None:
        """Print human-readable configuration summary."""
        print("\n" + "=" * 70)
        print("Training Configuration Summary")
        print("=" * 70)
        
        print("\n📊 Model Configuration:")
        print(f"  Type: {self.model.model_type}")
        print(f"  Input size: {self.model.input_size}")
        if self.model.model_type == 'temporal':
            print(f"  Memory size: {self.model.memory_size} timesteps")
            print(f"  Decay factor: {self.model.decay_factor}")
        print(f"  Total parameters: {self.model.get_param_size()}")
        
        print("\n🧬 NES Configuration:")
        print(f"  Population size: {self.nes.population_size}")
        print(f"  Sigma (exploration): {self.nes.sigma}")
        print(f"  Learning rate: {self.nes.learning_rate}")
        print(f"  Min sigma: {self.nes.min_sigma}")
        print(f"  Sigma decay: {self.nes.sigma_decay}")
        
        print("\n🚗 Training Configuration:")
        print(f"  Training type: {self.training.training_type}")
        print(f"  Generations: {self.training.num_generations}")
        print(f"  Steps per episode: {self.training.max_steps}")
        print(f"  Route distance: {self.training.min_route_distance:.0f}-{self.training.max_route_distance:.0f}m")
        print(f"  Completion bonus: {self.training.completion_bonus:.0f}")
        if self.training.training_type == 'batch':
            print(f"  Batch size: {self.training.batch_size} vehicles")
        print(f"  NPC vehicles: {self.training.num_npc_vehicles}")
        print(f"  NPC pedestrians: {self.training.num_npc_pedestrians}")
        print(f"  Save interval: {self.training.save_interval} generations")
        print("=" * 70 + "\n")


# Predefined configurations for different use cases

def get_quick_test_config() -> Config:
    """
    Configuration for quick testing (fast iterations, small population).
    
    Use this for:
    - Debugging code changes
    - Testing new features
    - Quick validation
    """
    return Config(
        model=ModelConfig(
            model_type='temporal',
            input_size=10,  # Comprehensive state with lane invasion
            memory_size=3,
            decay_factor=0.8
        ),
        nes=NESConfig(
            population_size=6,
            sigma=0.2,
            learning_rate=0.1
        ),
        training=TrainingConfig(
            training_type='temporal',
            num_generations=50,
            max_steps=300,
            min_route_distance=30.0,
            max_route_distance=80.0,
            save_interval=10
        )
    )


def get_standard_temporal_config() -> Config:
    """
    Standard configuration for temporal model training.
    
    Use this for:
    - Production training runs
    - Baseline experiments
    - Reproducible results
    """
    return Config(
        model=ModelConfig(
            model_type='temporal',
            input_size=10,  # Comprehensive state with lane invasion
            memory_size=5,
            decay_factor=0.85
        ),
        nes=NESConfig(
            population_size=50,  # Increased from 12 for better exploration
            sigma=0.1,           # Balanced exploration
            learning_rate=0.01   # Smaller steps for stability
        ),
        training=TrainingConfig(
            training_type='temporal',
            num_generations=300,
            max_steps=600,
            min_route_distance=50.0,
            max_route_distance=150.0,
            completion_bonus=500.0
        )
    )


def get_batch_training_config() -> Config:
    """
    Configuration for fast batch training with parallel evaluation.
    
    Use this for:
    - Fast iteration during development
    - Large-scale hyperparameter search
    - When GPU is available
    """
    return Config(
        model=ModelConfig(
            model_type='temporal',
            input_size=10,  # Comprehensive state with lane invasion
            memory_size=5,
            decay_factor=0.85
        ),
        nes=NESConfig(
            population_size=50,  # Increased from 12
            sigma=0.1,           # Balanced exploration
            learning_rate=0.01   # Smaller steps
        ),
        training=TrainingConfig(
            training_type='batch',
            num_generations=300,
            max_steps=600,
            batch_size=4,
            completion_bonus=500.0
        )
    )


def get_linear_model_config() -> Config:
    """
    Configuration for simple linear model training.
    
    Use this for:
    - Baseline comparisons
    - Understanding model capacity requirements
    - Fast prototyping
    """
    return Config(
        model=ModelConfig(
            model_type='linear',
            input_size=5,  # Basic features only
            memory_size=0,  # Not used for linear
            decay_factor=0.0  # Not used for linear
        ),
        nes=NESConfig(
            population_size=10,
            sigma=0.1,
            learning_rate=0.1
        ),
        training=TrainingConfig(
            training_type='nes',
            num_generations=500,
            max_steps=300,
            completion_bonus=500.0
        )
    )


# Configuration presets registry
CONFIG_PRESETS = {
    'quick_test': get_quick_test_config,
    'standard': get_standard_temporal_config,
    'batch': get_batch_training_config,
    'linear': get_linear_model_config
}


def get_config(preset: str = 'standard') -> Config:
    """
    Get a configuration by preset name.
    
    Args:
        preset: Name of configuration preset
                Options: 'quick_test', 'standard', 'batch', 'linear'
    
    Returns:
        Config: Configuration object
    
    Raises:
        ValueError: If preset name is not recognized
    """
    if preset not in CONFIG_PRESETS:
        available = ', '.join(CONFIG_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
    
    return CONFIG_PRESETS[preset]()


if __name__ == '__main__':
    """Demonstrate configuration system usage."""
    print("NES-CARLA Configuration System Demo\n")
    
    # Show all available presets
    print("Available configuration presets:")
    for name in CONFIG_PRESETS.keys():
        print(f"  - {name}")
    
    # Demo: Create and display standard config
    print("\n" + "="*70)
    print("Standard Configuration:")
    print("="*70)
    config = get_standard_temporal_config()
    config.print_summary()
    
    # Demo: Save and load config
    print("\nTesting serialization...")
    config.to_json('demo_config.json')
    loaded_config = Config.from_json('demo_config.json')
    print("✓ Configuration saved and loaded successfully")
    
    # Demo: Modify config programmatically
    print("\nModifying configuration programmatically...")
    config.training.num_generations = 500
    config.nes.population_size = 20
    print(f"  Changed num_generations to {config.training.num_generations}")
    print(f"  Changed population_size to {config.nes.population_size}")
    
    print("\nDemo complete!")
