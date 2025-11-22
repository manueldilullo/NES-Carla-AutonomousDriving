"""
NES Carla Models Package

This package contains machine learning models specifically designed for autonomous
driving in the CARLA simulator. The models are optimized for use with Natural
Evolution Strategies (NES) and other evolutionary optimization algorithms.

Key Features:
- Lightweight models suitable for evolutionary optimization
- Real-time inference capabilities for autonomous driving
- Parameter-based architectures compatible with NES algorithms
- Modular design for easy experimentation and extension

Models Included:
- CustomMLModel: Simple linear model for basic autonomous driving tasks

The models are designed to be:
1. Fast to evaluate (suitable for population-based optimization)
2. Simple enough for evolutionary algorithms to optimize effectively
3. Capable of real-time control in driving scenarios
4. Easy to save/load for model persistence

Usage Example:
    from NES_Carla.models import CustomMLModel
    from NES_Carla.agents import CustomAdvancedMLAgent, NES
    
    # Create model
    model = CustomMLModel(param_size=4)
    
    # Use with agent
    agent = CustomAdvancedMLAgent(vehicle, model)
    
    # Optimize with NES
    nes = NES(agent, population_size=50)

Architecture Guidelines:
- Keep models simple for evolutionary optimization
- Use parameter counts that allow reasonable exploration
- Ensure real-time inference capabilities
- Include proper save/load functionality
- Provide parameter introspection for debugging

Author: Manuel Di Lullo
Date: 2025
"""

from .custom_ml_model import CustomMLModel

# Package metadata
__version__ = "1.0.0"
__author__ = "Manuel Di Lullo"

# Export main classes
__all__ = [
    "CustomMLModel"
]

# Model factory functions for convenience
def create_basic_model(state_dim: int = 4) -> CustomMLModel:
    """
    Create a basic linear model for autonomous driving.
    
    Args:
        state_dim (int): Dimensionality of the input state vector
        
    Returns:
        CustomMLModel: Initialized model ready for training
    """
    return CustomMLModel(param_size=state_dim)

def create_model_from_checkpoint(filepath: str, state_dim: int = 4) -> CustomMLModel:
    """
    Create and load a model from a saved checkpoint.
    
    Args:
        filepath (str): Path to the saved model parameters
        state_dim (int): Expected state dimensionality
        
    Returns:
        CustomMLModel: Loaded model ready for evaluation
    """
    model = CustomMLModel(param_size=state_dim)
    model.load(filepath)
    return model