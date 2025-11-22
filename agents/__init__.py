"""
NES Carla Agents Package

This package contains machine learning agents and optimization algorithms for
autonomous driving in the CARLA simulator using Natural Evolution Strategies (NES).

Modules:
    nes_agent: Natural Evolution Strategies optimizer for neural network training
    custom_ml_agent: ML-based autonomous driving agents with sensor integration

Classes:
    NES: Natural Evolution Strategies optimizer
    CustomMLAgent: Basic ML agent for autonomous driving
    CustomAdvancedMLAgent: Advanced ML agent with comprehensive sensor suite

Usage:
    from NES_Carla.agents import NES, CustomAdvancedMLAgent
    from NES_Carla.agents.custom_ml_model import CustomMLModel
    
    # Create model and agent
    model = CustomMLModel()
    agent = CustomAdvancedMLAgent(vehicle, model)
    
    # Setup NES optimizer
    nes = NES(agent, population_size=50, sigma=0.1, learning_rate=0.1)

Author: Manuel Di Lullo
Date: 2025
"""

from .nes_agent import NES
from .custom_ml_agent import CustomAdvancedMLAgent

# Package version
__version__ = "1.0.0"

# Export main classes
__all__ = [
    "NES",
    "CustomAdvancedMLAgent"
]