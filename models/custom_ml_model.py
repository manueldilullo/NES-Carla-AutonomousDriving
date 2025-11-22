"""
Custom Machine Learning Model for Autonomous Driving

This module implements a simple neural network model specifically designed for
autonomous driving tasks in the CARLA simulator. The model uses a linear
transformation approach to map vehicle state information to driving actions.

Key Features:
- Lightweight linear model for fast inference
- Parameter-based control suitable for evolutionary optimization
- Direct state-to-action mapping for real-time driving decisions
- Compatible with Natural Evolution Strategies (NES) optimization
- Configurable parameter dimensionality for different state representations

The model architecture:
- Input: Vehicle state vector (position, velocity, sensor data)
- Processing: Linear transformation with learnable parameters
- Output: Driving actions (steering, throttle, brake)

Author: Manuel Di Lullo
Date: 2025
Dependencies: numpy
"""

import numpy as np
from typing import Tuple, Union, List, Optional


class CustomMLModel:
    """
    Simple linear model for autonomous driving control in CARLA.
    
    This model implements a minimal neural network architecture using a single
    linear layer to map vehicle state to driving actions. The simplicity makes
    it ideal for evolutionary optimization methods like NES, where the parameter
    space needs to be manageable and the model needs to be fast to evaluate.
    
    The model performs the following transformation:
    1. Takes vehicle state as input (position, velocity, sensors)
    2. Applies linear transformation: output = parameters · state
    3. Maps output to driving actions (steering, throttle, brake)
    4. Applies appropriate activation functions for action bounds
    
    Architecture:
    - Linear layer: state_dim -> 1 output
    - Activation functions: tanh for steering, sigmoid-like for throttle
    - No brake control in base implementation (can be extended)
    
    Attributes:
        parameters (np.ndarray): Learnable parameters of the model
        param_size (int): Number of parameters in the model
    """
    
    def __init__(self, param_size: int = 4):
        """
        Initialize the custom ML model with random parameters.
        
        Creates a simple linear model with the specified number of parameters.
        The parameters are initialized using a normal distribution to provide
        a good starting point for optimization algorithms.
        
        Args:
            param_size (int): Number of parameters in the model (default: 4)
                            Should match the dimensionality of the input state vector.
                            Common values:
                            - 3: Basic state (x, y, speed)
                            - 4: Basic state + collision flag
                            - 10+: Extended state with sensor data
        
        Note:
            The parameter size should match the expected input state dimensionality.
            For basic vehicle state (x, y, speed), use param_size=3.
            For extended state with collision detection, use param_size=4.
        """
        self.param_size = param_size
        # Initialize parameters with small random values from normal distribution
        # Small values help with initial stability during training
        self.parameters = np.random.randn(param_size) * 0.1

    def set_parameters(self, params: Union[np.ndarray, List[float]]) -> None:
        """
        Set the model parameters to new values.
        
        This method is crucial for evolutionary optimization algorithms like NES,
        which update the model parameters based on fitness evaluations. The method
        ensures parameters are stored as numpy arrays for efficient computation.
        
        Args:
            params (Union[np.ndarray, List[float]]): New parameter values
                                                   Must have length equal to param_size
        
        Raises:
            ValueError: If parameter array size doesn't match expected param_size
        """
        params_array = np.array(params)
        if len(params_array) != self.param_size:
            raise ValueError(f"Parameter size mismatch: expected {self.param_size}, got {len(params_array)}")
        
        self.parameters = params_array

    def old_predict(self, state: Union[List[float], np.ndarray]) -> Tuple[float, float, float]:
        """
        Predict driving actions based on current vehicle state.
        
        This is the core inference method that transforms vehicle state into
        driving commands. The method applies a linear transformation followed
        by appropriate activation functions to ensure actions are within valid ranges.
        
        Processing pipeline:
        1. Convert state to numpy array for efficient computation
        2. Compute linear output: dot product of parameters and state
        3. Apply activation functions to map to valid action ranges:
           - Steering: tanh activation for [-1, 1] range
           - Throttle: scaled sigmoid for [0, 1] range
           - Brake: constant 0.0 (can be extended for more complex models)
        
        Args:
            state (Union[List[float], np.ndarray]): Current vehicle state vector
                Expected format: [x_position, y_position, speed, ...additional_features]
                
        Returns:
            Tuple[float, float, float]: Driving actions (steering, throttle, brake)
                - steering: Steering angle in [-1, 1] range (left negative, right positive)
                - throttle: Throttle amount in [0, 1] range (0 = no throttle, 1 = full throttle)
                - brake: Brake amount in [0, 1] range (currently fixed at 0.0)
        
        Note:
            The model assumes state vector length matches parameter size.
            For robust operation, consider padding or truncating state vector if needed.
        """
        # Convert state to numpy array for vectorized operations
        state_array = np.array(state)
        
        # Ensure state and parameters have compatible dimensions
        if len(state_array) != len(self.parameters):
            # Handle dimension mismatch by padding or truncating
            if len(state_array) < len(self.parameters):
                # Pad state with zeros if too short
                state_array = np.pad(state_array, (0, len(self.parameters) - len(state_array)))
            else:
                # Truncate state if too long
                state_array = state_array[:len(self.parameters)]
        
        # Compute linear transformation: output = parameters · state
        linear_output = np.dot(self.parameters, state_array)
        
        # Apply activation functions to map to valid action ranges
        
        # Steering: Use tanh to map to [-1, 1] range
        # tanh provides smooth transitions and natural bounds
        steering = np.tanh(linear_output)
        
        # Throttle: Transform to [0, 1] range using shifted and scaled output
        # (output + 1) / 2 maps from [-inf, inf] to [0, 1] approximately
        # Clip to ensure strict bounds
        throttle = np.clip((linear_output + 1) / 2, 0.0, 1.0)
        
        # Brake: Currently fixed at 0.0 for simplicity
        # Can be extended to use separate parameters or more complex logic
        brake = 0.0
        
        return float(steering), float(throttle), float(brake)
    
    def predict(self, state: Union[List[float], np.ndarray]) -> Tuple[float, float, float]:
        """
        Predict driving actions based on current vehicle state using separate parameter sets.
        
        This method implements a more sophisticated linear model that uses separate sets of
        parameters for each action output (steering, throttle, brake). Unlike the old_predict
        method, this approach allows for independent control of each action dimension.

        Processing pipeline:
        1. Convert state to numpy array and handle dimension matching
        2. Split parameters into three equal groups for each action
        3. Compute linear transformations for each action independently
        4. Apply appropriate activation functions for valid action ranges
        
        Args:
            state (Union[List[float], np.ndarray]): Current vehicle state vector
                Expected format: [x_position, y_position, speed, ...additional_features]
                
        Returns:
            Tuple[float, float, float]: Driving actions (steering, throttle, brake)
                - steering: Steering angle in [-1, 1] range (left negative, right positive)
                - throttle: Throttle amount in [0, 1] range (0 = no throttle, 1 = full throttle)
                - brake: Brake amount in [0, 1] range (0 = no brake, 1 = full brake)
        
        Note:
            This method requires parameters to be divisible by 3 for equal weight distribution.
            Input state dimensions are automatically adjusted to match parameter expectations.
        """
        x = np.array(state)

        # Adapt input vector dimensions to match expected state size
        # Expected state size is parameters divided by 3 (one set per action)
        expected_state_size = len(self.parameters) // 3
        
        if len(x) < expected_state_size:
            # Pad state with zeros if input is too short
            padded = np.zeros(expected_state_size)
            padded[:len(x)] = x
            x = padded
        elif len(x) > expected_state_size:
            # Truncate state if input is too long
            x = x[:expected_state_size]

        # Split parameters into 3 equal parts for independent action control
        n = len(x)
        steering_weights = self.parameters[:n]           # First n parameters for steering
        throttle_weights = self.parameters[n:2*n]        # Next n parameters for throttle
        brake_weights = self.parameters[2*n:3*n]         # Last n parameters for brake

        # Compute raw linear outputs for each action
        steer_raw = np.dot(steering_weights, x)
        throttle_raw = np.dot(throttle_weights, x)
        brake_raw = np.dot(brake_weights, x)

        # Apply activation functions to map to valid action ranges
        
        # Steering: Use tanh to map to [-1, 1] range
        steer = np.tanh(steer_raw)
        
        # Throttle: Transform to [0, 1] range and clip for safety
        throttle = np.clip((throttle_raw + 1) / 2, 0, 1)
        
        # Brake: Transform to [0, 1] range and clip for safety (now variable and normalized)
        brake = np.clip((brake_raw + 1) / 2, 0, 1)

        return float(steer), float(throttle), float(brake)

    def save(self, filepath: str) -> None:
        """
        Save model parameters to a file for persistence.
        
        Saves the current model parameters to a numpy binary file (.npy format)
        for later loading and evaluation. This is essential for preserving
        trained models and resuming training from checkpoints.
        
        Args:
            filepath (str): Path where to save the parameters
                          Should include .npy extension for clarity
                          
        Example:
            model.save('trained_model.npy')
            model.save('checkpoints/generation_100.npy')
        
        Note:
            The saved file contains only the parameter array.
            Model architecture (param_size) should be known when loading.
        """
        try:
            np.save(filepath, self.parameters)
            print(f"Model parameters saved to {filepath}")
        except Exception as e:
            print(f"Error saving model parameters: {e}")
            raise

    def load(self, filepath: str) -> None:
        """
        Load model parameters from a saved file.
        
        Loads previously saved parameters from a numpy binary file and updates
        the model state. This allows resuming training or evaluating pre-trained models.
        
        Args:
            filepath (str): Path to the saved parameter file (.npy format)
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            ValueError: If loaded parameters don't match expected dimensions
            
        Example:
            model.load('trained_model.npy')
            model.load('checkpoints/best_model.npy')
        
        Note:
            The loaded parameters must have the same dimensionality as the current
            model configuration (param_size). Consider validation after loading.
        """
        try:
            loaded_params = np.load(filepath)
            
            # Validate parameter dimensions
            if len(loaded_params) != self.param_size:
                raise ValueError(f"Parameter size mismatch: model expects {self.param_size}, "
                               f"file contains {len(loaded_params)}")
            
            self.parameters = loaded_params
            print(f"Model parameters loaded from {filepath}")
            
        except FileNotFoundError:
            print(f"Error: Parameter file not found: {filepath}")
            raise
        except Exception as e:
            print(f"Error loading model parameters: {e}")
            raise
    
    def get_parameter_info(self) -> dict:
        """
        Get information about current model parameters for debugging and monitoring.
        
        Returns:
            dict: Parameter statistics including:
                - param_count: Number of parameters
                - param_mean: Mean value of parameters
                - param_std: Standard deviation of parameters
                - param_min: Minimum parameter value
                - param_max: Maximum parameter value
                - param_norm: L2 norm of parameter vector
        """
        return {
            'param_count': len(self.parameters),
            'param_mean': float(np.mean(self.parameters)),
            'param_std': float(np.std(self.parameters)),
            'param_min': float(np.min(self.parameters)),
            'param_max': float(np.max(self.parameters)),
            'param_norm': float(np.linalg.norm(self.parameters))
        }
    
    def reset_parameters(self, seed: Optional[int] = None) -> None:
        """
        Reset parameters to random initial values.
        
        Useful for restarting training or comparing different initializations.
        
        Args:
            seed (Optional[int]): Random seed for reproducible initialization
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.parameters = np.random.randn(self.param_size) * 0.1
        print(f"Parameters reset with {self.param_size} dimensions")
