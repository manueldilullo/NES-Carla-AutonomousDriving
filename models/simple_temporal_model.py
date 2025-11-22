"""
Simple Temporal Model for Autonomous Driving

This module implements a lightweight temporal model that adds memory capabilities
to the basic linear model without the complexity of full LSTM. It maintains
the KISS principle while adding sequential processing capabilities.

Key Features:
- Temporal memory without LSTM complexity
- Minimal parameter increase (suitable for NES)
- Simple recurrent connections
- Fast inference for real-time driving
- Compatible with existing NES optimization

Architecture:
- Input: Current state + previous memory
- Processing: Linear transformation + simple recurrence
- Output: Driving actions (steering, throttle, brake)
- Memory: Simple exponential moving average

Author: Manuel Di Lullo
Date: 2025
Dependencies: numpy
"""

import numpy as np
from typing import Tuple, Union, List, Optional


class SimpleTemporalModel:
    """
    Simple temporal model with memory for autonomous driving.
    
    This model extends the basic linear approach by adding a simple memory
    mechanism that remembers previous states and actions. It's much simpler
    than LSTM but provides temporal awareness for better driving decisions.
    
    The model performs:
    1. Maintains exponential moving average of past states
    2. Combines current state with memory
    3. Applies linear transformation to extended state
    4. Updates memory with current information
    
    Architecture:
    - Memory size: Small fixed buffer (3-5 values)
    - Parameters: ~15-30 (still manageable for NES)
    - Recurrence: Simple exponential moving average
    - Output: Same as linear model (steering, throttle, brake)
    
    Attributes:
        input_size (int): Dimensionality of input state
        memory_size (int): Size of memory buffer
        param_size (int): Total number of parameters
        parameters (np.ndarray): Model parameters
        memory (np.ndarray): Internal memory state
        decay_factor (float): Memory decay rate
    """
    
    def __init__(self, input_size: int = 5, memory_size: int = 10, decay_factor: float = 0.7):
        """
        Initialize the simple temporal model.
        
        Args:
            input_size (int): Dimensionality of input state vector (default: 5)
                             Enhanced: [speed, waypoint_dist, waypoint_angle, obstacle_dist, collision_flag,
                                       curvature_ahead, lateral_deviation, velocity_to_obstacle, steering_derivative]
            memory_size (int): Number of past timesteps to remember (default: 10)
                              Memory stores [steering, throttle, brake] for last N steps
            decay_factor (float): Memory decay rate [0,1] (default: 0.7)
                                0.5 = balanced (50% old, 50% new)
                                0.8 = long memory (good for smooth highways)
                                0.2 = reactive (good for tight turns)
        
        Note:
            Total parameters = (input_size + memory_size*3) * 2 controls
            Memory stores 3 values per timestep (steering, throttle, brake)
            Uses 2 weight vectors: W_steer and W_accel (unified throttle/brake control)
            Example: (9 + 10*3) * 2 = 78 parameters
        """
        self.input_size = input_size
        self.memory_size = memory_size
        self.decay_factor = decay_factor
        
        # Memory stores [steering, throttle, brake] from last N steps
        # Total memory elements = memory_size * 3
        memory_elements = memory_size * 3
        
        # Calculate total parameters needed
        # Extended state = current state + memory
        extended_size = input_size + memory_elements
        
        # We use 2 weight vectors: steering and acceleration (unified control)
        # Acceleration: positive = throttle, negative = brake
        self.param_size = extended_size * 2
        
        # Initialize parameters using Xavier/Glorot initialization for better gradient flow
        xavier_std = np.sqrt(2.0 / (extended_size + 2))  # 2 = number of outputs
        self.parameters = np.random.randn(self.param_size) * xavier_std
        
        # CRITICAL FIX: Add STRONG bias to acceleration to FORCE movement
        # This must be strong enough to overcome NES noise (sigma=0.3)
        # Default behavior should be to accelerate
        accel_params_start = extended_size
        accel_params_end = 2 * extended_size
        self.parameters[accel_params_start:accel_params_end] += 2.5  # Strong positive bias for throttle
        
        # Initialize memory buffer (stores steering, throttle, brake for last N steps)
        self.memory = np.zeros(memory_elements)
        
        print(f"SimpleTemporalModel initialized:")
        print(f"  Input size: {input_size}")
        print(f"  Memory size: {memory_size} timesteps x 3 controls = {memory_elements} elements")
        print(f"  Total parameters: {self.param_size}")
        print(f"  Decay factor: {decay_factor}")
        print(f"  Xavier std: {xavier_std:.4f}")
    
    def set_parameters(self, params: Union[np.ndarray, List[float]]) -> None:
        """
        Set the model parameters to new values.
        
        Args:
            params: New parameter values (must match param_size)
        
        Raises:
            ValueError: If parameter array size doesn't match expected param_size
        """
        params_array = np.array(params)
        if len(params_array) != self.param_size:
            raise ValueError(f"Parameter size mismatch: expected {self.param_size}, got {len(params_array)}")
        
        self.parameters = params_array
    
    def get_parameters(self) -> np.ndarray:
        """
        Get the current model parameters.
        
        Returns:
            np.ndarray: Copy of current parameter values
        """
        return self.parameters.copy()
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state values to stable ranges for better learning.
        
        This prevents numerical instability and helps NES optimization converge faster.
        
        Args:
            state: Raw state vector [speed, waypoint_dist, waypoint_angle, obstacle_dist, collision_flag,
                                    curvature_ahead, lateral_deviation, velocity_to_obstacle, steering_derivative]
        
        Returns:
            np.ndarray: Normalized state with values roughly in [0, 1] or [-1, 1] range
        """
        normalized = state.copy()
        
        # Expected state format (9 features):
        if len(normalized) >= 5:
            normalized[0] = np.clip(normalized[0] / 50.0, 0, 1)      # Speed: 0-50 m/s -> [0,1]
            normalized[1] = np.clip(normalized[1] / 100.0, 0, 1)     # Distance: 0-100m -> [0,1]
            normalized[2] = normalized[2] / np.pi                     # Angle: [-π, π] -> [-1,1]
            normalized[3] = np.clip(normalized[3] / 50.0, 0, 1)      # Obstacle: 0-50m -> [0,1]
            # collision_flag[4] already in [0, 1]
        
        # Enhanced features (already normalized by agent methods)
        if len(normalized) >= 9:
            # curvature_ahead[5] already in [0, 1]
            # lateral_deviation[6] already in [-2, 2], clip to [-1, 1]
            normalized[6] = np.clip(normalized[6] / 2.0, -1, 1)
            # velocity_to_obstacle[7] already in [0, 1]
            # steering_derivative[8] already in [0, 1]
        
        return normalized
    
    def _update_memory(self, steering: float, throttle: float, brake: float) -> None:
        """
        Update internal memory with latest control outputs using exponential decay.
        
        Stores the last N control commands (steering, throttle, brake) to provide
        temporal context for future decisions. This enables smoother driving by
        allowing the model to understand its recent control history.
        
        Memory structure:
        [steer_t-1, throttle_t-1, brake_t-1, steer_t-2, throttle_t-2, brake_t-2, ...]
        
        Uses exponential moving average for smooth memory updates:
        new_memory = (1-decay) * current + decay * old_memory
        
        Args:
            steering: Current steering command [-1, 1]
            throttle: Current throttle command [0, 1]
            brake: Current brake command [0, 1]
        """
        # Shift memory backward (oldest entries decay away)
        # Roll by 3 positions (one timestep = 3 controls)
        self.memory = np.roll(self.memory, 3)
        
        # Insert new control values at the front (most recent)
        self.memory[0] = steering
        self.memory[1] = throttle
        self.memory[2] = brake
        
        # Apply exponential decay to older memories
        # First 3 elements (current step) stay as-is
        # Older elements decay based on decay_factor
        for i in range(3, len(self.memory)):
            self.memory[i] *= self.decay_factor
    
    def predict(self, state: Union[List[float], np.ndarray]) -> Tuple[float, float, float]:
        """
        Predict driving actions using current state and temporal memory.
        
        Processing pipeline:
        1. Convert state to numpy array and normalize
        2. Create extended state (normalized current + memory)
        3. Split parameters for steering and acceleration
        4. Compute linear transformations
        5. Apply activation functions (tanh for steering, mutual exclusion for throttle/brake)
        6. Update memory for next step
        
        Args:
            state: Current vehicle state vector
                  Expected format (9 features): [speed, waypoint_dist, waypoint_angle, obstacle_dist, 
                                                collision_flag, curvature_ahead, lateral_deviation,
                                                velocity_to_obstacle, steering_derivative]
        
        Returns:
            Tuple[float, float, float]: Driving actions (steering, throttle, brake)
                - steering: Steering angle in [-1, 1] range (tanh activation)
                - throttle: Throttle amount in [0, 1] range (mutual exclusion with brake)
                - brake: Brake amount in [0, 1] range (mutual exclusion with throttle)
        """
        # Convert state to numpy array and normalize
        state_array = self._normalize_state(np.array(state))
        
        # Pad/truncate state to expected size
        if len(state_array) < self.input_size:
            # Pad with zeros if too short
            padded_state = np.zeros(self.input_size)
            padded_state[:len(state_array)] = state_array
            state_array = padded_state
        elif len(state_array) > self.input_size:
            # Truncate if too long
            state_array = state_array[:self.input_size]
        
        # Create extended state by combining current state with memory
        extended_state = np.concatenate([state_array, self.memory])
        
        # Split parameters into 2 parts: steering and acceleration
        extended_size = len(extended_state)
        steering_weights = self.parameters[:extended_size]
        accel_weights = self.parameters[extended_size:2*extended_size]
        
        # Compute raw linear outputs
        steer_raw = np.dot(steering_weights, extended_state)
        accel_raw = np.dot(accel_weights, extended_state)
        
        # Apply activation functions for natural bounds
        # Steering: tanh maps to [-1, 1]
        steering = np.tanh(steer_raw)
        
        # Acceleration: tanh maps to [-1, 1]
        # Positive = throttle, Negative = brake (mutual exclusion)
        accel = np.tanh(accel_raw)
        
        # Scale down acceleration to prevent extreme values (max 70% throttle/brake)
        accel = accel * 0.7
        
        if accel >= 0:
            # Positive acceleration = throttle
            throttle = float(accel)
            brake = 0.0
        else:
            # Negative acceleration = brake
            throttle = 0.0
            brake = float(-accel)
        
        # Update memory with actual control outputs (CRITICAL for temporal awareness)
        self._update_memory(float(steering), throttle, brake)
        
        return float(steering), throttle, brake
    
    def reset_memory(self) -> None:
        """
        Reset internal memory to zero state.
        
        Useful when starting new episodes or when vehicle position changes
        dramatically (e.g., respawn after collision).
        """
        self.memory.fill(0.0)
    
    def get_memory_info(self) -> dict:
        """
        Get current memory state for debugging and monitoring.
        
        Returns:
            dict: Memory information including current values and statistics
        """
        return {
            'memory_values': self.memory.tolist(),
            'memory_mean': float(np.mean(self.memory)),
            'memory_std': float(np.std(self.memory)),
            'memory_norm': float(np.linalg.norm(self.memory)),
            'decay_factor': self.decay_factor
        }
    
    def save(self, filepath: str) -> None:
        """
        Save model parameters to a file.
        
        Note: Only saves parameters, not memory state.
        Memory is reset when model is loaded.
        
        Args:
            filepath: Path where to save the parameters (.npy format)
        """
        try:
            np.save(filepath, self.parameters)
            print(f"SimpleTemporalModel parameters saved to {filepath}")
        except Exception as e:
            print(f"Error saving parameters: {e}")
            raise
    
    def load(self, filepath: str) -> None:
        """
        Load model parameters from a saved file.
        
        Args:
            filepath: Path to the saved parameter file (.npy format)
        
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            ValueError: If loaded parameters don't match expected dimensions
        """
        try:
            loaded_params = np.load(filepath)
            
            # Validate parameter dimensions
            if len(loaded_params) != self.param_size:
                raise ValueError(f"Parameter size mismatch: model expects {self.param_size}, "
                               f"file contains {len(loaded_params)}")
            
            self.parameters = loaded_params
            self.reset_memory()  # Reset memory when loading new parameters
            print(f"SimpleTemporalModel parameters loaded from {filepath}")
            
        except FileNotFoundError:
            print(f"Error: Parameter file not found: {filepath}")
            raise
        except Exception as e:
            print(f"Error loading parameters: {e}")
            raise
    
    def get_parameter_info(self) -> dict:
        """
        Get information about current model parameters.
        
        Returns:
            dict: Parameter statistics and model configuration
        """
        return {
            'param_count': len(self.parameters),
            'param_mean': float(np.mean(self.parameters)),
            'param_std': float(np.std(self.parameters)),
            'param_min': float(np.min(self.parameters)),
            'param_max': float(np.max(self.parameters)),
            'param_norm': float(np.linalg.norm(self.parameters)),
            'input_size': self.input_size,
            'memory_size': self.memory_size,
            'extended_size': self.input_size + self.memory_size,
            'decay_factor': self.decay_factor
        }
    
    def reset_parameters(self, seed: Optional[int] = None) -> None:
        """
        Reset parameters to random initial values using Xavier initialization.
        
        Args:
            seed: Random seed for reproducible initialization
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Use Xavier initialization for better learning dynamics
        memory_elements = self.memory_size * 3
        extended_size = self.input_size + memory_elements
        xavier_std = np.sqrt(2.0 / (extended_size + 2))  # 2 outputs
        self.parameters = np.random.randn(self.param_size) * xavier_std
        self.reset_memory()
        print(f"Parameters reset with {self.param_size} dimensions (Xavier std: {xavier_std:.4f})")