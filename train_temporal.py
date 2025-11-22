"""
Training Script for Simple Temporal Model with Natural Evolution Strategies

This script trains a temporal-aware autonomous driving agent using Natural Evolution 
Strategies (NES) optimization in the CARLA simulator. It uses the SimpleTemporalModel
which adds memory capabilities to the basic linear model without LSTM complexity.

Key Features:
- Memory-enhanced driving decisions
- Faster convergence than full LSTM
- Compatible with existing NES infrastructure
- Minimal parameter increase (15-30 parameters)
- Temporal awareness for better navigation
- Comprehensive logging system for training analysis

Author: Manuel Di Lullo
Date: 2025
Dependencies: CARLA simulator, numpy, custom temporal model, logging system
"""

import sys
import os
import time
import csv
import json
import pickle
import signal
import carla
import random
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional
from pathlib import Path

# Import temporal model and NES optimizer
from models.simple_temporal_model import SimpleTemporalModel
from agents.nes_agent import NES
from agents.custom_ml_agent import CustomAdvancedMLAgent

# Import utility functions
from utils.camera import center_camera_on_vehicle
from utils.actors_handler import spawn_vehicles_and_pedestrians, cleanup_actors
from utils.reward import calcola_reward
from utils.logger import get_logger, log_performance_info, configure_logging, log_prediction_debug
from utils.hud import create_hud

# Import shared training utilities (DRY principle)
from training.training_utils import (
    initialize_generation_log, log_generation, save_checkpoint, load_checkpoint,
    reset_vehicle_physics, cleanup_sensors_safe, deep_cleanup,
    create_training_directory, print_training_config, save_best_model, load_best_model
)


class TemporalMLAgent(CustomAdvancedMLAgent):
    """
    Extended ML agent that uses SimpleTemporalModel for memory-enhanced driving.
    
    This agent inherits all sensor capabilities from CustomAdvancedMLAgent
    but uses the temporal model instead of the basic linear model.
    """
    
    def __init__(self, vehicle: carla.Vehicle, model: SimpleTemporalModel):
        """
        Initialize temporal ML agent.
        
        Args:
            vehicle: CARLA vehicle instance to control
            model: SimpleTemporalModel instance for temporal decision making
        """
        # Initialize parent class with all sensor setup
        super().__init__(vehicle, model)
        
        # Cache for waypoints queue (used in curvature calculation)
        self.waypoints_queue = None
        
        print("TemporalMLAgent initialized with SimpleTemporalModel")
    
    def get_curvature_ahead(self, lookahead_distance: float = 15.0) -> float:
        """
        Calculate road curvature ahead using upcoming waypoints.
        
        Args:
            lookahead_distance: Distance to look ahead in meters
        
        Returns:
            float: Curvature value (0 = straight, 1 = sharp turn)
        """
        if self.route is None or len(self.route) < 3:
            return 0.0
        
        # Get next waypoints within lookahead distance
        waypoints_to_check = []
        cumulative_distance = 0
        vehicle_location = self.vehicle.get_location()
        
        for wp, _ in self.route[:min(10, len(self.route))]:
            if cumulative_distance >= lookahead_distance:
                break
            waypoints_to_check.append(wp)
            if len(waypoints_to_check) > 1:
                cumulative_distance += waypoints_to_check[-1].transform.location.distance(
                    waypoints_to_check[-2].transform.location
                )
        
        if len(waypoints_to_check) < 3:
            return 0.0
        
        # Calculate angle changes between consecutive waypoint pairs
        angles = []
        for i in range(len(waypoints_to_check) - 2):
            wp1 = waypoints_to_check[i]
            wp2 = waypoints_to_check[i + 1]
            wp3 = waypoints_to_check[i + 2]
            
            # Vector from wp1 to wp2
            v1 = np.array([
                wp2.transform.location.x - wp1.transform.location.x,
                wp2.transform.location.y - wp1.transform.location.y
            ])
            
            # Vector from wp2 to wp3
            v2 = np.array([
                wp3.transform.location.x - wp2.transform.location.x,
                wp3.transform.location.y - wp2.transform.location.y
            ])
            
            # Calculate angle between vectors
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norms > 1e-6:
                cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        if not angles:
            return 0.0
        
        # Average angle change = curvature
        # Normalize: 0 radians = straight (0), π/2 radians = 90° turn (1)
        avg_angle = np.mean(angles)
        curvature = min(avg_angle / (np.pi / 2), 1.0)
        
        return float(curvature)
    
    def get_lateral_deviation(self) -> float:
        """
        Calculate lateral distance from lane center.
        
        Returns:
            float: Lateral deviation normalized by lane width
        """
        if not hasattr(self, 'map') or self.vehicle is None:
            return 0.0
        
        vehicle_location = self.vehicle.get_location()
        
        # Get current waypoint on the road
        current_waypoint = self.map.get_waypoint(
            vehicle_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        
        if current_waypoint is None:
            return 0.0
        
        # Calculate lateral offset from waypoint (lane center)
        waypoint_location = current_waypoint.transform.location
        
        # Vector from waypoint to vehicle
        offset_vector = np.array([
            vehicle_location.x - waypoint_location.x,
            vehicle_location.y - waypoint_location.y
        ])
        
        # Waypoint forward direction
        waypoint_yaw = np.radians(current_waypoint.transform.rotation.yaw)
        forward_vector = np.array([
            np.cos(waypoint_yaw),
            np.sin(waypoint_yaw)
        ])
        
        # Right vector (perpendicular to forward)
        right_vector = np.array([
            -forward_vector[1],
            forward_vector[0]
        ])
        
        # Project offset onto right vector = lateral deviation
        lateral_offset = np.dot(offset_vector, right_vector)
        
        # Normalize by typical lane width (3.5m)
        lane_width = current_waypoint.lane_width if current_waypoint.lane_width > 0 else 3.5
        normalized_deviation = lateral_offset / (lane_width / 2)
        
        return float(np.clip(normalized_deviation, -2.0, 2.0))
    
    def get_velocity_towards_obstacle(self) -> float:
        """
        Calculate urgency of potential collision with nearest obstacle.
        
        Returns:
            float: Inverse time-to-collision (0 = safe, 1 = imminent)
        """
        obstacle_distance = self.get_front_obstacle_distance()
        
        if obstacle_distance is None or obstacle_distance > 40.0:
            return 0.0
        
        # Get vehicle velocity
        velocity_vector = self.vehicle.get_velocity()
        speed = np.sqrt(velocity_vector.x**2 + velocity_vector.y**2 + velocity_vector.z**2)
        
        if speed < 0.5:  # m/s (~1.8 km/h)
            return 0.0  # Nearly stopped, no collision risk
        
        # Time to collision = distance / speed
        ttc = obstacle_distance / speed
        
        # Map TTC to urgency: <1s = critical (1.0), >5s = safe (0.0)
        if ttc >= 5.0:
            urgency = 0.0
        elif ttc <= 1.0:
            urgency = 1.0
        else:
            urgency = 1.0 - (ttc - 1.0) / 4.0  # Linear mapping between 1s and 5s
        
        return float(urgency)
    
    def get_steering_derivative(self) -> float:
        """
        Calculate rate of steering change to detect oscillations.
        
        Returns:
            float: Steering change rate (0 = stable, 1 = oscillating)
        """
        if not hasattr(self, 'last_steer') or not hasattr(self, 'steer'):
            return 0.0
        
        # Steering change from last step
        steering_change = abs(self.steer - self.last_steer)
        
        # Normalize: max steering change per step is ~0.2 for smooth driving
        normalized_change = min(steering_change / 0.2, 1.0)
        
        return float(normalized_change)
    
    def get_enhanced_state(self) -> np.ndarray:
        """
        Get enhanced state with all 9 features for temporal model.
        
        Returns:
            np.ndarray: Feature vector [speed, dist_to_waypoint, angle_to_waypoint,
                                       obstacle_dist, collision, curvature_ahead,
                                       lateral_deviation, velocity_to_obstacle, steering_derivative]
        """
        # Get basic waypoint-aware state (5 features)
        basic_state = self.get_waypoint_aware_state()
        
        # Add enhanced features (4 new features)
        curvature = self.get_curvature_ahead()
        lateral_dev = self.get_lateral_deviation()
        velocity_to_obs = self.get_velocity_towards_obstacle()
        steering_deriv = self.get_steering_derivative()
        
        # Combine into 9-feature vector
        enhanced_state = np.array([
            basic_state[0],      # speed
            basic_state[1],      # distance to waypoint
            basic_state[2],      # angle to waypoint
            basic_state[3],      # obstacle distance
            basic_state[4],      # collision flag
            curvature,           # curvature ahead
            lateral_dev,         # lateral deviation
            velocity_to_obs,     # velocity towards obstacle
            steering_deriv       # steering derivative
        ])
        
        return enhanced_state
    
    def run_step(self) -> carla.VehicleControl:
        """
        Execute one control step using enhanced state features.
        
        Overrides parent method to use the enhanced 9-feature state vector
        instead of the basic 5-feature state.
        
        Returns:
            carla.VehicleControl: Applied vehicle control
        """
        # Get enhanced state (9 features)
        state = self.get_enhanced_state()
        
        # Store previous control values
        self.last_steer = self.steer if hasattr(self, 'steer') else 0.0
        self.last_throttle = self.throttle if hasattr(self, 'throttle') else 0.0
        self.last_brake = self.brake if hasattr(self, 'brake') else 0.0
        self.last_yaw = self.yaw if hasattr(self, 'yaw') else 0.0
        
        # Get control actions from temporal model
        self.steer, self.throttle, self.brake = self.model.predict(state)
        
        # Log predictions at DEBUG level
        log_prediction_debug(
            self.logger, 
            self.model.__class__.__name__, 
            f"Steer: {self.steer:.3f}, Throttle: {self.throttle:.3f}, Brake: {self.brake:.3f}"
        )
        
        # SPEED GOVERNOR: Limit max speed to prevent overspeeding
        # Target: ~40-50 km/h (11-14 m/s) for controlled, safe driving
        vehicle_velocity = self.vehicle.get_velocity()
        current_speed_ms = np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)
        MAX_SPEED_MS = 10.0  # ~50 km/h - reasonable urban speed
        
        # If exceeding max speed, override throttle to 0 and apply gentle brake
        if current_speed_ms > MAX_SPEED_MS:
            self.throttle = 0.0
            self.brake = 0.3  # Gentle braking
        # If approaching max speed, reduce throttle proportionally
        elif current_speed_ms > MAX_SPEED_MS * 0.85:  # Above 85% of max (11.9 m/s)
            speed_ratio = (current_speed_ms - MAX_SPEED_MS * 0.85) / (MAX_SPEED_MS * 0.15)
            self.throttle *= (1.0 - speed_ratio * 0.7)  # Reduce throttle up to 70%
        
        # Create and apply vehicle control
        control = carla.VehicleControl()
        control.steer = float(np.clip(self.steer, -0.5, 0.5))  # Cap steering to prevent tight circles
        control.throttle = float(np.clip(self.throttle, 0.0, 1.0))
        control.brake = float(np.clip(self.brake, 0.0, 1.0))
        control.hand_brake = False
        control.manual_gear_shift = False
        
        self.vehicle.apply_control(control)
        
        # Update yaw for tracking
        self.yaw = self.vehicle.get_transform().rotation.yaw
        
        return control
    
    def _setup_destination(self, generation: int = 0, max_generations: int = 300, 
                          visualize: bool = False, world: carla.World = None) -> None:
        """
        Setup destination for training episode.
        Fixed difficulty - no progressive scaling.
        
        Args:
            generation: Current training generation (unused - no progressive difficulty)
            max_generations: Total number of generations (unused)
            visualize: If True, draw the destination waypoint in the simulator
            world: CARLA world instance (required if visualize=True)
        """
        # Use current vehicle location as start
        current_location = self.vehicle.get_location()
        self.start = current_location
        
        # Fixed difficulty range (no progression)
        min_distance = 50   # Fixed minimum
        max_distance = 150  # Fixed maximum
        
        # Get current lane to ensure destination is on same traffic direction
        current_wp = self.map.get_waypoint(
            current_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        current_lane_sign = np.sign(current_wp.lane_id) if current_wp else 1
        
        # Filter spawn points to same side of road and within distance range
        spawn_points = self.map.get_spawn_points()
        valid_destinations = []
        
        for spawn_point in spawn_points:
            # CRITICAL: Skip spawn points too close to current position (avoid same spot)
            distance = current_location.distance(spawn_point.location)
            if distance < 10.0:  # Skip if within 10 meters of current position
                continue
            
            # Get waypoint at spawn point to check lane direction
            sp_wp = self.map.get_waypoint(
                spawn_point.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            
            if sp_wp is None:
                continue
            
            # Check if on same traffic direction (same lane sign)
            sp_lane_sign = np.sign(sp_wp.lane_id)
            if sp_lane_sign != current_lane_sign:
                continue  # Skip opposite-side destinations
            
            # Check if within target range
            if min_distance <= distance <= max_distance:
                valid_destinations.append((spawn_point.location, distance))
        
        # Find best destination from valid set (farthest within range)
        best_destination = None
        best_distance = 0
        
        if valid_destinations:
            best_destination, best_distance = max(valid_destinations, key=lambda x: x[1])
        else:
            # Fallback: find any reachable destination on same side (but not at spawn)
            fallback_destinations = []
            for spawn_point in spawn_points:
                distance = current_location.distance(spawn_point.location)
                if distance < 10.0:  # Skip if too close to current position
                    continue
                    
                sp_wp = self.map.get_waypoint(
                    spawn_point.location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )
                if sp_wp and np.sign(sp_wp.lane_id) == current_lane_sign:
                    fallback_destinations.append((spawn_point.location, distance))
            
            if fallback_destinations:
                # Use closest valid destination as fallback
                best_destination, best_distance = min(fallback_destinations, key=lambda x: x[1])
            else:
                # Last resort: use any spawn point (shouldn't happen on proper maps)
                distances = [current_location.distance(sp.location) for sp in spawn_points]
                farthest_idx = np.argmax(distances)
                best_destination = spawn_points[farthest_idx].location
                best_distance = distances[farthest_idx]
        
        self.destination = best_destination
        distance_to_dest = current_location.distance(self.destination)
        
        # Draw destination waypoint if visualization is enabled
        if visualize and world is not None:
            # Draw a red point at the destination location with SHORT life_time
            # so it disappears quickly if reached
            world.debug.draw_point(
                self.destination, 
                size=0.5, 
                color=carla.Color(r=255, g=0, b=0), 
                life_time=5.0  # Reduced from 30.0 - clears after 5 seconds
            )
        
        self.logger.info(f"🎯 Navigation setup (Gen {generation}): "
                        f"Target range {min_distance:.0f}-{max_distance:.0f}m, "
                        f"Actual: {distance_to_dest:.1f}m")
    
    def reset_for_new_episode(self, generation: int = 0, max_generations: int = 300,
                             visualize: bool = False, world: carla.World = None) -> None:
        """
        Reset agent state for a new training episode.
        
        Calls parent reset methods and also resets the temporal model's memory.
        CRITICAL: Ensures clean state between episodes to prevent bias.
        
        Args:
            generation: Current generation for progressive difficulty
            max_generations: Total generations for scaling
            visualize: If True, draw waypoints in the simulator
            world: CARLA world instance (required if visualize=True)
        """
        # CRITICAL: Reset temporal model memory first!
        # This prevents steering/acceleration bias from previous episode
        if hasattr(self, 'model') and hasattr(self.model, 'reset_memory'):
            self.model.reset_memory()
        
        # Reset sensor data and counters
        self.reset_sensors()
        
        # Update navigation for new location with progressive difficulty
        current_location = self.vehicle.get_location()
        self.start = current_location
        
        # Setup destination with generation-based difficulty
        self._setup_destination(generation, max_generations, visualize, world)
        
        # Setup route
        self._setup_route()
        
        # Reset control variables to neutral
        self.steer, self.throttle, self.brake, self.yaw = [0.0] * 4
        self.last_steer, self.last_throttle, self.last_brake, self.last_yaw = [0.0] * 4
        
        # Reset collision tracking
        self.collision_data = False
        self.collision_last_step = False
        
        # Reset reward tracking (for new destination)
        for attr in ['_prev_waypoint_dist', '_stuck_check_counter', '_stuck_start_pos']:
            if hasattr(self, attr):
                delattr(self, attr)


# reset_vehicle_physics moved to training/training_utils.py (DRY)


# cleanup_sensors_safe moved to training/training_utils.py (DRY)
# deep_cleanup moved to training/training_utils.py (DRY)



# save_best_model moved to training/training_utils.py (DRY)
# load_best_model moved to training/training_utils.py (DRY)


# initialize_generation_log moved to training/training_utils.py (DRY)
# log_generation moved to training/training_utils.py (DRY)


# save_checkpoint and load_checkpoint moved to training/training_utils.py (DRY)


def evaluate_individual(vehicle: carla.Vehicle, agent: TemporalMLAgent, 
                       world: carla.World, spawn_point: carla.Transform,
                       params: np.ndarray, max_steps: int, generation: int,
                       max_generations: int, training_config: dict,
                       hud=None, idx: int = 0) -> Tuple[float, bool]:
    """
    Evaluate a single individual by running a driving episode.
    
    Args:
        vehicle: CARLA vehicle instance
        agent: Agent controlling the vehicle
        world: CARLA world instance
        spawn_point: Spawn location for episode
        params: Neural network parameters for this individual
        max_steps: Maximum steps per episode
        generation: Current generation number
        max_generations: Total generations
        training_config: Training configuration dict
        hud: HUD instance for visualization (optional)
        idx: Individual index for logging
        
    Returns:
        Tuple of (total_reward, episode_completed)
    """
    # Set parameters and reset
    agent.model.set_parameters(params)
    agent.model.reset_memory()
    
    # Reset vehicle physics
    reset_vehicle_physics(vehicle, spawn_point, world)
    time.sleep(0.1)
    
    # Reset agent for new episode
    agent.reset_for_new_episode(
        generation=generation,
        max_generations=max_generations,
        visualize=training_config.get('visualize_waypoints', False),
        world=world
    )
    
    # Verify navigation setup
    if agent.get_next_global_waypoint_location() is None:
        print(f"  Warning: Navigation failed for individual {idx + 1}")
        return -500.0, False
    
    # Run episode
    total_reward = 0.0
    episode_complete = False
    
    for step in range(max_steps):
        try:
            # Execute control step
            agent.run_step()
            
            # Calculate reward
            reward = calcola_reward(vehicle, agent)
            total_reward += reward
            
            # Check destination
            if agent.destination is not None:
                current_loc = vehicle.get_location()
                dist_to_dest = current_loc.distance(agent.destination)
                
                if dist_to_dest < 10.0:
                    if not episode_complete:
                        # First completion - add bonus
                        total_reward += training_config['completion_bonus']
                        episode_complete = True
                        print(f"    Individual {idx + 1}: ✅ Completed at step {step}")
                        
                        # Setup new destination to continue
                        agent.reset_for_new_episode(
                            generation, max_generations,
                            training_config.get('visualize_waypoints', False), world
                        )
            
            # Update HUD every few steps
            if hud and step % 2 == 0:
                state = agent.get_enhanced_state()
                hud.draw(
                    vehicle=vehicle,
                    destination=agent.destination,
                    next_waypoint=agent.get_next_global_waypoint_location(),
                    generation=generation + 1,
                    episode_step=step,
                    fitness=total_reward,
                    additional_info={
                        'Curvature': state[5] if len(state) > 5 else 0.0,
                        'Lateral Dev': state[6] if len(state) > 6 else 0.0,
                        'Reward': reward
                    }
                )
            
            # Reset collision flag
            agent.collision_last_step = False
            
            # Tick simulation
            world.tick()
            
        except Exception as e:
            print(f"  Error in step {step}: {e}")
            total_reward -= 50
            break
    
    return total_reward, episode_complete


def train_generation(nes, vehicle: carla.Vehicle, agent: TemporalMLAgent,
                     world: carla.World, spawn_points: List[carla.Transform],
                     generation: int, training_config: dict, hud=None) -> Tuple[List[float], int]:
    """
    Train a single generation by evaluating all population members.
    
    Args:
        nes: NES optimizer instance
        vehicle: Training vehicle
        agent: Training agent
        world: CARLA world
        spawn_points: Available spawn locations
        generation: Current generation number
        training_config: Training configuration
        hud: HUD instance (optional)
        
    Returns:
        Tuple of (fitness_list, completion_count)
    """
    fitnesses = []
    completions_count = 0
    
    for idx, individual in enumerate(nes.population):
        # Small delay between individuals
        if idx > 0:
            time.sleep(0.1)
        
        # Random spawn point
        spawn_point = random.choice(spawn_points)
        
        # Evaluate individual
        fitness, completed = evaluate_individual(
            vehicle=vehicle,
            agent=agent,
            world=world,
            spawn_point=spawn_point,
            params=individual,
            max_steps=training_config['max_steps'],
            generation=generation,
            max_generations=training_config['num_generations'],
            training_config=training_config,
            hud=hud,
            idx=idx
        )
        
        fitnesses.append(fitness)
        if completed:
            completions_count += 1
        
        # Print progress every 5 individuals
        if idx == 0 or (idx + 1) % 5 == 0:
            status = "✅" if completed else "⏸️"
            print(f"  Individual {idx + 1:2d}: {fitness:7.1f} {status}")
    
    return fitnesses, completions_count


def main():
    """
    Main training function for SimpleTemporalModel with NES optimization.
    
    This function:
    1. Sets up CARLA environment and spawns training vehicle
    2. Initializes SimpleTemporalModel and TemporalMLAgent
    3. Creates NES optimizer for evolutionary training
    4. Runs training loop with fitness evaluation
    5. Saves best models and training progress
    6. Supports resuming from checkpoints
    """
    print("=" * 60)
    print("Simple Temporal Model Training with NES")
    print("=" * 60)
    
    # Parse command line arguments for resume
    import argparse
    parser = argparse.ArgumentParser(description='Train Temporal Model with NES')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file to resume training')
    args = parser.parse_args()
    
    # Initialize CARLA client and connect to simulator
    try:
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        cleanup_actors(world)
        print("✓ Connected to CARLA simulator")
    except Exception as e:
        print(f"✗ Failed to connect to CARLA: {e}")
        print("Make sure CARLA simulator is running on localhost:2000")
        return

    # Get vehicle blueprint and spawn main training vehicle
    try:
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print("✓ Training vehicle spawned")
        
        # CRITICAL: Give CARLA time to render and initialize the vehicle
        for _ in range(10):
            world.tick()
        time.sleep(0.5)
        
    except Exception as e:
        print(f"✗ Failed to spawn vehicle: {e}")
        cleanup_actors(world)
        return

    # Spawn minimal NPC traffic (optional - can be disabled for faster training)
    try:
        vehicles, pedestrians = spawn_vehicles_and_pedestrians(world, num_vehicles=2, num_pedestrians=0)
        print(f"✓ Spawned {len(vehicles)} NPC vehicles")
    except Exception as e:
        print(f"Warning: Failed to spawn NPCs: {e}")
        vehicles, pedestrians = [], []

    # Initialize SimpleTemporalModel
    try:
        # Model configuration - optimized for NES learning
        model_config = {
            'input_size': 9,        # Enhanced features
            'memory_size': 5,       # Reduced: 5 timesteps (was 10) - less parameters, faster learning
            'decay_factor': 0.85    # Increased: longer memory (was 0.7) - smoother driving
        }
        
        temporal_model = SimpleTemporalModel(**model_config)
        print("✓ SimpleTemporalModel initialized")
        print(f"  Input features: {temporal_model.input_size}")
        print(f"  Memory timesteps: {temporal_model.memory_size}")
        print(f"  Total memory elements: {temporal_model.memory_size * 3}")
        print(f"  Total parameters: {temporal_model.param_size}")
        print(f"  Decay factor: {temporal_model.decay_factor}")
        
    except Exception as e:
        print(f"✗ Failed to initialize temporal model: {e}")
        cleanup_actors(world)
        return

    # Initialize TemporalMLAgent
    try:
        agent = TemporalMLAgent(vehicle, temporal_model)
        print("✓ TemporalMLAgent initialized with sensors")
        
        # CRITICAL: Give sensors time to initialize and attach to vehicle
        for _ in range(10):
            world.tick()
        time.sleep(0.3)
        
    except Exception as e:
        print(f"✗ Failed to initialize agent: {e}")
        cleanup_actors(world)
        return

    # Initialize HUD for visualization
    try:
        hud = create_hud(world)
        print("✓ Training HUD initialized (separate console window)")
    except Exception as e:
        print(f"Warning: HUD initialization failed: {e}")
        hud = None

    # Initialize NES optimizer
    try:
        # NES configuration - optimized for faster convergence
        nes_config = {
            'population_size': 12,   # Reduced from 15 - faster iterations
            'sigma': 0.15,           # Increased from 0.12 - more exploration
            'learning_rate': 0.08    # Increased from 0.05 - faster learning
        }
        
        nes = NES(agent=agent, **nes_config)
        print("✓ NES optimizer initialized")
        print(f"  Population size: {nes_config['population_size']}")
        print(f"  Sigma (exploration): {nes_config['sigma']}")
        print(f"  Learning rate: {nes_config['learning_rate']}")
        
    except Exception as e:
        print(f"✗ Failed to initialize NES: {e}")
        cleanup_actors(world)
        return

    # Training hyperparameters
    training_config = {
        'num_generations': 300,          # Total training generations
        'max_steps': 600,                # Increased: more time to reach destination
        'min_route_distance': 50,        # Start distance (no progressive difficulty)
        'max_route_distance': 150,       # End distance (no progressive difficulty)
        'completion_bonus': 500.0,       # Big reward for reaching destination
        'save_interval': 10,             # Save every 10 generations (was 5)
        'print_interval': 1,             # Print every generation for monitoring
        'visualize_waypoints': False     # Disable for speed (was True)
    }
    
    print("\n" + "=" * 60)
    print("Training Configuration:")
    print(f"  Generations: {training_config['num_generations']}")
    print(f"  Steps per episode: {training_config['max_steps']}")
    print(f"  Route distance: {training_config['min_route_distance']}-{training_config['max_route_distance']}m")
    print(f"  Completion bonus: {training_config['completion_bonus']}")
    print(f"  Save interval: {training_config['save_interval']} generations")
    print("=" * 60 + "\n")

    # Create training directory or use existing one for resume
    if args.resume:
        # Resume from checkpoint - use existing directory
        checkpoint_dir = os.path.dirname(os.path.dirname(args.resume))  # Go up from checkpoints/
        save_dir = checkpoint_dir
        log_path = os.path.join(save_dir, 'training_log.csv')
        print(f"Resuming training from: {save_dir}")
    else:
        # New training - create new directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"training/temporal_model_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize generation log (CSV file)
        log_path = initialize_generation_log(save_dir, training_config, nes_config)
    
    # Training statistics tracking
    fitness_history = []
    best_fitness = float('-inf')
    best_generation = 0
    start_generation = 0
    
    # CRITICAL: Load checkpoint AFTER creating vehicle/agent/NES
    # This is because checkpoint only contains parameters, not physical objects
    if args.resume:
        try:
            print("\n" + "=" * 60)
            print("Loading checkpoint...")
            
            start_generation, best_fitness, fitness_history = load_checkpoint(
                args.resume, temporal_model, nes
            )
            
            # CRITICAL: Update agent's model reference after loading parameters
            # This ensures the agent uses the loaded parameters
            agent.model = temporal_model
            nes.agent = agent
            
            print(f"✓ Resuming from generation {start_generation}")
            print(f"  Best fitness so far: {best_fitness:.2f}")
            print(f"  Continuing to generation {training_config['num_generations']}")
            print("=" * 60 + "\n")
            
            # CRITICAL: Extra world ticks to stabilize after loading checkpoint
            print("Stabilizing simulation after checkpoint load...")
            for _ in range(20):
                world.tick()
            time.sleep(1.0)
            print("✓ Simulation stabilized\n")
            
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            print("\nStarting fresh training instead...")
            start_generation = 0
            best_fitness = float('-inf')
            fitness_history = []

    try:
        # Main evolutionary training loop
        for generation in range(start_generation, training_config['num_generations']):
            generation_start_time = time.time()
            
            print(f"\n=== Generation {generation + 1}/{training_config['num_generations']} ===")
            
            # Periodic deep cleanup every 10 generations to prevent resource leaks
            if generation > 0 and generation % 10 == 0:
                deep_cleanup(world, vehicle)
            
            # Evaluate generation using refactored function
            fitnesses, completions_count = train_generation(
                nes=nes,
                vehicle=vehicle,
                agent=agent,
                world=world,
                spawn_points=spawn_points,
                generation=generation,
                training_config=training_config,
                hud=hud
            )
            
            
            # Update NES with fitness results
            nes_stats = nes.step(fitnesses)
            fitness_history.append(fitnesses)
            
            # Track best performance
            max_fitness = np.max(fitnesses)
            mean_fitness = np.mean(fitnesses)
            generation_time = time.time() - generation_start_time
            
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_generation = generation
                
                # Save best model
                save_best_model(
                    temporal_model, generation, best_fitness, save_dir,
                    {**model_config, **training_config, **nes_config}
                )
            
            # Log generation
            log_generation(
                log_path, generation, fitnesses, best_fitness,
                best_generation, generation_time, training_config,
                completions_count, nes_config['population_size']
            )
            
            # Print summary
            print(f"\nGeneration {generation + 1} Summary:")
            print(f"  Mean: {mean_fitness:7.1f} | Max: {max_fitness:7.1f} | "
                  f"Best: {best_fitness:7.1f} (gen {best_generation + 1})")
            print(f"  Completions: {completions_count}/{nes_config['population_size']} "
                  f"({completions_count/nes_config['population_size']*100:.0f}%)")
            print(f"  Time: {generation_time:.1f}s")
            
            # Periodic checkpoint
            if (generation + 1) % training_config['save_interval'] == 0:
                save_checkpoint(temporal_model, nes, generation,
                              best_fitness, fitness_history, save_dir)


        # Training completed
        print("\n" + "=" * 60)
        print("Training Completed!")
        print(f"Best fitness: {best_fitness:.1f} at generation {best_generation + 1}")
        print(f"Final mean fitness: {np.mean(fitnesses):.1f}")
        print(f"Training log: {log_path}")
        print(f"Best model: {os.path.join(save_dir, 'best_models', 'best_model.npy')}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Training interrupted by user (Ctrl+C)")
        print("=" * 60)
        
        # Save interruption checkpoint
        print("\nSaving checkpoint...")
        save_checkpoint(
            temporal_model,
            nes,
            generation,
            best_fitness,
            fitness_history,
            save_dir
        )
        print(f"✓ Checkpoint saved to: {save_dir}/checkpoints/")
        print(f"✓ Training log preserved: {log_path}")
        
    except Exception as e:
        print(f"\nUnexpected error during training: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Perform deep cleanup of all CARLA resources
        print("\n" + "=" * 60)
        print("Performing deep cleanup...")
        print("=" * 60)
        try:
            # Destroy HUD
            if 'hud' in locals() and hud:
                try:
                    hud.destroy()
                    print("✓ HUD destroyed")
                except Exception as e:
                    print(f"  Warning: HUD cleanup: {e}")
            
            # Destroy agent (includes sensors)
            if 'agent' in locals() and agent:
                try:
                    agent.destroy()
                    print("✓ Agent destroyed")
                except Exception as e:
                    print(f"  Warning: Agent cleanup: {e}")
            
            # Destroy vehicle
            if 'vehicle' in locals() and vehicle:
                try:
                    if vehicle.is_alive:
                        vehicle.destroy()
                    print("✓ Vehicle destroyed")
                except Exception as e:
                    print(f"  Warning: Vehicle cleanup: {e}")
            
            # Deep cleanup: remove all actors and orphaned sensors
            if 'world' in locals() and world:
                try:
                    print("\nPerforming deep cleanup of world actors...")
                    deep_cleanup(world, active_vehicle=None)
                    print("✓ Deep cleanup completed")
                except Exception as e:
                    print(f"  Warning: Deep cleanup: {e}")
            
            print("\n" + "=" * 60)
            print("Cleanup completed successfully!")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n✗ Cleanup error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    """
    Entry point for temporal model training with checkpoint resume support.
    
    Usage:
        # Start new training:
        python train_temporal.py
        
        # Resume from checkpoint:
        python train_temporal.py --resume training/temporal_model_20251108_180525/checkpoints/checkpoint_gen_50.pkl
        
    Make sure CARLA simulator is running before starting training.
    """
    main()