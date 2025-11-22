"""
Training Utilities - Shared Functions for NES Training

Provides common training functions used across different training scripts.
Follows DRY principle to eliminate code duplication.

Author: Manuel Di Lullo
Date: 2025
"""

import os
import csv
import json
import pickle
import time
import carla
import numpy as np
from datetime import datetime
from typing import List, Tuple, Any
from pathlib import Path


def initialize_generation_log(save_dir: str, training_config: dict, nes_config: dict) -> str:
    """
    Initialize CSV log file for generation statistics.
    
    Args:
        save_dir: Directory to save log
        training_config: Training configuration dict
        nes_config: NES optimizer configuration
        
    Returns:
        Path to CSV log file
    """
    log_path = os.path.join(save_dir, 'training_log.csv')
    
    headers = [
        'generation', 'mean_fitness', 'max_fitness', 'min_fitness', 'std_fitness',
        'best_fitness_overall', 'best_generation', 'target_min_distance',
        'target_max_distance', 'completions', 'completion_rate',
        'generation_time_s', 'timestamp'
    ]
    
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(headers)
    
    config_path = os.path.join(save_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump({'training': training_config, 'nes': nes_config}, f, indent=2)
    
    return log_path


def log_generation(log_path: str, generation: int, fitnesses: List[float],
                  best_fitness: float, best_generation: int, generation_time: float,
                  training_config: dict, completions: int = 0, population_size: int = 12) -> None:
    """Append generation statistics to CSV log."""
    
    progress_ratio = min((generation + 1) / training_config.get('num_generations', 300), 1.0)
    target_min = training_config.get('min_route_distance', 50)
    target_max = training_config.get('max_route_distance', 150)
    
    row = [
        generation + 1,
        f"{np.mean(fitnesses):.2f}",
        f"{np.max(fitnesses):.2f}",
        f"{np.min(fitnesses):.2f}",
        f"{np.std(fitnesses):.2f}",
        f"{best_fitness:.2f}",
        best_generation + 1,
        f"{target_min:.0f}",
        f"{target_max:.0f}",
        completions,
        f"{(completions / population_size * 100):.1f}",
        f"{generation_time:.1f}",
        datetime.now().isoformat()
    ]
    
    with open(log_path, 'a', newline='') as f:
        csv.writer(f).writerow(row)


def save_best_model(model: Any, generation: int, fitness: float, 
                    save_dir: str, config: dict) -> str:
    """Save best model with metadata."""
    
    best_dir = os.path.join(save_dir, 'best_models')
    os.makedirs(best_dir, exist_ok=True)
    
    model_path = os.path.join(best_dir, 'best_model.npy')
    model.save(model_path)
    
    metadata = {
        'generation': generation,
        'fitness': float(fitness),
        'timestamp': datetime.now().isoformat(),
        'model_config': {
            'input_size': model.input_size,
            'memory_size': model.memory_size,
            'decay_factor': model.decay_factor,
            'param_size': model.param_size
        },
        'training_config': config
    }
    
    with open(os.path.join(best_dir, 'best_model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model_path


def load_best_model(save_dir: str, model_class: Any) -> Tuple[Any, dict]:
    """Load best model with metadata."""
    
    best_dir = os.path.join(save_dir, 'best_models')
    
    with open(os.path.join(best_dir, 'best_model_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    model_config = metadata['model_config']
    model = model_class(
        input_size=model_config['input_size'],
        memory_size=model_config['memory_size'],
        decay_factor=model_config['decay_factor']
    )
    
    model.load(os.path.join(best_dir, 'best_model.npy'))
    return model, metadata


def save_checkpoint(model: Any, nes: Any, generation: int,
                   best_fitness: float, fitness_history: List,
                   save_dir: str) -> None:
    """Save complete training checkpoint."""
    
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'generation': generation,
        'best_fitness': best_fitness,
        'fitness_history': fitness_history,
        'model_parameters': model.parameters.copy(),
        'nes_state': {
            'population': nes.population.copy(),
            'mean': nes.agent.model.parameters.copy(),
            'sigma': nes.sigma,
            'learning_rate': nes.learning_rate
        }
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_gen_{generation}.pkl')
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(checkpoint_path: str, model: Any, nes: Any) -> Tuple[int, float, List]:
    """Load training checkpoint to resume."""
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    model.set_parameters(checkpoint['model_parameters'])
    nes.population = checkpoint['nes_state']['population'].copy()
    nes.agent.model.set_parameters(checkpoint['nes_state']['mean'])
    nes.sigma = checkpoint['nes_state']['sigma']
    nes.learning_rate = checkpoint['nes_state']['learning_rate']
    
    return checkpoint['generation'] + 1, checkpoint['best_fitness'], checkpoint['fitness_history']


def reset_vehicle_physics(vehicle: carla.Vehicle, spawn_point: carla.Transform, 
                          world: carla.World) -> None:
    """Reset vehicle physics and position."""
    
    for attempt in range(3):
        try:
            vehicle.set_transform(spawn_point)
            break
        except RuntimeError:
            world.tick()
            time.sleep(0.1)
    
    vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
    
    control = carla.VehicleControl()
    control.throttle = 0.0
    control.steer = 0.0
    control.brake = 1.0
    vehicle.apply_control(control)
    
    for _ in range(5):
        world.tick()
    
    control.brake = 0.0
    vehicle.apply_control(control)
    world.tick()


def cleanup_sensors_safe(world: carla.World, active_vehicle_ids: set = None) -> None:
    """Safely clean up orphaned sensors."""
    
    if active_vehicle_ids is None:
        active_vehicle_ids = set()
    
    try:
        sensors = world.get_actors().filter('sensor.*')
        cleaned_count = 0
        
        for sensor in sensors:
            try:
                parent = sensor.parent
                
                if parent is None or not parent.is_alive:
                    sensor.stop()
                    sensor.destroy()
                    cleaned_count += 1
                    continue
                
                if active_vehicle_ids and parent.id not in active_vehicle_ids:
                    sensor.stop()
                    sensor.destroy()
                    cleaned_count += 1
                    
            except RuntimeError:
                pass
        
        if cleaned_count > 0:
            world.tick()
            
    except Exception as e:
        print(f"Warning: Sensor cleanup error: {e}")


def deep_cleanup(world: carla.World, active_vehicle: carla.Vehicle = None) -> None:
    """Perform periodic deep cleanup while preserving active vehicle."""
    
    active_ids = set()
    if active_vehicle is not None and active_vehicle.is_alive:
        active_ids.add(active_vehicle.id)
    
    cleanup_sensors_safe(world, active_ids)
    
    from utils.actors_handler import cleanup_actors
    cleanup_actors(world)
    
    for _ in range(5):
        world.tick()
        time.sleep(0.02)


def create_training_directory(prefix: str = "temporal_model") -> str:
    """Create timestamped training directory."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"training/{prefix}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def print_training_config(training_config: dict, nes_config: dict, model_config: dict) -> None:
    """Print training configuration summary."""
    
    print("\n" + "=" * 60)
    print("Training Configuration:")
    print(f"  Generations: {training_config['num_generations']}")
    print(f"  Steps per episode: {training_config['max_steps']}")
    print(f"  Route distance: {training_config['min_route_distance']}-{training_config['max_route_distance']}m")
    print(f"  Completion bonus: {training_config.get('completion_bonus', 500)}")
    print("\nNES Configuration:")
    print(f"  Population size: {nes_config['population_size']}")
    print(f"  Sigma (exploration): {nes_config['sigma']}")
    print(f"  Learning rate: {nes_config['learning_rate']}")
    print("\nModel Configuration:")
    print(f"  Input features: {model_config['input_size']}")
    print(f"  Memory timesteps: {model_config['memory_size']}")
    print(f"  Decay factor: {model_config['decay_factor']}")
    print("=" * 60 + "\n")
