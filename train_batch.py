"""
Batch Training Script - Parallel Vehicle Evaluation

This script demonstrates how to use the batch trainer to evaluate
multiple vehicles in parallel, significantly reducing training time.

Usage:
    python train_batch.py [--batch-size 4]

Author: Manuel Di Lullo
Date: 2025
"""

import sys
import carla
import random
import argparse
from datetime import datetime

from models.simple_temporal_model import SimpleTemporalModel
from agents.nes_agent import NES
from train_temporal import TemporalMLAgent
from training.batch_trainer import BatchVehicleTrainer
from training.training_utils import (
    initialize_generation_log, log_generation,
    save_best_model, save_checkpoint, load_checkpoint
)
from utils.actors_handler import cleanup_actors
from utils.hud import create_hud


def main():
    """Main batch training loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Number of vehicles to evaluate in parallel')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file to resume from (e.g., training/batch_temporal_XXX/checkpoints/checkpoint_gen_50.pkl)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Batch Training with Parallel Vehicle Evaluation")
    print(f"Batch Size: {args.batch_size} vehicles in parallel")
    if args.resume:
        print(f"Resume Mode: Loading from {args.resume}")
    print("=" * 60)
    print("\n💡 Tip: To resume from a checkpoint, use:")
    print("   python train_batch.py --resume <path_to_checkpoint.pkl>")
    print("   Example: python train_batch.py --resume training/batch_temporal_XXX/checkpoints/checkpoint_gen_50.pkl\n")
    
    # Connect to CARLA
    try:
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        # CRITICAL: Ensure synchronous mode is enabled for proper physics simulation
        settings = world.get_settings()
        if not settings.synchronous_mode:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            world.apply_settings(settings)
            print("✓ Enabled synchronous mode (0.05s timestep)")
        else:
            print(f"✓ Synchronous mode already enabled (timestep: {settings.fixed_delta_seconds}s)")
        
        cleanup_actors(world)
        print("✓ Connected to CARLA simulator")
    except Exception as e:
        print(f"✗ Failed to connect to CARLA: {e}")
        return
    
    # Initialize batch trainer
    model_config = {
        'input_size': 9,
        'memory_size': 5,
        'decay_factor': 0.85
    }
    
    # Initialize HUD first
    try:
        hud = create_hud(world)
        print("✓ Training HUD initialized")
    except Exception as e:
        print(f"Warning: HUD initialization failed: {e}")
        hud = None
    
    batch_trainer = BatchVehicleTrainer(
        world=world,
        agent_class=TemporalMLAgent,
        model_class=SimpleTemporalModel,
        batch_size=args.batch_size,
        enable_camera_tracking=True,
        hud=hud
    )
    
    print("\n" + "=" * 60)
    print("Batch Trainer Configuration:")
    print(f"  Batch size: {args.batch_size} vehicles in parallel")
    print(f"  Camera tracking: Enabled (follows best performer)")
    print(f"  HUD Display: {'Enabled' if hud else 'Disabled'}")
    print(f"  Model: SimpleTemporalModel")
    print(f"    - Input features: {model_config['input_size']}")
    print(f"    - Memory timesteps: {model_config['memory_size']}")
    print(f"    - Decay factor: {model_config['decay_factor']}")
    print("=" * 60 + "\n")
    
    # Create a single vehicle for NES initialization
    # (NES needs an agent reference, but we'll use batch trainer for evaluation)
    try:
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        reference_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        
        for _ in range(10):
            world.tick()
        
        # Create reference model and agent for NES
        reference_model = SimpleTemporalModel(**model_config)
        reference_agent = TemporalMLAgent(reference_vehicle, reference_model)
        
        for _ in range(10):
            world.tick()
        
        print("✓ Reference agent created for NES")
        
    except Exception as e:
        print(f"✗ Failed to create reference agent: {e}")
        cleanup_actors(world)
        return
    
    # Initialize NES with balanced exploration
    nes_config = {
        'population_size': 12,
        'sigma': 0.3,  # Balanced exploration - not too high to avoid extreme values
        'learning_rate': 0.05
    }
    
    nes = NES(agent=reference_agent, **nes_config)
    print(f"✓ NES initialized (population={nes_config['population_size']})")
    
    # Load checkpoint if resuming
    start_generation = 0
    if args.resume:
        try:
            print(f"\n📂 Loading checkpoint from: {args.resume}")
            start_generation, best_fitness, fitness_history = load_checkpoint(
                args.resume, reference_model, nes
            )
            print(f"✓ Resumed from generation {start_generation}")
            print(f"  Best fitness so far: {best_fitness:.2f}")
            print(f"  Loaded {len(fitness_history)} generations of history")
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
            print("  Starting from scratch instead...")
            best_fitness = float('-inf')
            fitness_history = []
    else:
        best_fitness = float('-inf')
        fitness_history = []
    
    # Training config
    training_config = {
        'num_generations': 300,
        'max_steps': 600,
        'min_route_distance': 50,
        'max_route_distance': 150,
        'completion_bonus': 500.0,
        'save_interval': 10,
        'print_interval': 1
    }
    
    # Create training directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"training/batch_temporal_{timestamp}"
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    log_path = initialize_generation_log(save_dir, training_config, nes_config)
    
    # Training loop
    best_generation = start_generation
    
    try:
        for generation in range(start_generation, training_config['num_generations']):
            import time
            gen_start = time.time()
            
            print(f"\n=== Generation {generation + 1}/{training_config['num_generations']} ===")
            
            # Evaluate population using batch trainer
            fitnesses = batch_trainer.evaluate_batch(
                population=nes.population,
                max_steps=training_config['max_steps'],
                generation=generation,
                max_generations=training_config['num_generations']
            )
            
            # Update NES
            nes.step(fitnesses)
            fitness_history.append(fitnesses)
            
            # Track best
            max_fitness = max(fitnesses)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_generation = generation
                save_best_model(reference_model, generation, best_fitness,
                              save_dir, {**model_config, **training_config})
            
            # Log
            gen_time = time.time() - gen_start
            log_generation(log_path, generation, fitnesses, best_fitness,
                         best_generation, gen_time, training_config, 0,
                         nes_config['population_size'])
            
            # Print summary
            import numpy as np
            mean_fit = np.mean(fitnesses)
            std_fit = np.std(fitnesses)
            min_fit = np.min(fitnesses)
            
            print(f"\n  📄 Generation {generation + 1} Summary:")
            print(f"    Mean: {mean_fit:7.1f} ± {std_fit:6.1f}")
            print(f"    Range: [{min_fit:7.1f}, {max_fitness:7.1f}]")
            print(f"    Best Overall: {best_fitness:7.1f} (gen {best_generation + 1})")
            print(f"    Time: {gen_time:.1f}s")
            print("    " + "-" * 50)
            
            # Periodic checkpoint
            if (generation + 1) % training_config['save_interval'] == 0:
                save_checkpoint(reference_model, nes, generation,
                              best_fitness, fitness_history, save_dir)
                checkpoint_path = f"{save_dir}/checkpoints/checkpoint_gen_{generation}.pkl"
                print(f"    💾 Checkpoint saved: {checkpoint_path}")
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best fitness: {best_fitness:.1f} at generation {best_generation + 1}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Training interrupted by user (Ctrl+C)")
        print("=" * 60)
        print("\nSaving checkpoint...")
        save_checkpoint(reference_model, nes, generation,
                       best_fitness, fitness_history, save_dir)
        print("✓ Checkpoint saved")
        
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
            
            # Cleanup batch trainer vehicles
            if 'batch_trainer' in locals():
                try:
                    batch_trainer.cleanup_batch()
                    print("✓ Batch vehicles destroyed")
                except Exception as e:
                    print(f"  Warning: Batch cleanup: {e}")
            
            # Destroy reference agent
            if 'reference_agent' in locals() and reference_agent:
                try:
                    reference_agent.destroy()
                    print("✓ Reference agent destroyed")
                except Exception as e:
                    print(f"  Warning: Reference agent cleanup: {e}")
            
            # Destroy reference vehicle
            if 'reference_vehicle' in locals() and reference_vehicle:
                try:
                    if reference_vehicle.is_alive:
                        reference_vehicle.destroy()
                    print("✓ Reference vehicle destroyed")
                except Exception as e:
                    print(f"  Warning: Reference vehicle cleanup: {e}")
            
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
    main()
