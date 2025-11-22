"""
Unified Training Script for NES-CARLA

Single entry point for all training modes: batch, nes, temporal.
Simplifies training workflow and reduces code duplication.

Usage:
    # Standard temporal training
    python train.py --type temporal
    
    # Batch training (parallel evaluation)
    python train.py --type batch --batch-size 4
    
    # Basic linear model training
    python train.py --type nes
    
    # Use configuration preset
    python train.py --type temporal --preset quick_test
    
    # Resume from checkpoint
    python train.py --type temporal --resume training/temporal_model_XXX/checkpoints/checkpoint_gen_50.pkl
    
    # Custom configuration
    python train.py --type temporal --generations 500 --population 20

Author: Manuel Di Lullo
Date: 2025
"""

import argparse
import signal
import sys
import time
import random
import numpy as np
from datetime import datetime
from pathlib import Path

from config import Config, get_config
from training.training_engine import TrainingEngine
from training.training_utils import (
    initialize_generation_log, log_generation, save_checkpoint,
    load_checkpoint, save_best_model, reset_vehicle_physics
)
from utils.reward import calcola_reward


def train_sequential(engine: TrainingEngine, config: Config, save_dir: str, 
                     start_generation: int = 0, best_fitness: float = float('-inf'),
                     fitness_history: list = None) -> None:
    """
    Sequential training: evaluate population one vehicle at a time.
    
    Args:
        engine: TrainingEngine instance
        config: Training configuration
        save_dir: Directory to save results
        start_generation: Generation to start from (for resume)
        best_fitness: Best fitness so far
        fitness_history: History of fitness values
    """
    if fitness_history is None:
        fitness_history = []
    
    spawn_points = engine.world.get_map().get_spawn_points()
    log_path = initialize_generation_log(save_dir, config.training.to_dict(), config.nes.to_dict())
    
    try:
        for generation in range(start_generation, config.training.num_generations):
            gen_start = time.time()
            print(f"\n=== Generation {generation + 1}/{config.training.num_generations} ===")
            
            fitnesses = []
            completions_count = 0
            
            # Evaluate each individual
            for idx, individual in enumerate(engine.nes.population):
                engine.agent.model.set_parameters(individual)
                engine.agent.model.reset_memory()
                
                # Reset vehicle
                spawn_point = random.choice(spawn_points)
                reset_vehicle_physics(engine.vehicle, spawn_point, engine.world)
                time.sleep(0.1)
                
                # Reset agent for new episode
                engine.agent.reset_for_new_episode(
                    generation=generation,
                    max_generations=config.training.num_generations,
                    visualize=config.training.visualize_waypoints,
                    world=engine.world
                )
                
                # DEBUG: Print initial position and waypoints
                vehicle_loc = engine.vehicle.get_location()
                next_wp = engine.agent.get_next_global_waypoint_location()
                dest = engine.agent.destination
                print(f"    🚗 Individual {idx + 1} START:")
                print(f"       Vehicle pos: ({vehicle_loc.x:.1f}, {vehicle_loc.y:.1f}, {vehicle_loc.z:.1f})")
                if next_wp:
                    print(f"       Next waypoint: ({next_wp.x:.1f}, {next_wp.y:.1f}, {next_wp.z:.1f})")
                    dist_to_wp = vehicle_loc.distance(next_wp)
                    print(f"       Distance to waypoint: {dist_to_wp:.1f}m")
                if dest:
                    print(f"       Destination: ({dest.x:.1f}, {dest.y:.1f}, {dest.z:.1f})")
                    dist_to_dest = vehicle_loc.distance(dest)
                    print(f"       Distance to destination: {dist_to_dest:.1f}m")
                
                # Run episode
                total_reward = 0.0
                episode_complete = False
                last_waypoint_count = len(engine.agent.route) if hasattr(engine.agent, 'route') and engine.agent.route else 0
                
                for step in range(config.training.max_steps):
                    try:
                        # Check if waypoint was reached (route got shorter)
                        current_waypoint_count = len(engine.agent.route) if hasattr(engine.agent, 'route') and engine.agent.route else 0
                        if current_waypoint_count < last_waypoint_count:
                            waypoints_passed = last_waypoint_count - current_waypoint_count
                            vehicle_loc = engine.vehicle.get_location()
                            next_wp = engine.agent.get_next_global_waypoint_location()
                            print(f"       [Step {step}] ✅ Waypoint reached! {waypoints_passed} waypoint(s) passed")
                            print(f"       Vehicle pos: ({vehicle_loc.x:.1f}, {vehicle_loc.y:.1f})")
                            if next_wp:
                                print(f"       Next waypoint: ({next_wp.x:.1f}, {next_wp.y:.1f})")
                                dist_to_wp = vehicle_loc.distance(next_wp)
                                print(f"       Distance to next: {dist_to_wp:.1f}m")
                            print(f"       Remaining waypoints: {current_waypoint_count}")
                            last_waypoint_count = current_waypoint_count
                        
                        engine.agent.run_step()
                        reward = calcola_reward(engine.vehicle, engine.agent)
                        total_reward += reward
                        
                        # Check completion
                        if engine.agent.destination is not None:
                            current_loc = engine.vehicle.get_location()
                            dist_to_dest = current_loc.distance(engine.agent.destination)
                            
                            if dist_to_dest < 10.0 and not episode_complete:
                                total_reward += config.training.completion_bonus
                                episode_complete = True
                                completions_count += 1
                                print(f"    Individual {idx + 1}: ✅ Completed at step {step}")
                                break
                        
                        engine.agent.collision_last_step = False
                        engine.world.tick()
                        
                    except Exception as e:
                        print(f"  Error in step {step}: {e}")
                        total_reward -= 50
                        break
                
                fitnesses.append(total_reward)
                if (idx + 1) % 5 == 0 or idx == 0:
                    status = "✅" if episode_complete else "⏸️"
                    print(f"  Individual {idx + 1:2d}: {total_reward:7.1f} {status}")
            
            # Update NES
            engine.nes.step(fitnesses)
            fitness_history.append(fitnesses)
            
            # Track best
            max_fitness = max(fitnesses)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                save_best_model(
                    engine.agent.model, generation, best_fitness, save_dir,
                    {**config.model.to_dict(), **config.training.to_dict(), **config.nes.to_dict()}
                )
            
            # Log
            gen_time = time.time() - gen_start
            log_generation(
                log_path, generation, fitnesses, best_fitness, generation,
                gen_time, config.training.to_dict(), completions_count,
                config.nes.population_size
            )
            
            # Print summary
            mean_fit = np.mean(fitnesses)
            print(f"\n  Generation {generation + 1} Summary:")
            print(f"    Mean: {mean_fit:7.1f} | Max: {max_fitness:7.1f} | Best: {best_fitness:7.1f}")
            print(f"    Completions: {completions_count}/{config.nes.population_size} "
                  f"({completions_count/config.nes.population_size*100:.0f}%)")
            print(f"    Time: {gen_time:.1f}s")
            
            # Periodic checkpoint
            if (generation + 1) % config.training.save_interval == 0:
                save_checkpoint(engine.agent.model, engine.nes, generation,
                              best_fitness, fitness_history, save_dir)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best fitness: {best_fitness:.1f}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Training interrupted by user (Ctrl+C)")
        print("=" * 60)
        print("\nSaving checkpoint...")
        save_checkpoint(engine.agent.model, engine.nes, generation,
                       best_fitness, fitness_history, save_dir)
        print("✓ Checkpoint saved")


def train_batch(engine: TrainingEngine, config: Config, save_dir: str,
                start_generation: int = 0, best_fitness: float = float('-inf'),
                fitness_history: list = None) -> None:
    """
    Batch training: evaluate multiple vehicles in parallel.
    
    Args:
        engine: TrainingEngine instance
        config: Training configuration
        save_dir: Directory to save results
        start_generation: Generation to start from (for resume)
        best_fitness: Best fitness so far
        fitness_history: History of fitness values
    """
    if fitness_history is None:
        fitness_history = []
    
    from training.batch_trainer import BatchVehicleTrainer
    from train_temporal import TemporalMLAgent
    from models.simple_temporal_model import SimpleTemporalModel
    
    # Initialize batch trainer
    batch_trainer = BatchVehicleTrainer(
        world=engine.world,
        agent_class=TemporalMLAgent,
        model_class=SimpleTemporalModel,
        batch_size=config.training.batch_size,
        enable_camera_tracking=True,
        hud=engine.hud,
        config=config  # Pass config for model initialization
    )
    
    log_path = initialize_generation_log(save_dir, config.training.to_dict(), config.nes.to_dict())
    
    try:
        for generation in range(start_generation, config.training.num_generations):
            gen_start = time.time()
            print(f"\n=== Generation {generation + 1}/{config.training.num_generations} ===")
            
            # Evaluate population using batch trainer
            fitnesses = batch_trainer.evaluate_batch(
                population=engine.nes.population,
                max_steps=config.training.max_steps,
                generation=generation,
                max_generations=config.training.num_generations
            )
            
            # Update NES
            engine.nes.step(fitnesses)
            fitness_history.append(fitnesses)
            
            # Track best
            max_fitness = max(fitnesses)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                save_best_model(
                    engine.agent.model, generation, best_fitness, save_dir,
                    {**config.model.to_dict(), **config.training.to_dict(), **config.nes.to_dict()}
                )
            
            # Log
            gen_time = time.time() - gen_start
            completions = sum(1 for f in fitnesses if f > 500)  # Estimate based on bonus
            log_generation(
                log_path, generation, fitnesses, best_fitness, generation,
                gen_time, config.training.to_dict(), completions,
                config.nes.population_size
            )
            
            # Print summary
            mean_fit = np.mean(fitnesses)
            print(f"\n  Generation {generation + 1} Summary:")
            print(f"    Mean: {mean_fit:7.1f} | Max: {max_fitness:7.1f} | Best: {best_fitness:7.1f}")
            print(f"    Time: {gen_time:.1f}s")
            
            # Periodic checkpoint
            if (generation + 1) % config.training.save_interval == 0:
                save_checkpoint(engine.agent.model, engine.nes, generation,
                              best_fitness, fitness_history, save_dir)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best fitness: {best_fitness:.1f}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Training interrupted by user (Ctrl+C)")
        print("=" * 60)
        print("\nSaving checkpoint...")
        save_checkpoint(engine.agent.model, engine.nes, generation,
                       best_fitness, fitness_history, save_dir)
        print("✓ Checkpoint saved")
    
    # Return batch_trainer for cleanup in main()
    return batch_trainer


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='NES-CARLA Unified Training')
    parser.add_argument('--type', type=str, default='temporal', 
                       choices=['nes', 'temporal', 'batch'],
                       help='Training type (nes=linear, temporal=memory, batch=parallel)')
    parser.add_argument('--preset', type=str, default='standard',
                       choices=['quick_test', 'standard', 'batch', 'linear'],
                       help='Configuration preset')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file to resume from (e.g., training_logs/temporal_XXX/checkpoints/checkpoint_gen_50.pkl)')
    parser.add_argument('--generations', type=int, default=None,
                       help='Override number of generations')
    parser.add_argument('--population', type=int, default=None,
                       help='Override population size')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size (batch training only)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.preset)
    config.training.training_type = args.type
    
    # Apply command-line overrides
    if args.generations is not None:
        config.training.num_generations = args.generations
    if args.population is not None:
        config.nes.population_size = args.population
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    
    # Adjust model type for 'nes' training
    if args.type == 'nes':
        config.model.model_type = 'linear'
        config.model.input_size = 5
    
    # Print configuration
    config.print_summary()
    
    # Initialize training engine
    engine = TrainingEngine(config)
    
    # Connect to CARLA
    if not engine.connect_to_carla():
        print("Make sure CARLA simulator is running on localhost:2000")
        return 1
    
    # Setup training environment
    if not engine.spawn_vehicle():
        return 1
    
    engine.spawn_npcs()
    
    # Initialize model and agent
    model = engine.initialize_model()
    agent = engine.initialize_agent(model)
    nes = engine.initialize_nes()
    
    # Initialize HUD
    if args.type == 'batch':
        hud = engine.initialize_hud()
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"training_logs/{args.type}_{timestamp}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config.to_json(f"{save_dir}/config.json")
    
    # Handle resume
    start_generation = 0
    best_fitness = float('-inf')
    fitness_history = []
    
    if args.resume:
        try:
            print(f"\n📂 Loading checkpoint from: {args.resume}")
            start_generation, best_fitness, fitness_history = load_checkpoint(
                args.resume, model, nes
            )
            agent.model = model
            nes.agent = agent
            print(f"✓ Resumed from generation {start_generation}")
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
            return 1
    
    # Run training
    batch_trainer = None
    try:
        if args.type == 'batch':
            batch_trainer = train_batch(engine, config, save_dir, start_generation, best_fitness, fitness_history)
        else:
            train_sequential(engine, config, save_dir, start_generation, best_fitness, fitness_history)
    finally:
        # Cleanup batch trainer BEFORE engine cleanup to avoid double sensor destruction
        if batch_trainer is not None:
            try:
                batch_trainer.cleanup_batch()
            except Exception as e:
                pass  # Ignore cleanup errors
        
        # Then perform engine cleanup
        engine.cleanup()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
