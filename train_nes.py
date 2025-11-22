"""
Autonomous Driving Training with Natural Evolution Strategies (NES) in CARLA

This script implements an autonomous driving training system using Natural Evolution 
Strategies (NES) optimization algorithm in the CARLA simulator. The system trains 
a neural network-based agent to control a vehicle by optimizing parameters through 
evolutionary computation.

Features comprehensive logging system for training monitoring and analysis.

Author: [Manuel Di Lullo]
Date: 2025
Dependencies: CARLA simulator, numpy, custom NES agents, logging system
"""

import sys
import os
import time
import carla
import random
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional

# Import custom NES-based agents and models
from agents.nes_agent import NES
from agents.custom_ml_agent import CustomAdvancedMLAgent

from models.custom_ml_model import CustomMLModel

# Import utility functions for camera control, actor management, and reward calculation
from utils.camera import center_camera_on_vehicle
from utils.actors_handler import spawn_vehicles_and_pedestrians, cleanup_actors
from utils.logger import get_logger, log_performance_info, configure_logging, log_reward_info
from utils.reward import calcola_reward

def main():
    """
    Main training function that orchestrates the NES-based autonomous driving training.
    
    This function sets up the CARLA environment, spawns the training vehicle and NPCs,
    initializes the ML model and NES optimizer, and runs the evolutionary training loop.
    The training uses multiple generations where each individual in the population
    is evaluated based on driving performance.
    """
    # Configure logging system
    configure_logging(level="INFO", force_reconfigure=True)  # Set to INFO level to show all messages
    logger = get_logger(__name__)
    
    logger.info("🚀 Starting NES-CARLA autonomous driving training")
    
    # Initialize CARLA client and connect to simulator
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    cleanup_actors(world)
    
    logger.info("🌍 CARLA world initialized successfully")

    # Get vehicle blueprint and spawn main training vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    
    # Spawn NPC vehicles and pedestrians for realistic traffic scenario
    vehicles, pedestrians = spawn_vehicles_and_pedestrians(world, num_vehicles=0, num_pedestrians=0)

    # Initialize ML model and NES-based training agent
    # Updated model size for waypoint-aware state: [speed, dist_to_wp, angle_to_wp, obstacle_dist, collision]
    ml_model = CustomMLModel(param_size=15)  # 5 state dimensions × 3 actions = 15 parameters
    agent = CustomAdvancedMLAgent(vehicle, ml_model)
    nes = NES(agent=agent, population_size=10, sigma=0.1, learning_rate=0.1)

    # Training hyperparameters
    num_generations = 500  # Number of evolutionary generations
    max_steps = 300        # Maximum steps per individual evaluation

    try:
        # Main evolutionary training loop
        for generation in range(num_generations):
            print(f"\n=== Generazione {generation + 1}/{num_generations} ===")
            fitnesses = []
            for idx, individual in enumerate(nes.population):
                ml_model.set_parameters(individual)
                
                # Reset vehicle to random spawn point for fair evaluation
                vehicle.set_transform(spawn_point)
                
                # Reset agent state and update navigation for new spawn point
                agent.reset_for_new_episode()
                
                # Reset agent state for new evaluation
                agent.collision_data = False
                agent.low_speed_counter = 0
                
                # Run individual evaluation episode
                total_reward = 0
                for step in range(max_steps):
                    # Agent performs one control step
                    agent.run_step()
                    
                    # Calculate reward based on performance
                    reward = calcola_reward(vehicle, agent)
                    total_reward += reward
                    
                    agent.collision_last_step = False
                    
                    # Update simulation and camera view
                    world.tick()
                    center_camera_on_vehicle(world=world, vehicle=vehicle, FROM_ABOVE=False)
                
                # Log total reward at INFO level
                log_reward_info(logger, "TOTAL REWARD", total_reward)

                fitnesses.append(total_reward)
                print(f"Individuo {idx + 1} fitness finale: {total_reward:.2f}")

            nes.step(fitnesses)
            print(f"Generazione {generation} fitness media: {np.mean(fitnesses)}")

        # Save trained model parameters after training completion
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ml_model.save(f'training/trained_model_{timestamp}.npy')
        agent.destroy()
        vehicle.destroy()
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Training interrupted by user (Ctrl+C)")
        print("=" * 60)
    except Exception as e:
        print(f"\nErrore durante l'addestramento: {e}")
    finally:
        # Perform deep cleanup of all CARLA resources
        print("\n" + "=" * 60)
        print("Performing deep cleanup...")
        print("=" * 60)
        try:
            # Destroy agent
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
            
            # Deep cleanup of world actors
            if 'world' in locals() and world:
                try:
                    from training.training_utils import deep_cleanup
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

if __name__ == '__main__':
    main()