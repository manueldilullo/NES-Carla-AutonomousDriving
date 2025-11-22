"""
Batch Training System for Parallel Vehicle Evaluation

This module enables training multiple vehicles simultaneously in a single CARLA instance,
significantly improving sample efficiency and reducing wall-clock training time.

Key Features:
- Spawn multiple vehicles at once
- Evaluate population in parallel batches
- Track each vehicle independently
- Clean synchronization and resource management

Author: Manuel Di Lullo
Date: 2025
"""

import carla
import random
import time
import numpy as np
from typing import List, Tuple, Dict, Optional
from utils.reward import calcola_reward
from utils.camera import center_camera_on_vehicle


class BatchVehicleTrainer:
    """
    Manages parallel evaluation of multiple vehicles in same CARLA instance.
    
    Instead of evaluating population sequentially (1 vehicle at a time),
    this spawns multiple vehicles simultaneously to evaluate several
    individuals in parallel, reducing total training time.
    """
    
    def __init__(self, world: carla.World, agent_class, model_class, 
                 batch_size: int = 4, enable_camera_tracking: bool = True, hud=None, config=None):
        """
        Initialize batch trainer.
        
        Args:
            world: CARLA world instance
            agent_class: Agent class to instantiate
            model_class: Model class for agents
            batch_size: Number of vehicles to evaluate in parallel
            enable_camera_tracking: Whether to track vehicles with camera
            hud: HUD instance for real-time display (optional)
            config: Training configuration with model parameters
        """
        self.world = world
        self.agent_class = agent_class
        self.model_class = model_class
        self.batch_size = max(1, batch_size)  # Ensure at least 1
        self.config = config  # Store config for model initialization
        self.map = world.get_map()
        self.enable_camera_tracking = enable_camera_tracking
        self.hud = hud
        
        # Track active vehicles and agents
        self.active_vehicles: List[carla.Vehicle] = []
        self.active_agents = []
        self.active_models = []
        
        # Performance tracking
        self.current_fitnesses: List[float] = []
        self.tracked_vehicle_idx: Optional[int] = None
        
        print(f"✓ BatchVehicleTrainer initialized (batch_size={self.batch_size}, camera_tracking={enable_camera_tracking})")
    
    def _get_safe_spawn_points(self, all_spawns: List[carla.Transform], count: int, min_distance: float = 15.0) -> List[carla.Transform]:
        """Get spawn points that are clear of existing vehicles."""
        # Get all existing vehicles in world
        existing_vehicles = self.world.get_actors().filter('vehicle.*')
        
        # Filter spawn points that are far from existing vehicles
        safe_spawns = []
        for spawn in all_spawns:
            is_safe = True
            for vehicle in existing_vehicles:
                if spawn.location.distance(vehicle.get_location()) < min_distance:
                    is_safe = False
                    break
            if is_safe:
                safe_spawns.append(spawn)
        
        # If not enough safe spawns, use all available and hope for the best
        if len(safe_spawns) < count:
            safe_spawns = all_spawns
        
        # Return random selection
        return random.sample(safe_spawns, min(count, len(safe_spawns)))
    
    def spawn_batch(self, spawn_points: List[carla.Transform]) -> bool:
        """
        Spawn a batch of vehicles at different spawn points.
        
        Args:
            spawn_points: List of spawn transforms for vehicles
            
        Returns:
            bool: True if at least one vehicle spawned successfully
        """
        if not spawn_points:
            print("  ❌ No spawn points provided")
            return False
        
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        
        # Clear previous batch
        self.cleanup_batch()
        
        # Spawn vehicles
        for i in range(min(self.batch_size, len(spawn_points))):
            try:
                vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[i])
                self.active_vehicles.append(vehicle)
            except RuntimeError as e:
                print(f"  Warning: Failed to spawn vehicle {i+1}: {e}")
                continue
        
        if len(self.active_vehicles) == 0:
            print("  ❌ Failed to spawn any vehicles")
            return False
        
        # Give CARLA time to initialize vehicles
        for _ in range(10):
            self.world.tick()
        time.sleep(0.3)
        
        return True
    
    def setup_agents(self, model_config: dict, silent: bool = True) -> None:
        """
        Setup agents for all spawned vehicles.
        
        Args:
            model_config: Configuration dict for model initialization
            silent: If True, suppress agent initialization messages
        """
        self.active_agents = []
        self.active_models = []
        self.current_fitnesses = [0.0] * len(self.active_vehicles)
        
        # Suppress print statements during agent creation if silent
        import sys
        import io
        old_stdout = sys.stdout if silent else None
        
        for i, vehicle in enumerate(self.active_vehicles):
            if silent:
                sys.stdout = io.StringIO()  # Redirect stdout
            
            try:
                # Create model instance
                model = self.model_class(**model_config)
                self.active_models.append(model)
                
                # Create agent instance
                agent = self.agent_class(vehicle, model)
                self.active_agents.append(agent)
            finally:
                if silent:
                    sys.stdout = old_stdout  # Restore stdout
            
            # Give sensors time to initialize
            for _ in range(5):
                self.world.tick()
        
        # Print summary instead of individual agent info
        if not silent:
            print(f"  ✓ Setup {len(self.active_agents)} agents with sensors")
        
        time.sleep(0.2)
    
    def evaluate_batch(self, population: List[np.ndarray], max_steps: int,
                      generation: int = 0, max_generations: int = 300) -> List[float]:
        """
        Evaluate a batch of population members in parallel.
        
        Args:
            population: List of parameter arrays to evaluate
            max_steps: Maximum steps per episode
            generation: Current generation number
            max_generations: Total generations
            
        Returns:
            List of fitness values for evaluated individuals
        """
        if len(population) == 0:
            return []
        
        batch_fitnesses = []
        # Ensure batch_size is at least 1
        batch_size = max(1, min(self.batch_size, len(population)))
        
        for batch_start in range(0, len(population), batch_size):
            batch_end = min(batch_start + batch_size, len(population))
            current_batch = population[batch_start:batch_end]
            
            # Setup this batch
            batch_results = self._evaluate_single_batch(
                current_batch, max_steps, generation, max_generations
            )
            batch_fitnesses.extend(batch_results)
        
        return batch_fitnesses
    
    def _evaluate_single_batch(self, parameter_batch: List[np.ndarray],
                               max_steps: int, generation: int,
                               max_generations: int) -> List[float]:
        """
        Evaluate a single batch of individuals in parallel.
        
        Args:
            parameter_batch: List of parameter arrays for this batch
            max_steps: Maximum steps per episode
            generation: Current generation
            max_generations: Total generations
            
        Returns:
            List of fitness values
        """
        num_vehicles = len(parameter_batch)
        
        if num_vehicles == 0:
            return []
        
        # Always spawn fresh batch to avoid state issues
        self.cleanup_batch()
        
        # Wait for cleanup to complete
        for _ in range(3):
            self.world.tick()
        time.sleep(0.2)
        
        # Get safe spawn points (far from existing vehicles)
        all_spawn_points = self.map.get_spawn_points()
        selected_spawns = self._get_safe_spawn_points(all_spawn_points, num_vehicles)
        
        if not self.spawn_batch(selected_spawns):
            print(f"  ❌ Failed to spawn batch of {num_vehicles} vehicles")
            return [0.0] * num_vehicles
        
        # Setup agents with config values (silent mode)
        if self.config:
            model_config = {
                'input_size': self.config.model.input_size,
                'memory_size': self.config.model.memory_size,
                'decay_factor': self.config.model.decay_factor
            }
        else:
            # Fallback to defaults if no config provided
            model_config = {
                'input_size': 10,
                'memory_size': 5,
                'decay_factor': 0.85
            }
        self.setup_agents(model_config, silent=True)
        
        print(f"  📊 Evaluating batch of {num_vehicles} vehicles...")
        
        # Set parameters for each agent and setup individual destinations
        for i, (agent, params) in enumerate(zip(self.active_agents[:num_vehicles],
                                                parameter_batch)):
            agent.model.set_parameters(params)
            agent.model.reset_memory()
            # Enable waypoint visualization (visualize=True) so each vehicle has red waypoints
            agent.reset_for_new_episode(generation, max_generations, visualize=True, world=self.world)
            
            # Draw ONLY the next waypoint with short life_time
            next_wp = agent.get_next_global_waypoint_location()
            if next_wp:
                self.world.debug.draw_point(
                    next_wp,
                    size=0.4,
                    color=carla.Color(r=255, g=0, b=0),
                    life_time=3.0  # Short life - will refresh during episode
                )
            
            # DEBUG: Print detailed initial state
            vehicle_loc = self.active_vehicles[i].get_location()
            print(f"    🚗 Vehicle {i+1} START:")
            print(f"       Position: ({vehicle_loc.x:.1f}, {vehicle_loc.y:.1f}, {vehicle_loc.z:.1f})")
            
            # Verify route was set up
            if not hasattr(agent, 'route') or agent.route is None or len(agent.route) == 0:
                print(f"       ⚠️ Warning: No route!")
            elif agent.destination:
                next_wp = agent.get_next_global_waypoint_location()
                dest_loc = agent.destination
                distance_to_dest = vehicle_loc.distance(dest_loc)
                print(f"       Route: {len(agent.route)} waypoints")
                if next_wp:
                    print(f"       Next waypoint: ({next_wp.x:.1f}, {next_wp.y:.1f}, {next_wp.z:.1f})")
                    dist_to_wp = vehicle_loc.distance(next_wp)
                    print(f"       Distance to next waypoint: {dist_to_wp:.1f}m")
                print(f"       Destination: ({dest_loc.x:.1f}, {dest_loc.y:.1f}, {dest_loc.z:.1f})")
                print(f"       Distance to destination: {distance_to_dest:.1f}m")
            else:
                print(f"       ⚠️ Warning: Route exists but no destination!")
        
        # Run parallel evaluation
        fitnesses = [0.0] * num_vehicles
        active_mask = [True] * num_vehicles  # Track which vehicles are still running
        step_rewards = [[] for _ in range(num_vehicles)]  # Track rewards per step
        last_waypoint_counts = [len(self.active_agents[i].route) if hasattr(self.active_agents[i], 'route') and self.active_agents[i].route else 0 for i in range(num_vehicles)]
        
        # Select vehicle to track with camera at the start
        if self.enable_camera_tracking:
            self.tracked_vehicle_idx = random.randint(0, num_vehicles - 1)
            # Set camera on first vehicle immediately
            try:
                center_camera_on_vehicle(
                    world=self.world,
                    vehicle=self.active_vehicles[self.tracked_vehicle_idx],
                    FROM_ABOVE=False
                )
                print(f"  📷 Camera tracking Vehicle {self.tracked_vehicle_idx + 1}")
            except Exception as e:
                print(f"  Warning: Initial camera setup failed: {e}")
        
        # Print status every N steps
        status_interval = 100
        control_debug_interval = 50  # Debug control values every 50 steps
        
        # Track initial positions to detect movement
        initial_positions = [v.get_transform().location for v in self.active_vehicles[:num_vehicles]]
        
        # Adjust num_vehicles if some failed to spawn
        actual_num_vehicles = len(self.active_vehicles)
        if actual_num_vehicles < num_vehicles:
            print(f"  ⚠️ Warning: Only spawned {actual_num_vehicles}/{num_vehicles} vehicles")
            # Pad fitnesses and active_mask to match expected size
            fitnesses = [0.0] * num_vehicles
            active_mask = [i < actual_num_vehicles for i in range(num_vehicles)]
            step_rewards = [[] for _ in range(num_vehicles)]
            num_vehicles = actual_num_vehicles
            
            # CRITICAL: Adjust tracked vehicle index if it's out of bounds
            if self.enable_camera_tracking and self.tracked_vehicle_idx >= actual_num_vehicles:
                self.tracked_vehicle_idx = max(0, actual_num_vehicles - 1)
                if actual_num_vehicles > 0:
                    try:
                        center_camera_on_vehicle(
                            world=self.world,
                            vehicle=self.active_vehicles[self.tracked_vehicle_idx],
                            FROM_ABOVE=False
                        )
                        print(f"  📷 Camera adjusted to Vehicle {self.tracked_vehicle_idx + 1}")
                    except:
                        pass
            
            # CRITICAL: Adjust tracked vehicle index if it's out of bounds
            if self.enable_camera_tracking and self.tracked_vehicle_idx >= actual_num_vehicles:
                self.tracked_vehicle_idx = max(0, actual_num_vehicles - 1)
                if actual_num_vehicles > 0:
                    try:
                        center_camera_on_vehicle(
                            world=self.world,
                            vehicle=self.active_vehicles[self.tracked_vehicle_idx],
                            FROM_ABOVE=False
                        )
                        print(f"  📷 Camera adjusted to Vehicle {self.tracked_vehicle_idx + 1}")
                    except:
                        pass
        
        for step in range(max_steps):
            # Update all active vehicles
            for i in range(num_vehicles):
                if not active_mask[i]:
                    continue
                
                try:
                    vehicle = self.active_vehicles[i]
                    agent = self.active_agents[i]
                    
                    # EARLY TERMINATION: Stop vehicles that are stuck or heavily penalized
                    # If fitness is very negative, vehicle is stuck/colliding repeatedly
                    if fitnesses[i] < -1000:
                        print(f"       [Vehicle {i+1}, Step {step}] ⏹️ Early stop (stuck/collided, fitness={fitnesses[i]:.0f})")
                        active_mask[i] = False
                        continue
                    
                    # DEBUG: Check if waypoint was reached
                    current_waypoint_count = len(agent.route) if hasattr(agent, 'route') and agent.route else 0
                    if current_waypoint_count < last_waypoint_counts[i]:
                        waypoints_passed = last_waypoint_counts[i] - current_waypoint_count
                        vehicle_loc = vehicle.get_location()
                        next_wp = agent.get_next_global_waypoint_location()
                        print(f"       [Vehicle {i+1}, Step {step}] ✅ Waypoint reached! {waypoints_passed} passed")
                        print(f"       Position: ({vehicle_loc.x:.1f}, {vehicle_loc.y:.1f})")
                        if next_wp:
                            print(f"       Next waypoint: ({next_wp.x:.1f}, {next_wp.y:.1f})")
                            dist_to_wp = vehicle_loc.distance(next_wp)
                            print(f"       Distance: {dist_to_wp:.1f}m")
                        print(f"       Remaining waypoints: {current_waypoint_count}")
                        last_waypoint_counts[i] = current_waypoint_count
                    
                    # Execute control step
                    control = agent.run_step()
                    
                    # Debug: Print control values and vehicle state for first vehicle periodically
                    if i == 0 and step % control_debug_interval == 0:
                        velocity = vehicle.get_velocity()
                        import math
                        speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                        speed_kmh = speed_ms * 3.6
                        location = vehicle.get_transform().location
                        
                        print(f"    [Step {step}] Vehicle 1:")
                        print(f"      Control: Steer={control.steer:.3f}, Throttle={control.throttle:.3f}, Brake={control.brake:.3f}")
                        print(f"      Speed: {speed_kmh:.1f} km/h ({speed_ms:.2f} m/s)")
                        print(f"      Position: ({location.x:.1f}, {location.y:.1f}, {location.z:.1f})")
                        print(f"      Current fitness: {fitnesses[i]:.1f}")
                    
                    # Calculate reward
                    reward = calcola_reward(vehicle, agent)
                    fitnesses[i] += reward
                    step_rewards[i].append(reward)
                    self.current_fitnesses[i] = fitnesses[i]
                    
                    # Check if destination reached
                    if agent.destination is not None:
                        current_loc = vehicle.get_location()
                        dist_to_dest = current_loc.distance(agent.destination)
                        
                        if dist_to_dest < 10.0:
                            # Completed! Add bonus and mark as done
                            completion_bonus = 500.0
                            fitnesses[i] += completion_bonus
                            self.current_fitnesses[i] = fitnesses[i]
                            active_mask[i] = False
                            print(f"    🎯 Vehicle {i+1} completed at step {step}! Fitness: {fitnesses[i]:.1f}")
                            
                            # Setup new destination with visible waypoints
                            agent.reset_for_new_episode(generation, max_generations, visualize=True, world=self.world)
                            if agent.destination:
                                new_dest = agent.destination
                                new_distance = current_loc.distance(new_dest)
                                print(f"    🎯 Vehicle {i+1} new destination: ({new_dest.x:.1f}, {new_dest.y:.1f}), Distance: {new_distance:.1f}m")
                    
                except Exception as e:
                    print(f"  ❌ Error in vehicle {i+1}: {e}")
                    fitnesses[i] -= 50
                    active_mask[i] = False
            
            # Refresh waypoint visualization periodically (every 50 steps)
            # Only draw NEXT waypoint with short life_time so passed waypoints disappear
            if step % 50 == 0:
                for i in range(num_vehicles):
                    if active_mask[i]:
                        agent = self.active_agents[i]
                        # Get next waypoint location from agent (respects removal logic)
                        next_wp_loc = agent.get_next_global_waypoint_location()
                        if next_wp_loc:
                            self.world.debug.draw_point(
                                next_wp_loc,
                                size=0.4,
                                color=carla.Color(r=255, g=0, b=0),
                                life_time=3.0  # SHORT life_time - disappears quickly
                            )
            
            # Update camera every step to follow tracked vehicle smoothly
            if self.enable_camera_tracking and self.tracked_vehicle_idx is not None:
                if self.tracked_vehicle_idx < len(self.active_vehicles):
                    try:
                        center_camera_on_vehicle(
                            world=self.world,
                            vehicle=self.active_vehicles[self.tracked_vehicle_idx],
                            FROM_ABOVE=False
                        )
                    except:
                        pass  # Ignore camera errors
            
            # Print progress update and update HUD
            if step % status_interval == 0 and step > 0:
                active_count = sum(active_mask)
                avg_fitness = sum(fitnesses) / num_vehicles
                
                # Check if vehicles have moved from initial positions
                total_movement = 0
                movement_count = 0
                for i in range(min(num_vehicles, len(self.active_vehicles))):
                    if i < len(initial_positions):
                        current_pos = self.active_vehicles[i].get_transform().location
                        movement = initial_positions[i].distance(current_pos)
                        total_movement += movement
                        movement_count += 1
                
                avg_movement = total_movement / movement_count if movement_count > 0 else 0
                
                print(f"    Step {step}/{max_steps}: {active_count}/{num_vehicles} active, Avg fitness: {avg_fitness:.1f}")
                print(f"    Average movement from start: {avg_movement:.1f}m (total: {total_movement:.1f}m)")
            
            # Update HUD with tracked vehicle info (every 10 steps)
            if self.hud and step % 10 == 0 and self.tracked_vehicle_idx is not None:
                try:
                    tracked_vehicle = self.active_vehicles[self.tracked_vehicle_idx]
                    tracked_agent = self.active_agents[self.tracked_vehicle_idx]
                    tracked_fitness = fitnesses[self.tracked_vehicle_idx]
                    
                    self.hud.draw(
                        vehicle=tracked_vehicle,
                        destination=tracked_agent.destination,
                        next_waypoint=tracked_agent.get_next_global_waypoint_location(),
                        generation=generation + 1,
                        episode_step=step,
                        fitness=tracked_fitness,
                        additional_info={
                            f'Vehicle {self.tracked_vehicle_idx + 1}/{num_vehicles}': '(Tracked)',
                            'Active Vehicles': f'{sum(active_mask)}/{num_vehicles}',
                            'Avg Fitness': f'{avg_fitness:.1f}',
                            'Best Fitness': f'{max(fitnesses):.1f}'
                        }
                    )
                except Exception as e:
                    pass  # Silently ignore HUD errors
            
            # Tick world once for all vehicles
            self.world.tick()
            
            # Early exit if all vehicles are done
            if not any(active_mask):
                print(f"    ✓ All vehicles completed at step {step}")
                break
        
        # Print final results for this batch
        print(f"\n  📈 Batch Results:")
        for i in range(num_vehicles):
            avg_reward = sum(step_rewards[i]) / len(step_rewards[i]) if step_rewards[i] else 0
            status = "✅" if not active_mask[i] else "⏸️"
            print(f"    Vehicle {i+1}: Fitness={fitnesses[i]:7.1f}, Avg Reward/step={avg_reward:6.2f} {status}")
        
        # Update camera at end of episode to track best performing vehicle
        if self.enable_camera_tracking:
            self._update_camera_tracking()
        
        return fitnesses
    
    def _update_camera_tracking(self) -> None:
        """
        Update camera to follow best performing or tracked vehicle.
        """
        if not self.active_vehicles or not self.enable_camera_tracking:
            return
        
        # Safety check: ensure we have valid data
        if not self.current_fitnesses or len(self.current_fitnesses) == 0:
            return
        
        try:
            # Find best performing vehicle
            best_idx = max(range(len(self.current_fitnesses)), 
                          key=lambda i: self.current_fitnesses[i])
            
            # Safety check: ensure best_idx is within bounds
            if best_idx >= len(self.active_vehicles):
                best_idx = len(self.active_vehicles) - 1
            
            # If there's a tie or similar performance, keep tracking same vehicle
            if (self.tracked_vehicle_idx is not None and 
                self.tracked_vehicle_idx < len(self.current_fitnesses) and
                self.tracked_vehicle_idx < len(self.active_vehicles)):
                current_best = self.current_fitnesses[best_idx]
                tracked_fitness = self.current_fitnesses[self.tracked_vehicle_idx]
                
                # Only switch if new best is significantly better (>10% difference)
                if tracked_fitness >= current_best * 0.9:
                    best_idx = self.tracked_vehicle_idx
            
            self.tracked_vehicle_idx = best_idx
            
            # Update camera
            if best_idx < len(self.active_vehicles):
                try:
                    center_camera_on_vehicle(
                        world=self.world,
                        vehicle=self.active_vehicles[best_idx],
                        FROM_ABOVE=False
                    )
                except:
                    pass  # Ignore camera errors
        except Exception as e:
            # Silently handle any camera tracking errors
            pass
    
    def cleanup_batch(self) -> None:
        """Clean up all vehicles and agents in current batch."""
        # Destroy agents (which destroys sensors)
        # Destroy agents (includes sensors)
        for agent in self.active_agents:
            try:
                if hasattr(agent, 'destroy') and not (hasattr(agent, '_destroyed') and agent._destroyed):
                    agent.destroy()
            except Exception as e:
                pass  # Silently ignore already destroyed actors
        
        # Destroy vehicles
        for vehicle in self.active_vehicles:
            try:
                if hasattr(vehicle, 'is_alive') and vehicle.is_alive:
                    vehicle.destroy()
            except Exception as e:
                pass  # Silently ignore already destroyed actors
        
        self.active_vehicles = []
        self.active_agents = []
        self.active_models = []
        
        # Clean up orphaned actors - ensure cleanup is processed
        try:
            for _ in range(5):
                self.world.tick()
            time.sleep(0.05)
        except:
            pass  # Ignore tick errors if world is being cleaned up
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup_batch()
        except:
            pass
