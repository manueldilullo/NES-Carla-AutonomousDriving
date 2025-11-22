"""
Actor Management Utilities for CARLA Autonomous Driving

This module provides comprehensive utilities for managing actors (vehicles, pedestrians,
and their controllers) in the CARLA simulator environment. These utilities are essential
for creating realistic traffic scenarios and maintaining proper resource management
during autonomous driving training and evaluation.

Key Features:
- Automated NPC vehicle spawning with autopilot behavior
- Pedestrian generation with AI-controlled movement
- Comprehensive cleanup system to prevent memory leaks
- Robust error handling for unstable simulation environments
- Configurable parameters for different traffic densities

The module handles the complete lifecycle of background actors:
1. Spawning vehicles and pedestrians at strategic locations
2. Configuring AI behaviors for realistic traffic simulation
3. Managing actor states during simulation
4. Proper cleanup and resource deallocation

Author: Manuel Di Lullo
Date: 2025
Dependencies: CARLA Python API
"""

import carla
import random
from typing import List, Tuple, Optional, Dict, Any


def spawn_vehicles_and_pedestrians(world: carla.World, num_vehicles: int = 10, num_pedestrians: int = 20,
                                 safe_distance: float = 5.0, max_attempts: int = 50) -> Tuple[List[carla.Vehicle], List[carla.Walker], List[carla.Actor]]:
    """
    Spawn NPC vehicles and pedestrians in the CARLA world to create a realistic traffic scenario.
    
    This function creates a dynamic environment by spawning autonomous vehicles and 
    pedestrians that move independently, providing realistic traffic conditions for training.
    The function ensures safe spawning by checking distances and provides fallback mechanisms
    for unreliable spawn points.
    
    Spawning Strategy:
    1. Vehicles: Use predefined spawn points with autopilot behavior
    2. Pedestrians: Generate random navigation-accessible locations
    3. Controllers: Attach AI controllers for autonomous behavior
    4. Safety: Maintain minimum distances between actors
    
    Args:
        world (carla.World): The CARLA world instance
        num_vehicles (int): Number of NPC vehicles to spawn (default: 10)
        num_pedestrians (int): Number of pedestrians to spawn (default: 20)
        safe_distance (float): Minimum distance between spawned vehicles (default: 5.0m)
        max_attempts (int): Maximum spawn attempts per actor (default: 50)
        
        Returns:
            Tuple[List[carla.Vehicle], List[carla.Walker], List[carla.Actor]]: 
                - vehicles: List of spawned NPC vehicles
                - pedestrians: List of spawned pedestrian walkers
        Actual number of spawned actors may be less than requested due to:
        - Limited spawn points
        - Safety distance requirements
        - Failed spawn attempts
        - Navigation mesh limitations for pedestrians
    """
    blueprint_library = world.get_blueprint_library()
    vehicles = []
    pedestrians = []
    controllers = []

    print(f"Attempting to spawn {num_vehicles} vehicles and {num_pedestrians} pedestrians...")

    # === VEHICLE SPAWNING ===
    print("Spawning NPC vehicles...")
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)  # Randomize spawn order for variety

    vehicles_spawned = 0
    for i, spawn_point in enumerate(spawn_points):
        if vehicles_spawned >= num_vehicles:
            break
            
        # Check safe distance from existing vehicles
        safe_to_spawn = True
        if safe_distance > 0:
            for existing_vehicle in vehicles:
                if existing_vehicle and existing_vehicle.is_alive:
                    distance = spawn_point.location.distance(existing_vehicle.get_location())
                    if distance < safe_distance:
                        safe_to_spawn = False
                        break
        
        if not safe_to_spawn:
            continue
            
        # Attempt to spawn vehicle
        try:
            vehicle_bp = random.choice(vehicle_blueprints)
            # Randomize vehicle color for visual variety
            if vehicle_bp.has_attribute('color'):
                color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
                vehicle_bp.set_attribute('color', color)
                
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle:
                vehicles.append(vehicle)
                print(f"Vehicle spawned successfully at point {i}")
                
                try:
                    # Enable autopilot for realistic NPC behavior
                    vehicle.set_autopilot(True)
                    print(f"Autopilot enabled for vehicle {len(vehicles)}")
                except Exception as autopilot_error:
                    print(f"Warning: Failed to set autopilot for vehicle: {autopilot_error}")
                
                vehicles_spawned += 1
            else:
                print(f"Vehicle spawn returned None at point {i}")
                
        except Exception as e:
            print(f"Failed to spawn vehicle at spawn point {i}: {e}")
            continue

    print(f"Successfully spawned {vehicles_spawned} vehicles")

    # === PEDESTRIAN SPAWNING ===
    print("Spawning pedestrians...")
    walker_blueprints = blueprint_library.filter('walker.pedestrian.*')
    walker_controller_bp = blueprint_library.find('controller.ai.walker')
    
    pedestrians_spawned = 0
    attempts = 0
    
    while pedestrians_spawned < num_pedestrians and attempts < max_attempts * num_pedestrians:
        attempts += 1
        
        try:
            # Get random location from navigation mesh
            spawn_location = world.get_random_location_from_navigation()
            if spawn_location is None:
                continue
                
            spawn_transform = carla.Transform(spawn_location)
            
            # Check safe distance from vehicles
            safe_to_spawn = True
            for vehicle in vehicles:
                if vehicle and vehicle.is_alive:
                    distance = spawn_location.distance(vehicle.get_location())
                    if distance < safe_distance:
                        safe_to_spawn = False
                        break
            
            if not safe_to_spawn:
                continue
            
            # Spawn pedestrian
            walker_bp = random.choice(walker_blueprints)
            # Randomize pedestrian appearance
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            if walker_bp.has_attribute('speed'):
                # Set random walking speed
                walker_bp.set_attribute('speed', str(random.uniform(1.0, 2.0)))
                
            walker = world.try_spawn_actor(walker_bp, spawn_transform)
            if walker:
                pedestrians.append(walker)
                
                # Attach AI controller for autonomous pedestrian movement
                controller = world.try_spawn_actor(walker_controller_bp, carla.Transform(), attach_to=walker)
                if controller:
                    controllers.append(controller)
                    controller.start()
                    
                    # Set random destination and speed
                    destination = world.get_random_location_from_navigation()
                    if destination:
                        controller.go_to_location(destination)
                    controller.set_max_speed(random.uniform(1.0, 2.5))  # 1-2.5 m/s walking speed
                    
                pedestrians_spawned += 1
                
        except Exception as e:
            print(f"Failed to spawn pedestrian (attempt {attempts}): {e}")
            continue

    print(f"Successfully spawned {pedestrians_spawned} pedestrians")
    print(f"Total spawn attempts: {attempts}")

    return vehicles, pedestrians


def cleanup_actors(world: carla.World, exclude_hero: bool = True) -> int:
    """
    Clean up all user-spawned actors in the CARLA world.
    
    Destroys vehicles, pedestrians, controllers, and sensors while preserving
    hero vehicles and critical CARLA infrastructure.
    
    Args:
        world (carla.World): The CARLA world instance
        exclude_hero (bool): If True, keep hero vehicle (default: True)
        
    Returns:
        int: Number of actors destroyed
    """
    actors = world.get_actors()
    destroyed = 0
    
    # Safe actor types to destroy
    safe_types = ['vehicle.', 'walker.pedestrian', 'controller.ai.walker', 'sensor.camera', 'sensor.lidar']
    
    for actor in actors:
        # Skip hero vehicles
        if exclude_hero and 'hero' in actor.attributes.get('role_name', ''):
            continue
            
        # Only destroy safe actor types
        if not any(safe_type in actor.type_id for safe_type in safe_types):
            continue
            
        try:
            if hasattr(actor, 'stop'):
                actor.stop()
            actor.destroy()
            destroyed += 1
        except:
            pass  # Actor already destroyed or invalid
            
    print(f"Destroyed {destroyed} actors")
    return destroyed


def get_actor_counts(world: carla.World) -> Dict[str, int]:
    """
    Get current count of different actor types in the CARLA world.
    
    Useful for monitoring resource usage and detecting memory leaks.
    
    Args:
        world (carla.World): The CARLA world instance
        
    Returns:
        Dict[str, int]: Dictionary with counts of different actor types
    """
    actors = world.get_actors()
    
    counts = {
        'vehicles': 0,
        'pedestrians': 0,
        'sensors': 0,
        'controllers': 0,
        'other': 0,
        'total': len(actors)
    }
    
    for actor in actors:
        type_id = actor.type_id
        if 'vehicle' in type_id:
            counts['vehicles'] += 1
        elif 'walker' in type_id:
            counts['pedestrians'] += 1
        elif 'sensor' in type_id:
            counts['sensors'] += 1
        elif 'controller' in type_id:
            counts['controllers'] += 1
        else:
            counts['other'] += 1
    
    return counts


def monitor_traffic_manager(world: carla.World) -> Dict[str, Any]:
    """
    Monitor traffic manager status and vehicle behaviors.
    
    Args:
        world (carla.World): The CARLA world instance
        
    Returns:
        Dict[str, Any]: Traffic manager statistics and status
    """
    try:
        tm = carla.TrafficManager()
        vehicle_list = world.get_actors().filter('vehicle.*')
        
        autopilot_vehicles = 0
        for vehicle in vehicle_list:
            if hasattr(vehicle, 'get_traffic_manager_port'):
                autopilot_vehicles += 1
        
        return {
            'total_vehicles': len(vehicle_list),
            'autopilot_vehicles': autopilot_vehicles,
            'tm_port': tm.get_port(),
            'synchronous_mode': tm.get_synchronous_mode()
        }
    except Exception as e:
        return {'error': str(e)}