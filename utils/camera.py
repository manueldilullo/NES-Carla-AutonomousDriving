"""
Camera utilities for CARLA autonomous driving visualization.

This module provides simple camera positioning functions for monitoring
vehicle behavior during training and evaluation.
"""

import carla
import math
from typing import Optional

def center_camera_on_vehicle(world: carla.World, vehicle: carla.Vehicle, 
                            spectator: Optional[carla.Actor] = None, 
                            FROM_ABOVE: bool = True) -> carla.Actor:
    """
    Position the spectator camera to follow the vehicle by attaching it.
    
    Args:
        world: CARLA world instance
        vehicle: Vehicle to follow
        spectator: Spectator camera (uses default if None)
        FROM_ABOVE: If True, bird's eye view; if False, chase camera
        
    Returns:
        The spectator camera actor
    """
    if spectator is None:
        spectator = world.get_spectator()

    # Get current vehicle position and orientation
    transform = vehicle.get_transform()

    if FROM_ABOVE:
        # Bird's eye view: camera positioned above the vehicle
        camera_offset = carla.Transform(
            carla.Location(z=50),  # 50 meters above vehicle
            carla.Rotation(pitch=-90)  # looking straight down
        )
    else:
        # Chase camera: positioned behind the vehicle
        distance_behind = 10  # meters behind vehicle
        height = 5           # meters above ground
        
        camera_offset = carla.Transform(
            carla.Location(x=-distance_behind, z=height),
            carla.Rotation(pitch=-10)
        )

    # Calculate world position based on vehicle transform and camera offset
    camera_location = transform.location + transform.get_forward_vector() * camera_offset.location.x
    camera_location += transform.get_right_vector() * camera_offset.location.y
    camera_location.z += camera_offset.location.z
    
    if FROM_ABOVE:
        camera_rotation = camera_offset.rotation
    else:
        camera_rotation = carla.Rotation(
            pitch=camera_offset.rotation.pitch,
            yaw=transform.rotation.yaw,
            roll=0
        )
    
    # Apply camera transform
    spectator.set_transform(carla.Transform(camera_location, camera_rotation))
    return spectator