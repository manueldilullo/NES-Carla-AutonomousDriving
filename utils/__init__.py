"""
Utility functions for NES CARLA autonomous driving.

This package provides utilities for actor management, reward calculation,
and camera positioning for autonomous driving training in CARLA.

Modules:
    actors_handler: Functions for spawning and managing vehicles/pedestrians
    reward: Reward calculation functions for training
    camera: Camera positioning utilities for visualization
"""

from .actors_handler import spawn_vehicles_and_pedestrians, cleanup_actors
from .reward import calcola_reward, is_on_road
from .camera import center_camera_on_vehicle
from .misc import get_speed, vector, draw_waypoints

__all__ = [
    'spawn_vehicles_and_pedestrians',
    'cleanup_actors',
    'calcola_reward', 
    'is_on_road',
    'center_camera_on_vehicle',
    "get_speed",
    "vector",
    "draw_waypoints"
]