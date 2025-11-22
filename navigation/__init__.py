"""
Navigation Module for CARLA Autonomous Driving

This module provides navigation and control utilities for autonomous vehicles
in the CARLA simulator environment.

Components:
- controller: Vehicle control and steering logic
- global_route_planner: High-level route planning across the map
- local_planner: Local path planning and obstacle avoidance

Author: Manuel Di Lullo
Date: 2025
Dependencies: CARLA Python API
"""

from .controller import *
from .global_route_planner import *
from .local_planner import *

__all__ = [
    # Controller exports
    'controller',
    # Route planner exports  
    'global_route_planner',
    # Local planner exports
    'local_planner'
]