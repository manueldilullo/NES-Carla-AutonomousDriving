"""
Simplified Reward System for Autonomous Driving

Core reward components:
1. Waypoint Progress - Move toward next navigation point
2. Safety Penalties - Collisions and off-road driving
3. Stuck Detection - Single distance-based progress tracker
4. Speed Bonus - Encourage appropriate driving speed

Author: Manuel Di Lullo
Date: 2025
"""

import carla
import numpy as np
from typing import TYPE_CHECKING
from utils.logger import get_logger

if TYPE_CHECKING:
    from NES_Carla.agents.custom_ml_agent import CustomAdvancedMLAgent
else:
    CustomAdvancedMLAgent = None

# ============================================================================
# REWARD CONFIGURATION - SIMPLIFIED
# ============================================================================

# Core Progress Reward - DOMINANT SIGNAL
WAYPOINT_PROGRESS_SCALE = 100.0   # VERY LARGE reward per waypoint - must dominate

# Safety Penalties - MODERATE (discourage bad behavior)
COLLISION_PENALTY = 50.0          # Moderate collision penalty
OFF_ROAD_PENALTY = 10.0           # Moderate off-road penalty

# Stuck Detection (Distance-Based)
STUCK_CHECK_INTERVAL = 30         # Check progress every 30 steps
MIN_PROGRESS_DISTANCE = 10.0      # Minimum 10 meters to move in interval (increased from 5.0)
STUCK_PENALTY = 50.0              # Strong penalty for not making progress

# Movement Bonus - Encourage speed
MOVEMENT_THRESHOLD = 5.0          # km/h - reasonable movement
MOVEMENT_REWARD = 2.0             # Reward for moving at good speed
STANDING_STILL_PENALTY = 3.0      # Stronger penalty for not moving
GOOD_SPEED_THRESHOLD = 20.0       # km/h - good driving speed
GOOD_SPEED_REWARD = 5.0           # Bonus for driving at good speed

# Completion Bonus (Set in training script)
# COMPLETION_BONUS is passed from training config

# Initialize logger
logger = get_logger(__name__)

def calcola_reward(vehicle: carla.Vehicle, agent: 'CustomAdvancedMLAgent') -> float:
    """
    Simplified reward function with 4 core components.
    
    Components:
    1. Waypoint Progress - Primary signal for navigation (ONE-TIME reward per waypoint)
    2. Safety Penalties - Collisions and off-road
    3. Stuck Detection - Distance-based progress check
    4. Speed Bonus - Encourage movement
    
    Args:
        vehicle: CARLA vehicle instance
        agent: ML agent controlling the vehicle
        
    Returns:
        float: Reward value for current state
    """
    reward = 0.0
    
    # =============================================================================
    # 1. WAYPOINT PROGRESS (Primary Reward - ONE-TIME per waypoint)
    # =============================================================================
    # Track total waypoints in route (stored once at episode start)
    if not hasattr(agent, '_total_waypoints'):
        if hasattr(agent, 'route') and agent.route:
            agent._total_waypoints = len(agent.route)
            agent._waypoints_reached = 0
        else:
            agent._total_waypoints = 0
            agent._waypoints_reached = 0
    
    # Store previous route length to detect when waypoints are removed
    if not hasattr(agent, '_prev_route_length'):
        agent._prev_route_length = len(agent.route) if hasattr(agent, 'route') and agent.route else 0
    
    # Check if waypoints were removed (vehicle reached them)
    if hasattr(agent, 'route') and agent.route:
        current_route_length = len(agent.route)
        waypoints_removed = agent._prev_route_length - current_route_length
        
        if waypoints_removed > 0:
            # Reward for reaching waypoints
            waypoint_reward = waypoints_removed * WAYPOINT_PROGRESS_SCALE
            reward += waypoint_reward
            agent._waypoints_reached += waypoints_removed
            
            logger.info(f"🎯 Reached {waypoints_removed} waypoint(s)! "
                       f"Progress: {agent._waypoints_reached}/{agent._total_waypoints}, reward: +{waypoint_reward:.1f}")
        
        agent._prev_route_length = current_route_length
    
    # =============================================================================
    # 2. SAFETY PENALTIES
    # =============================================================================
    
    # Collision
    if agent.collision_last_step:
        reward -= COLLISION_PENALTY
        agent.collision_last_step = False
    
    # Off-road
    if not is_on_road(vehicle):
        reward -= OFF_ROAD_PENALTY
    
    # =============================================================================
    # 3. STUCK DETECTION (Distance-Based)
    # =============================================================================
    
    # Initialize stuck detection
    if not hasattr(agent, '_stuck_check_counter'):
        agent._stuck_check_counter = 0
        agent._stuck_start_pos = vehicle.get_location()
    
    agent._stuck_check_counter += 1
    
    # Check progress every N steps
    if agent._stuck_check_counter >= STUCK_CHECK_INTERVAL:
        current_pos = vehicle.get_location()
        distance_moved = current_pos.distance(agent._stuck_start_pos)
        
        # Penalize if didn't move minimum distance
        if distance_moved < MIN_PROGRESS_DISTANCE:
            reward -= STUCK_PENALTY
        
        # Reset counter and position
        agent._stuck_check_counter = 0
        agent._stuck_start_pos = current_pos
    
    # =============================================================================
    # 4. SPEED BONUS - Encourage appropriate driving speed
    # =============================================================================
    
    velocity = vehicle.get_velocity()
    speed_kmh = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    
    # Reward for good driving speed
    if speed_kmh > GOOD_SPEED_THRESHOLD:  # Driving at good speed (20+ km/h)
        reward += GOOD_SPEED_REWARD
    elif speed_kmh > MOVEMENT_THRESHOLD:  # Moving but slow (5-20 km/h)
        reward += MOVEMENT_REWARD
    else:  # Standing still or too slow (< 5 km/h)
        reward -= STANDING_STILL_PENALTY
    
    return reward

def is_on_road(vehicle: carla.Vehicle) -> bool:
    """
    Check if vehicle is on a drivable road surface.
    
    Args:
        vehicle: CARLA vehicle instance
        
    Returns:
        bool: True if on road, False otherwise
    """
    location = vehicle.get_location()
    world = vehicle.get_world()
    map_obj = world.get_map()
    
    # Get nearest waypoint
    waypoint = map_obj.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if waypoint is None:
        return False
    
    # Check if within reasonable distance from lane center
    distance_from_center = location.distance(waypoint.transform.location)
    return distance_from_center < 3.5  # ~lane width
