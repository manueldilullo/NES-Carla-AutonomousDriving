"""
Custom Machine Learning Agents for Autonomous Driving in CARLA

This module provides ML-based autonomous driving agents with varying levels of
sophistication. The agents use neural networks to make driving decisions based
on vehicle state and sensor data from the CARLA simulator.

Features:
- Basic ML agent with simple state representation
- Advanced ML agent with comprehensive sensor integration
- Real-time collision and obstacle detection
- Multi-modal sensor fusion (camera, lidar, collision detection)
- Performance monitoring and stuck vehicle detection
- Comprehensive logging system for debugging and analysis

Author: Manuel Di Lullo  
Date: 2025
Dependencies: CARLA Python API, numpy, logging system
"""

import carla
from carla import VehicleControl, Transform, Location
import numpy as np
from typing import List, Optional, Tuple, Any

from navigation.improved_global_route_planner import ImprovedGlobalRoutePlanner
from navigation.local_planner import LocalPlanner
from utils.logger import get_logger, log_prediction_debug, log_performance_info, log_route_info

class CustomAdvancedMLAgent:
    """
    Advanced machine learning agent with comprehensive sensor integration for autonomous driving.
    
    This agent provides a sophisticated implementation with multiple sensors and
    advanced state representation for complex autonomous driving scenarios. It includes
    collision detection, camera/lidar data processing, and performance monitoring.
    
    The agent features:
    - RGB camera for visual perception
    - LiDAR sensor for 3D environment mapping
    - Collision detection and response
    - Low-speed detection to prevent getting stuck
    - Comprehensive state representation with sensor fusion
    
    Attributes:
        vehicle (carla.Vehicle): The CARLA vehicle to control
        model: Neural network model for action prediction
        world (carla.World): CARLA world instance
        camera_data: Latest RGB camera frame
        collision_data (bool): Flag indicating if collision occurred
        collision_last_step (bool): Flag for collision in the last step
        lidar_data: Latest LiDAR point cloud data
        low_speed_counter (int): Counter for consecutive low-speed steps
        camera (carla.Actor): RGB camera sensor actor
        collision_sensor (carla.Actor): Collision detection sensor actor
        lidar_sensor (carla.Actor): LiDAR sensor actor
    """
    
    def __init__(self, vehicle: carla.Vehicle, model: Any):
        """
        Initialize the advanced ML agent with sensor setup.
        
        Args:
            vehicle (carla.Vehicle): CARLA vehicle instance to control
            model: ML model that implements predict(state) -> (steer, throttle, brake)
        """
        self.vehicle = vehicle
        self.model = model
        self.world = vehicle.get_world()
        self.map = self.world.get_map()
        
        # Initialize logger for this agent instance
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

        # Initialize global and local planners with improved route planner
        self.global_planner = ImprovedGlobalRoutePlanner(self.map, sampling_resolution=2.0)
        self.local_planner = LocalPlanner(self.vehicle)
        
        # Navigation will be set up when needed
        self.start = None
        self.destination = None
        self.route = None
        
        # Initialize sensor data storage
        self.camera_data = None
        self.collision_data = False
        self.collision_last_step = False
        self.lane_invasion_detected = False  # Track sidewalk/lane violations
        self.lidar_data = None
        self.low_speed_counter = 0
        
        # Initialize control variables
        self.steer, self.throttle, self.brake, self.yaw = [0.0] * 4
        self.last_steer, self.last_throttle, self.last_brake, self.last_yaw = [0.0] * 4
        
        # For stuck detection in reward function
        self.last_position = None
        
        # Setup all sensors
        self._setup_sensors()
        
    def _setup_destination(self) -> None:
        """
        Define start and end points for navigation.
        Uses current vehicle location as start and chooses a distant destination.
        """
        # Use current vehicle location as start
        current_location = self.vehicle.get_location()
        self.start = current_location
        
        # Choose a destination that's far enough away
        spawn_points = self.map.get_spawn_points()
        
        # Find a spawn point that's at least 50 meters away
        best_destination = None
        max_distance = 0
        
        for spawn_point in spawn_points:
            distance = current_location.distance(spawn_point.location)
            if distance > max_distance and distance > 50.0:  # At least 50m away
                max_distance = distance
                best_destination = spawn_point.location
        
        # Fallback: use the farthest spawn point if none is far enough
        if best_destination is None:
            distances = [current_location.distance(sp.location) for sp in spawn_points]
            farthest_idx = np.argmax(distances)
            best_destination = spawn_points[farthest_idx].location
        
        self.destination = best_destination
        distance_to_dest = current_location.distance(self.destination)
        self.logger.info(f"🎯 Navigation setup: Start ({self.start.x:.1f}, {self.start.y:.1f}) → "
                        f"Destination ({self.destination.x:.1f}, {self.destination.y:.1f}), "
                        f"Distance: {distance_to_dest:.1f}m")
    
    def _setup_route(self) -> None:
        """
        Setup the navigation route from start to destination.
        OPTION 1: Lane-aware route planning to ensure proper lane selection.
        Generates a route using the global planner with lane-aware waypoints.
        """
        if self.start is None or self.destination is None:
            self.logger.warning("Cannot setup route - start or destination not set")
            return
            
        try:
            # Get waypoints at start and destination on correct driving lanes
            start_wp = self.map.get_waypoint(
                self.start,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            
            dest_wp = self.map.get_waypoint(
                self.destination,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            
            if not start_wp or not dest_wp:
                self.logger.error("Failed to get valid waypoints for route")
                self.route = None
                return
            
            # Try to plan route
            try:
                self.route = self.global_planner.trace_route(
                    start_wp.transform.location,
                    dest_wp.transform.location
                )
            except Exception as route_error:
                self.logger.warning(f"Primary route failed ({route_error}), trying fallback destinations...")
                
                # FALLBACK 1: Try nearby spawn points as alternative destinations
                spawn_points = self.map.get_spawn_points()
                alternative_found = False
                
                # Sort by distance and try closest 5 alternative destinations
                distances = [(sp, self.start.distance(sp.location)) for sp in spawn_points]
                distances.sort(key=lambda x: x[1])
                
                for alt_spawn, dist in distances[1:6]:  # Skip first (too close), try next 5
                    if dist < 30.0:  # Too close
                        continue
                    if dist > 200.0:  # Too far
                        break
                    
                    try:
                        alt_dest_wp = self.map.get_waypoint(
                            alt_spawn.location,
                            project_to_road=True,
                            lane_type=carla.LaneType.Driving
                        )
                        
                        if alt_dest_wp:
                            self.route = self.global_planner.trace_route(
                                start_wp.transform.location,
                                alt_dest_wp.transform.location
                            )
                            
                            if self.route and len(self.route) > 0:
                                self.destination = alt_spawn.location
                                self.logger.info(f"✅ Alternative route found (distance: {dist:.1f}m)")
                                alternative_found = True
                                break
                    except:
                        continue
                
                if not alternative_found:
                    # FALLBACK 2: Create simple straight-line waypoint path
                    self.logger.warning("No connected route found, using simple waypoint path")
                    self.route = self._create_simple_waypoint_path(start_wp, dest_wp)
            
            if self.route and len(self.route) > 0:
                self.local_planner.set_global_plan(self.route)
                dest_loc = self.destination
                self.logger.info(f"✅ Route setup: {len(self.route)} waypoints to ({dest_loc.x:.1f}, {dest_loc.y:.1f})")
            else:
                self.logger.error("⚠️  Failed to create any valid route")
                self.route = None
                
        except Exception as e:
            self.logger.error(f"Failed to setup route: {e}")
            self.route = None
    
    def _create_simple_waypoint_path(self, start_wp, dest_wp, step_distance: float = 5.0):
        """
        Create a simple waypoint path when route planning fails.
        Uses forward propagation from start waypoint.
        
        Args:
            start_wp: Starting waypoint
            dest_wp: Destination waypoint  
            step_distance: Distance between waypoints
        
        Returns:
            List of (waypoint, RoadOption) tuples
        """
        from navigation.local_planner import RoadOption
        
        path = []
        current_wp = start_wp
        dest_location = dest_wp.transform.location
        max_waypoints = 50  # Limit path length
        
        for _ in range(max_waypoints):
            path.append((current_wp, RoadOption.LANEFOLLOW))
            
            # Check if close to destination
            if current_wp.transform.location.distance(dest_location) < step_distance:
                path.append((dest_wp, RoadOption.LANEFOLLOW))
                break
            
            # Get next waypoint
            next_wps = current_wp.next(step_distance)
            if not next_wps:
                break
            
            # Choose waypoint closest to destination
            best_wp = min(next_wps, key=lambda wp: wp.transform.location.distance(dest_location))
            current_wp = best_wp
        
        return path if len(path) > 1 else None
    
    def update_navigation_for_new_spawn(self) -> None:
        """
        Update navigation when vehicle is moved to a new spawn point.
        
        This method should be called after the vehicle is teleported to a new
        location during training to ensure proper navigation setup.
        """
        # Setup new destination based on current vehicle location
        self._setup_destination()
        
        # Setup new route
        self._setup_route()
        
        # Reset navigation-related tracking
        if hasattr(self, 'previous_distance_to_destination'):
            del self.previous_distance_to_destination  # Reset for reward calculation

    def _setup_sensors(self) -> None:
        """
        Setup and attach all sensors to the vehicle.
        
        Configures and spawns:
        1. RGB camera for visual perception
        2. Collision sensor for safety monitoring  
        3. LiDAR sensor for 3D environment mapping
        
        All sensors are attached to the vehicle and configured with appropriate
        parameters for autonomous driving applications.
        """
        blueprint_library = self.world.get_blueprint_library()

        # Setup front-facing RGB camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')    # Image width
        camera_bp.set_attribute('image_size_y', '480')    # Image height
        camera_bp.set_attribute('fov', '110')             # Field of view in degrees
        
        # Position camera in front of vehicle at appropriate height
        camera_spawn_point = Transform(Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_spawn_point, attach_to=self.vehicle)
        self.camera.listen(lambda data: self._on_camera_data(data))

        # Setup collision detection sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        # Setup LiDAR sensor for 3D environment perception
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')                    # Detection range in meters
        lidar_bp.set_attribute('rotation_frequency', '10')       # Rotations per second
        lidar_bp.set_attribute('channels', '32')                 # Number of laser channels
        lidar_bp.set_attribute('points_per_second', '56000')     # Point cloud density
        
        # Position LiDAR on top of vehicle
        lidar_spawn_point = Transform(Location(x=0, z=2.5))
        self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_spawn_point, attach_to=self.vehicle)
        self.lidar_sensor.listen(lambda data: self._on_lidar_data(data))

    def _on_camera_data(self, image: carla.Image) -> None:
        """
        Callback function for processing RGB camera data.
        
        Args:
            image (carla.Image): RGB image data from the camera sensor
        """
        self.camera_data = image

    def _on_collision(self, event: carla.CollisionEvent) -> None:
        """
        Callback function for handling collision events.
        
        Sets collision flags when the vehicle collides with any object in the environment.
        This information is used for safety monitoring and reward calculation.
        
        Args:
            event (carla.CollisionEvent): Collision event data containing collision details
        """
        self.collision_data = True
        self.collision_last_step = True

    def _on_lidar_data(self, data: carla.LidarMeasurement) -> None:
        """
        Callback function for processing LiDAR point cloud data.
        
        Args:
            data (carla.LidarMeasurement): LiDAR point cloud data
        """
        # Convert CARLA LidarMeasurement to numpy array
        # Each point is (x, y, z, intensity) in vehicle coordinate system
        try:
            points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            # Keep only x, y, z coordinates (ignore intensity)
            self.lidar_data = points[:, :3]
        except Exception as e:
            print(f"Warning: Failed to process LiDAR data: {e}")
            self.lidar_data = None

    def get_state(self) -> List[float]:
        """
        Extract comprehensive state information from vehicle and sensors.
        
        Computes an enhanced state representation that includes vehicle dynamics
        and sensor information for more sophisticated decision making.
        
        Returns:
            List[float]: Enhanced state vector [x_position, y_position, speed, collision_flag]
                - x_position: Vehicle's x-coordinate in world space
                - y_position: Vehicle's y-coordinate in world space
                - speed: Current speed in m/s
                - collision_flag: 1.0 if collision detected, 0.0 otherwise
                - distance_to_goal: Euclidean distance to the destination
        Note:
            This can be extended to include processed camera/lidar features,
            waypoint information, or other relevant driving context.
        """
        # Get basic vehicle state
        position = self.vehicle.get_location()
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        location = transform.location

        # Add collision information to state
        collision_flag = 1.0 if self.collision_data else 0.0
        
        # Calculate distance to goal
        distance_to_goal = position.distance(self.destination)

        return [location.x, location.y, speed, collision_flag, distance_to_goal]
    
    def get_rich_state(self) -> List[float]:
        """
        Extract comprehensive state information including navigation and sensor data.
        
        This method provides an enhanced state representation that combines vehicle
        dynamics, navigation context, and sensor information for sophisticated ML
        decision making.
        
        Returns:
            List[float]: Enhanced state vector containing:
                - speed: Current vehicle speed in m/s
                - dist_to_wp: Distance to next waypoint in meters
                - front_obstacle_dist: Distance to nearest forward obstacle in meters
                - collision: Collision flag (1.0 if collision occurred, 0.0 otherwise)
        """
        location = self.vehicle.get_location()
        velocity = self.vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])

        # Distance to the next global waypoint on the planned route
        next_wp_loc = self.get_next_global_waypoint_location()
        dist_to_wp = location.distance(next_wp_loc) if next_wp_loc else 0

        # Distance to the nearest obstacle detected by LiDAR sensors
        front_obstacle_dist = self.get_front_obstacle_distance()

        collision = 1.0 if self.collision_last_step else 0.0

        return [speed, dist_to_wp, front_obstacle_dist, collision]
    
    def get_waypoint_aware_state(self) -> List[float]:
        """
        Enhanced state that includes waypoint direction and navigation context (2D only).
        
        This method provides directional information about the next waypoint,
        enabling the model to learn proper steering toward navigation targets.
        
        Returns:
            List[float]: Enhanced state vector containing:
                - speed: Current vehicle speed in m/s
                - dist_to_wp: Distance to next waypoint in meters
                - angle_to_wp: Angle to waypoint relative to vehicle heading (-180° to +180°)
                - front_obstacle_dist: Distance to nearest forward obstacle in meters
                - collision: Collision flag (1.0 if collision occurred, 0.0 otherwise)
        """
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation
        velocity = self.vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
        
        # Get next waypoint
        next_wp_location = self.get_next_global_waypoint_location()
        
        if next_wp_location:
            # Calculate relative waypoint position
            dx = next_wp_location.x - vehicle_location.x
            dy = next_wp_location.y - vehicle_location.y
            distance_to_wp = np.sqrt(dx*dx + dy*dy)
            
            # Calculate angle to waypoint relative to vehicle heading
            waypoint_angle = np.arctan2(dy, dx)  # World angle to waypoint
            vehicle_yaw = np.radians(vehicle_rotation.yaw)  # Vehicle heading
            relative_angle = waypoint_angle - vehicle_yaw
            
            # Normalize angle to [-π, π]
            while relative_angle > np.pi:
                relative_angle -= 2 * np.pi
            while relative_angle < -np.pi:
                relative_angle += 2 * np.pi
            
            # Convert to normalized value [-1, 1] for model input
            relative_angle_normalized = relative_angle / np.pi
            
        else:
            distance_to_wp = 0.0
            relative_angle_normalized = 0.0
        
        # Other sensor data
        front_obstacle_dist = self.get_front_obstacle_distance()
        collision = 1.0 if self.collision_last_step else 0.0
        
        return [
            speed,                      # Vehicle speed
            distance_to_wp,             # Distance to next waypoint
            relative_angle_normalized,  # Angle to waypoint (normalized -1 to +1)
            front_obstacle_dist,        # Obstacle distance
            collision                   # Collision flag
        ]

    def get_comprehensive_state(self) -> List[float]:
        """
        Get comprehensive state with obstacle and lane awareness.
        This matches the expected 10-parameter input for CustomMLAgent.
        
        Returns:
            List[float]: [speed, dist_to_goal, angle_to_goal, dist_to_wp, angle_to_wp,
                         collision, lane_invasion, steer, throttle, brake]
        """
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation
        velocity = self.vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
        
        # Get destination
        if self.destination:
            destination = self.destination
        else:
            destination = vehicle_location
        
        # Get next waypoint (look ahead to avoid spinning)
        next_wp_location = self.get_lookahead_waypoint_location(lookahead_distance=15.0)
        if next_wp_location is None:
            next_wp_location = destination
        
        # Calculate distances
        distance_to_goal = vehicle_location.distance(destination)
        distance_to_waypoint = vehicle_location.distance(next_wp_location)
        
        # Calculate angles
        def calculate_angle(from_loc, yaw, to_loc):
            """Calculate angle to target in radians."""
            dx = to_loc.x - from_loc.x
            dy = to_loc.y - from_loc.y
            target_angle = np.arctan2(dy, dx)
            vehicle_angle = np.radians(yaw)
            angle_diff = target_angle - vehicle_angle
            
            # Normalize to [-π, π]
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            return angle_diff
        
        angle_to_goal = calculate_angle(vehicle_location, vehicle_rotation.yaw, destination)
        angle_to_waypoint = calculate_angle(vehicle_location, vehicle_rotation.yaw, next_wp_location)
        
        # Normalize state
        state = np.array([
            speed / 50.0,                           # Normalized speed (max ~50 m/s)
            distance_to_goal / 200.0,               # Normalized distance to goal
            angle_to_goal / np.pi,                  # Normalized angle to goal
            distance_to_waypoint / 50.0,            # Normalized distance to waypoint
            angle_to_waypoint / np.pi,              # Normalized angle to waypoint
            float(self.collision_data),             # Collision flag
            float(self.lane_invasion_detected),     # Lane invasion flag
            self.last_steer,                        # Previous steer
            self.last_throttle,                     # Previous throttle
            self.last_brake                         # Previous brake
        ], dtype=np.float32)
        
        return state.tolist()

    def get_temporal_state(self, history_length: int = 4) -> List[float]:
        """
        Get temporal state with history for TemporalMLAgent (14 parameters).
        
        Includes current state (10) + temporal features (4).
        
        Returns:
            List[float]: Current state + [prev_speed, prev_dist_wp, prev_angle_wp, prev_collision]
        """
        # Get current comprehensive state
        current_state = self.get_comprehensive_state()
        
        # Initialize temporal history if not exists
        if not hasattr(self, 'state_history'):
            self.state_history = []
        
        # Add current state to history
        self.state_history.append(current_state)
        
        # Keep only last N states
        if len(self.state_history) > history_length:
            self.state_history.pop(0)
        
        # Extract temporal features (use padding if history not full)
        temporal_features = []
        if len(self.state_history) >= 2:
            prev_state = self.state_history[-2]
            temporal_features = [
                prev_state[0],  # Previous speed
                prev_state[3],  # Previous distance to waypoint
                prev_state[4],  # Previous angle to waypoint
                prev_state[5],  # Previous collision
            ]
        else:
            # Padding with zeros if no history
            temporal_features = [0.0, 0.0, 0.0, 0.0]
        
        # Combine current + temporal = 10 + 4 = 14 parameters
        return current_state + temporal_features

    def get_next_global_waypoint_location(self) -> Optional[carla.Location]:
        """
        Get the location of the next waypoint with aggressive waypoint purging.
        Automatically removes reached waypoints from the route.
        Uses larger thresholds to prevent vehicles from spinning in circles.
        
        Returns:
            Optional[carla.Location]: Location of the next waypoint, or None if no route
        """
        if not hasattr(self, 'route') or not self.route:
            return None
        
        vehicle_location = self.vehicle.get_location()
        vehicle_velocity = self.vehicle.get_velocity()
        vehicle_speed = np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)
        
        # FIXED LARGER THRESHOLD to prevent spinning around waypoints
        # Use 15m base threshold (was 8m) to prevent vehicles targeting waypoints underneath
        # At 0 m/s: 15.0m threshold
        # At 10 m/s: 18.0m threshold  
        # At 20 m/s: 21.0m threshold
        base_threshold = 15.0  # Large base to prevent circular motion
        adaptive_threshold = base_threshold + 0.3 * vehicle_speed
        
        # AGGRESSIVE WAYPOINT PURGING: Remove all reached waypoints
        waypoints_removed = 0
        while self.route:
            waypoint = self.route[0][0]
            wp_location = waypoint.transform.location
            
            # Calculate 2D distance only (ignore Z-axis)
            dx = wp_location.x - vehicle_location.x
            dy = wp_location.y - vehicle_location.y
            distance_2d = np.sqrt(dx*dx + dy*dy)
            
            # If waypoint is reached (within adaptive threshold), remove it
            if distance_2d < adaptive_threshold:
                self.route.pop(0)
                waypoints_removed += 1
            else:
                # First unreached waypoint is our target
                if waypoints_removed > 0:
                    self.logger.debug(f"🎯 Removed {waypoints_removed} waypoint(s) (speed: {vehicle_speed:.1f}m/s, threshold: {adaptive_threshold:.1f}m). {len(self.route)} remaining")
                return wp_location
        
        # All waypoints reached
        if waypoints_removed > 0:
            self.logger.info(f"✅ All waypoints reached!")
        return None
    
    def get_lookahead_waypoint_location(self, lookahead_distance: float = 15.0) -> Optional[carla.Location]:
        """
        Get waypoint location at specified distance ahead for curve anticipation.
        
        Args:
            lookahead_distance: Distance to look ahead in meters (default 15m)
        
        Returns:
            Optional[carla.Location]: Location of lookahead waypoint, or None if route too short
        """
        if not hasattr(self, 'route') or not self.route:
            return None
        
        vehicle_location = self.vehicle.get_location()
        cumulative_distance = 0.0
        prev_location = vehicle_location
        
        # Traverse route and accumulate distance
        for waypoint, _ in self.route:
            wp_location = waypoint.transform.location
            segment_distance = np.sqrt(
                (wp_location.x - prev_location.x)**2 + 
                (wp_location.y - prev_location.y)**2
            )
            cumulative_distance += segment_distance
            
            if cumulative_distance >= lookahead_distance:
                return wp_location
            
            prev_location = wp_location
        
        # Route is shorter than lookahead distance - return last waypoint
        if self.route:
            return self.route[-1][0].transform.location
        
        return None
        
    def get_front_obstacle_distance(self, max_distance: float = 50.0) -> float:
        """
        Calculate distance to the nearest obstacle in front using raycasting.
        
        Uses multiple raycasts in a forward cone to detect obstacles including:
        - Other vehicles
        - Walls and buildings
        - Road boundaries
        
        Args:
            max_distance (float): Maximum detection range in meters. Defaults to 50.0.
        
        Returns:
            float: Distance to nearest forward obstacle in meters, or max_distance
                   if no obstacles detected
        """
        import math
        
        try:
            transform = self.vehicle.get_transform()
            location = transform.location
            forward_vector = transform.get_forward_vector()
            
            min_obstacle_distance = max_distance
            
            # Cast multiple rays in a forward cone (center, slight left/right)
            angles = [0, -10, 10, -20, 20]  # degrees from center
            
            for angle in angles:
                # Rotate forward vector by angle
                rad = math.radians(angle)
                ray_x = forward_vector.x * math.cos(rad) - forward_vector.y * math.sin(rad)
                ray_y = forward_vector.x * math.sin(rad) + forward_vector.y * math.cos(rad)
                
                # End point of ray
                end_location = carla.Location(
                    x=location.x + ray_x * max_distance,
                    y=location.y + ray_y * max_distance,
                    z=location.z + 0.5
                )
                
                # Cast ray and check for collision
                result = self.world.cast_ray(
                    carla.Location(location.x, location.y, location.z + 0.5),
                    end_location
                )
                
                if result:  # Hit something
                    hit_location = result[0].location
                    distance = location.distance(hit_location)
                    min_obstacle_distance = min(min_obstacle_distance, distance)
            
            # Also check for nearby vehicles using actor list
            nearby_vehicles = self.world.get_actors().filter('vehicle.*')
            for other_vehicle in nearby_vehicles:
                if other_vehicle.id == self.vehicle.id:
                    continue
                
                other_location = other_vehicle.get_location()
                
                # Check if vehicle is roughly ahead (within 90 degrees)
                to_other = np.array([other_location.x - location.x, other_location.y - location.y])
                forward = np.array([forward_vector.x, forward_vector.y])
                
                if np.dot(to_other, forward) > 0:  # In front
                    distance = location.distance(other_location)
                    if distance < max_distance:
                        min_obstacle_distance = min(min_obstacle_distance, distance)
            
            return min_obstacle_distance
            
        except Exception as e:
            # Fallback to LiDAR if available
            if hasattr(self, 'lidar_data') and self.lidar_data is not None:
                try:
                    points = self.lidar_data
                    if len(points) > 0:
                        forward_points = points[points[:, 0] > 0]
                        if len(forward_points) > 0:
                            distances = np.linalg.norm(forward_points, axis=1)
                            min_distance = np.min(distances)
                            return min_distance if min_distance <= max_distance else max_distance
                except:
                    pass
            
            return max_distance


    def run_step(self) -> None:
        """
        Execute one control step with comprehensive state.
        
        Determines which state representation to use based on model type,
        then applies the model's predicted control actions.
        """
        # Determine which state to use based on model type
        if hasattr(self.model, '__class__'):
            model_name = self.model.__class__.__name__
            if 'Temporal' in model_name:
                state = self.get_temporal_state()
            else:
                state = self.get_comprehensive_state()
        else:
            state = self.get_comprehensive_state()
        
        # Get control actions from ML model
        self.last_steer, self.last_throttle, self.last_brake = self.steer, self.throttle, self.brake
        
        self.steer, self.throttle, self.brake = self.model.predict(state)
        
        # Log model predictions at DEBUG level
        log_prediction_debug(self.logger, self.model.__class__.__name__, 
                           f"Steer: {self.steer:.3f}, Throttle: {self.throttle:.3f}, Brake: {self.brake:.3f}")
        
        # Create and apply vehicle control
        # Let the model learn speed control naturally through rewards
        control = VehicleControl()
        control.steer = float(np.clip(self.steer, -0.4, 0.4))  # Cap steering to prevent tight circles
        control.throttle = float(np.clip(self.throttle, 0.0, 1.0))
        control.brake = float(np.clip(self.brake, 0.0, 1.0))
        
        self.vehicle.apply_control(control)
        
        # Monitor vehicle speed to detect if stuck
        speed = state[0] * 50.0  # Denormalize speed
        speed_threshold = 1.0  # Minimum speed threshold in m/s
        
        # Update low-speed counter for stuck detection
        if speed < speed_threshold:
            self.low_speed_counter += 1
        else:
            self.low_speed_counter = 0
            
        self.yaw = self.vehicle.get_transform().rotation.yaw
        
    def run_step_correction(self):
        # Ottieni comandi base dal local planner
        local_control = self.local_planner.run_step()

        # Stato arricchito da sensori e posizione rispetto alla rotta
        state = self.get_rich_state()

        # Predizione modello ML (correzioni da applicare sul controllo base)
        steer_correction, throttle_correction, brake_correction = self.model.predict(state)

        control = carla.VehicleControl()
        # Mix delle azioni (esempio somma bilanciata)
        control.steer = local_control.steer + steer_correction
        control.throttle = max(0, min(1, local_control.throttle + throttle_correction))
        control.brake = max(0, min(1, local_control.brake + brake_correction))

        self.vehicle.apply_control(control)


    def destroy(self) -> None:
        """
        Clean up all sensor actors to prevent resource leaks.
        
        This method should be called when the agent is no longer needed
        to properly dispose of sensor actors and free up simulator resources.
        
        Destroys:
        - RGB camera sensor
        - Collision detection sensor
        - LiDAR sensor
        """
        # Prevent double destruction
        if hasattr(self, '_destroyed') and self._destroyed:
            return
        
        try:
            # Stop and destroy sensors
            if hasattr(self, 'camera') and self.camera is not None:
                try:
                    if hasattr(self.camera, 'is_listening') and self.camera.is_listening:
                        self.camera.stop()
                except:
                    pass
                try:
                    self.camera.destroy()
                except:
                    pass
                self.camera = None
                
            if hasattr(self, 'collision_sensor') and self.collision_sensor is not None:
                try:
                    if hasattr(self.collision_sensor, 'is_listening') and self.collision_sensor.is_listening:
                        self.collision_sensor.stop()
                except:
                    pass
                try:
                    self.collision_sensor.destroy()
                except:
                    pass
                self.collision_sensor = None
            
            # Destroy lane invasion sensor
            if hasattr(self, 'lane_invasion_sensor') and self.lane_invasion_sensor is not None:
                try:
                    if hasattr(self.lane_invasion_sensor, 'is_listening') and self.lane_invasion_sensor.is_listening:
                        self.lane_invasion_sensor.stop()
                except:
                    pass
                try:
                    self.lane_invasion_sensor.destroy()
                except:
                    pass
                self.lane_invasion_sensor = None
                
            if hasattr(self, 'lidar_sensor') and self.lidar_sensor is not None:
                try:
                    if hasattr(self.lidar_sensor, 'is_listening') and self.lidar_sensor.is_listening:
                        self.lidar_sensor.stop()
                except:
                    pass
                try:
                    self.lidar_sensor.destroy()
                except:
                    pass
                self.lidar_sensor = None
                
        except Exception as e:
            pass  # Silently handle cleanup errors
            
        # Clear sensor data references
        self.camera_data = None
        self.lidar_data = None
        self._destroyed = True
    
    def reset_sensors(self) -> None:
        """
        Reset all sensor data and flags to initial state.
        
        Useful for starting new episodes or resetting the agent state
        during training without recreating sensor actors.
        """
        self.camera_data = None
        self.collision_data = False
        self.collision_last_step = False
        self.lane_invasion_detected = False
        self.lidar_data = None
        self.low_speed_counter = 0
        
        # Clear temporal history
        if hasattr(self, 'state_history'):
            self.state_history = []
    
    def reset_for_new_episode(self) -> None:
        """
        Reset agent state for a new training episode.
        
        This method should be called when starting a new episode,
        especially after moving the vehicle to a new spawn point.
        """
        self.logger.info("🔄 Resetting agent for new episode")
        
        # Reset sensor data and counters
        self.reset_sensors()
        
        # Update navigation for new location
        self.update_navigation_for_new_spawn()
        
        # Verify navigation setup
        next_wp = self.get_next_global_waypoint_location()
        if next_wp:
            current_loc = self.vehicle.get_location()
            distance = current_loc.distance(next_wp)
            self.logger.info(f"🎯 Navigation verified - Next waypoint at {distance:.1f}m")
        else:
            self.logger.warning("⚠️  Navigation setup failed - no waypoints available")
        
        # Reset control variables
        self.steer, self.throttle, self.brake, self.yaw = [0.0] * 4
        self.last_steer, self.last_throttle, self.last_brake, self.last_yaw = [0.0] * 4
        
        # Reset collision and performance tracking
        self.collision_data = False
        self.collision_last_step = False
        self.low_speed_counter = 0
        
        # Reset reward calculation tracking
        if hasattr(self, 'previous_distance_to_destination'):
            del self.previous_distance_to_destination
    
    def get_sensor_status(self) -> dict:
        """
        Get current status of all sensors for debugging and monitoring.
        
        Returns:
            dict: Sensor status information including:
                - camera_active: Whether camera data is being received
                - lidar_active: Whether LiDAR data is being received  
                - collision_detected: Current collision status
                - low_speed_steps: Number of consecutive low-speed steps
        """
        return {
            'camera_active': self.camera_data is not None,
            'lidar_active': self.lidar_data is not None,
            'collision_detected': self.collision_data,
            'low_speed_steps': self.low_speed_counter
        }
    
    def get_next_waypoint_location(self):
        if hasattr(self.local_planner, '_global_plan') and self.local_planner._global_plan:
            return self.local_planner._global_plan[0][0].transform.location
        else:
            return self.vehicle.get_location()  # fallback

