# Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Improved Global Route Planner with caching and lane filtering.

Enhancements:
- Topology graph caching for faster initialization
- Non-driving lane filtering
- Direction awareness for better routing
- Memory optimization

Author: Manuel Di Lullo (improvements)
Date: 2025
"""

import math
import numpy as np
import networkx as nx

import carla
from navigation.local_planner import RoadOption
from utils.misc import vector

# Module-level cache for topology graphs
_TOPOLOGY_CACHE = {}

class ImprovedGlobalRoutePlanner(object):
    """
    Improved high-level route planner with caching and lane awareness.
    """

    def __init__(self, wmap, sampling_resolution):
        self._sampling_resolution = sampling_resolution
        self._wmap = wmap
        self._topology = None
        self._graph = None
        self._id_map = None
        self._road_id_to_edge = None

        self._intersection_end_node = -1
        self._previous_decision = RoadOption.VOID

        # Use cached topology if available
        cache_key = f"{id(wmap)}_{sampling_resolution}"
        
        if cache_key in _TOPOLOGY_CACHE:
            cached_data = _TOPOLOGY_CACHE[cache_key]
            self._topology = cached_data['topology']
            self._id_map = cached_data['id_map']
            self._road_id_to_edge = cached_data['road_id_to_edge']
            print(f"✅ Using cached topology ({len(self._topology)} segments)")
        else:
            print(f"🔨 Building improved topology (sampling_resolution={sampling_resolution})...")
            # Build the graph with improvements
            self._build_topology()
            
            # Cache the topology data
            _TOPOLOGY_CACHE[cache_key] = {
                'topology': self._topology,
                'id_map': self._id_map,
                'road_id_to_edge': self._road_id_to_edge
            }
            print(f"✅ Topology cached ({len(self._topology)} segments)")
        
        # Always rebuild graph (lightweight operation)
        self._build_graph()
        self._find_loose_ends()
        self._lane_change_link()

    def trace_route(self, origin, destination):
        """
        Returns list of (carla.Waypoint, RoadOption) from origin to destination.
        """
        route_trace = []
        route = self._path_search(origin, destination)
        current_waypoint = self._wmap.get_waypoint(origin)
        destination_waypoint = self._wmap.get_waypoint(destination)
        
        # Store vehicle's Z coordinate to fix waypoint elevation
        vehicle_z = origin.z

        for i in range(len(route) - 1):
            road_option = self._turn_decision(i, route)
            edge = self._graph.edges[route[i], route[i+1]]
            path = []

            if edge['type'] != RoadOption.LANEFOLLOW and edge['type'] != RoadOption.VOID:
                route_trace.append((current_waypoint, road_option))
                exit_wp = edge['exit_waypoint']
                n1, n2 = self._road_id_to_edge[exit_wp.road_id][exit_wp.section_id][exit_wp.lane_id]
                next_edge = self._graph.edges[n1, n2]
                if next_edge['path']:
                    closest_index = self._find_closest_in_list(current_waypoint, next_edge['path'])
                    closest_index = min(len(next_edge['path'])-1, closest_index+5)
                    current_waypoint = next_edge['path'][closest_index]
                else:
                    current_waypoint = next_edge['exit_waypoint']
                route_trace.append((current_waypoint, road_option))

            else:
                path = path + [edge['entry_waypoint']] + edge['path'] + [edge['exit_waypoint']]
                closest_index = self._find_closest_in_list(current_waypoint, path)
                for waypoint in path[closest_index:]:
                    current_waypoint = waypoint
                    # Fix Z-coordinate to match vehicle elevation
                    corrected_location = carla.Location(
                        x=current_waypoint.transform.location.x,
                        y=current_waypoint.transform.location.y,
                        z=vehicle_z
                    )
                    corrected_waypoint = self._wmap.get_waypoint(corrected_location)
                    route_trace.append((corrected_waypoint, road_option))
                    if len(route)-i <= 2 and waypoint.transform.location.distance(destination) < 2*self._sampling_resolution:
                        break
                    elif len(route)-i <= 2 and current_waypoint.road_id == destination_waypoint.road_id and current_waypoint.section_id == destination_waypoint.section_id and current_waypoint.lane_id == destination_waypoint.lane_id:
                        destination_index = self._find_closest_in_list(destination_waypoint, path)
                        if closest_index > destination_index:
                            break

        return route_trace

    def _is_valid_driving_lane(self, waypoint):
        """
        Check if waypoint is on a valid driving lane.
        
        Filters out:
        - Non-driving lanes (sidewalks, shoulders, parking, etc.)
        - Lanes with lane_id == 0 (invalid)
        - Restricted lanes
        
        Returns:
            bool: True if valid driving lane, False otherwise
        """
        # Must be driving lane type
        if waypoint.lane_type != carla.LaneType.Driving:
            return False
        
        # Lane ID must be non-zero
        if waypoint.lane_id == 0:
            return False
        
        # Additional check: not on shoulder or restricted
        if hasattr(waypoint, 'lane_type'):
            invalid_types = [
                carla.LaneType.Shoulder,
                carla.LaneType.Sidewalk,
                carla.LaneType.Parking,
                carla.LaneType.Rail
            ]
            if waypoint.lane_type in invalid_types:
                return False
        
        return True

    def _is_compatible_direction(self, wp1, wp2):
        """
        Check if two waypoints have compatible driving directions.
        
        Args:
            wp1: First waypoint
            wp2: Second waypoint
        
        Returns:
            bool: True if directions are compatible (same or junction)
        """
        # At junctions, allow all directions
        if wp1.is_junction or wp2.is_junction:
            return True
        
        # Check if yaw angles are similar (within 45 degrees)
        yaw1 = wp1.transform.rotation.yaw
        yaw2 = wp2.transform.rotation.yaw
        
        # Normalize yaw difference to [-180, 180]
        yaw_diff = abs(yaw1 - yaw2)
        if yaw_diff > 180:
            yaw_diff = 360 - yaw_diff
        
        # Allow up to 45 degrees difference (gentle curves OK)
        return yaw_diff < 45.0

    def _build_topology(self):
        """
        Build topology with lane filtering for driving lanes only.
        """
        self._topology = []
        self._id_map = dict()
        self._road_id_to_edge = dict()
        
        # Get topology from map
        raw_topology = self._wmap.get_topology()
        
        filtered_count = 0
        for segment in raw_topology:
            wp1, wp2 = segment[0], segment[1]
            
            # Filter: Only include valid driving lanes
            if not self._is_valid_driving_lane(wp1) or not self._is_valid_driving_lane(wp2):
                filtered_count += 1
                continue
            
            # Filter: Check direction compatibility
            if not self._is_compatible_direction(wp1, wp2):
                filtered_count += 1
                continue
            
            l1, l2 = wp1.transform.location, wp2.transform.location
            # Rounding off to avoid floating point imprecision
            x1, y1, z1 = np.round([l1.x, l1.y, l1.z], 0)
            x2, y2, z2 = np.round([l2.x, l2.y, l2.z], 0)
            
            seg_dict = dict()
            seg_dict['entry'], seg_dict['exit'] = wp1, wp2
            seg_dict['entryxyz'], seg_dict['exitxyz'] = (x1, y1, z1), (x2, y2, z2)
            seg_dict['path'] = []
            
            endloc = wp2.transform.location
            if wp1.transform.location.distance(endloc) > self._sampling_resolution:
                next_wps = wp1.next(self._sampling_resolution)
                if len(next_wps) > 0:
                    w = next_wps[0]
                    while w.transform.location.distance(endloc) > self._sampling_resolution:
                        # Only add waypoints on valid driving lanes
                        if self._is_valid_driving_lane(w):
                            seg_dict['path'].append(w)
                        next_ws = w.next(self._sampling_resolution)
                        if len(next_ws) == 0:
                            break
                        w = next_ws[0]
            else:
                next_wps = wp1.next(self._sampling_resolution)
                if len(next_wps) > 0 and self._is_valid_driving_lane(next_wps[0]):
                    seg_dict['path'].append(next_wps[0])
            
            self._topology.append(seg_dict)
        
        if filtered_count > 0:
            print(f"  Filtered {filtered_count} non-driving lane segments")

    def _build_graph(self):
        """
        Build networkx graph from topology with direction awareness.
        """
        self._graph = nx.DiGraph()

        for segment in self._topology:
            entry_xyz, exit_xyz = segment['entryxyz'], segment['exitxyz']
            path = segment['path']
            entry_wp, exit_wp = segment['entry'], segment['exit']
            intersection = entry_wp.is_junction
            road_id, section_id, lane_id = entry_wp.road_id, entry_wp.section_id, entry_wp.lane_id

            for vertex in entry_xyz, exit_xyz:
                # Adding unique nodes and populating id_map
                if vertex not in self._id_map:
                    new_id = len(self._id_map)
                    self._id_map[vertex] = new_id
                    self._graph.add_node(new_id, vertex=vertex)
            
            n1 = self._id_map[entry_xyz]
            n2 = self._id_map[exit_xyz]
            
            if road_id not in self._road_id_to_edge:
                self._road_id_to_edge[road_id] = dict()
            if section_id not in self._road_id_to_edge[road_id]:
                self._road_id_to_edge[road_id][section_id] = dict()
            self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)

            entry_carla_vector = entry_wp.transform.rotation.get_forward_vector()
            exit_carla_vector = exit_wp.transform.rotation.get_forward_vector()

            # Adding edge with attributes
            self._graph.add_edge(
                n1, n2,
                length=len(path) + 1, path=path,
                entry_waypoint=entry_wp, exit_waypoint=exit_wp,
                entry_vector=np.array(
                    [entry_carla_vector.x, entry_carla_vector.y, entry_carla_vector.z]),
                exit_vector=np.array(
                    [exit_carla_vector.x, exit_carla_vector.y, exit_carla_vector.z]),
                net_vector=vector(entry_wp.transform.location, exit_wp.transform.location),
                intersection=intersection, type=RoadOption.LANEFOLLOW)

    def _find_loose_ends(self):
        """
        Find and connect unconnected road segments.
        """
        count_loose_ends = 0
        hop_resolution = self._sampling_resolution
        for segment in self._topology:
            end_wp = segment['exit']
            exit_xyz = segment['exitxyz']
            road_id, section_id, lane_id = end_wp.road_id, end_wp.section_id, end_wp.lane_id
            if road_id in self._road_id_to_edge \
                    and section_id in self._road_id_to_edge[road_id] \
                    and lane_id in self._road_id_to_edge[road_id][section_id]:
                pass
            else:
                count_loose_ends += 1
                if road_id not in self._road_id_to_edge:
                    self._road_id_to_edge[road_id] = dict()
                if section_id not in self._road_id_to_edge[road_id]:
                    self._road_id_to_edge[road_id][section_id] = dict()
                n1 = self._id_map[exit_xyz]
                n2 = -1*count_loose_ends
                self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)
                next_wp = end_wp.next(hop_resolution)
                path = []
                while next_wp is not None and next_wp \
                        and next_wp[0].road_id == road_id \
                        and next_wp[0].section_id == section_id \
                        and next_wp[0].lane_id == lane_id:
                    # Only add valid driving lanes
                    if self._is_valid_driving_lane(next_wp[0]):
                        path.append(next_wp[0])
                    next_wp = next_wp[0].next(hop_resolution)
                if path:
                    n2_xyz = (path[-1].transform.location.x,
                              path[-1].transform.location.y,
                              path[-1].transform.location.z)
                    self._graph.add_node(n2, vertex=n2_xyz)
                    self._graph.add_edge(
                        n1, n2,
                        length=len(path) + 1, path=path,
                        entry_waypoint=end_wp, exit_waypoint=path[-1],
                        entry_vector=None, exit_vector=None, net_vector=None,
                        intersection=end_wp.is_junction, type=RoadOption.LANEFOLLOW)

    def _lane_change_link(self):
        """
        Add zero-cost links for lane changes (driving lanes only).
        """
        for segment in self._topology:
            left_found, right_found = False, False

            for waypoint in segment['path']:
                if not segment['entry'].is_junction:
                    next_waypoint, next_road_option, next_segment = None, None, None

                    if waypoint.right_lane_marking and waypoint.right_lane_marking.lane_change & carla.LaneChange.Right and not right_found:
                        next_waypoint = waypoint.get_right_lane()
                        if next_waypoint is not None \
                                and next_waypoint.lane_type == carla.LaneType.Driving \
                                and waypoint.road_id == next_waypoint.road_id \
                                and self._is_valid_driving_lane(next_waypoint):
                            next_road_option = RoadOption.CHANGELANERIGHT
                            next_segment = self._localize(next_waypoint.transform.location)
                            if next_segment is not None:
                                self._graph.add_edge(
                                    self._id_map[segment['entryxyz']], next_segment[0], entry_waypoint=waypoint,
                                    exit_waypoint=next_waypoint, intersection=False, exit_vector=None,
                                    path=[], length=0, type=next_road_option, change_waypoint=next_waypoint)
                                right_found = True
                    
                    if waypoint.left_lane_marking and waypoint.left_lane_marking.lane_change & carla.LaneChange.Left and not left_found:
                        next_waypoint = waypoint.get_left_lane()
                        if next_waypoint is not None \
                                and next_waypoint.lane_type == carla.LaneType.Driving \
                                and waypoint.road_id == next_waypoint.road_id \
                                and self._is_valid_driving_lane(next_waypoint):
                            next_road_option = RoadOption.CHANGELANELEFT
                            next_segment = self._localize(next_waypoint.transform.location)
                            if next_segment is not None:
                                self._graph.add_edge(
                                    self._id_map[segment['entryxyz']], next_segment[0], entry_waypoint=waypoint,
                                    exit_waypoint=next_waypoint, intersection=False, exit_vector=None,
                                    path=[], length=0, type=next_road_option, change_waypoint=next_waypoint)
                                left_found = True
                if left_found and right_found:
                    break

    def _localize(self, location):
        """
        Find the road segment that contains a given location.
        """
        waypoint = self._wmap.get_waypoint(location)
        edge = None
        try:
            edge = self._road_id_to_edge[waypoint.road_id][waypoint.section_id][waypoint.lane_id]
        except KeyError:
            pass
        return edge

    def _distance_heuristic(self, n1, n2):
        """
        Distance heuristic for A* search.
        """
        l1 = np.array(self._graph.nodes[n1]['vertex'])
        l2 = np.array(self._graph.nodes[n2]['vertex'])
        return np.linalg.norm(l1-l2)

    def _path_search(self, origin, destination):
        """
        Find shortest path using A* with distance heuristic.
        Raises informative exception if no path exists.
        """
        start = self._localize(origin)
        end = self._localize(destination)
        
        if start is None:
            raise ValueError(f"Cannot localize origin at ({origin.x:.1f}, {origin.y:.1f}, {origin.z:.1f})")
        if end is None:
            raise ValueError(f"Cannot localize destination at ({destination.x:.1f}, {destination.y:.1f}, {destination.z:.1f})")
        
        try:
            route = nx.astar_path(
                self._graph, source=start[0], target=end[0],
                heuristic=self._distance_heuristic, weight='length')
            route.append(end[1])
            return route
        except nx.NetworkXNoPath:
            raise ValueError(f"No path exists between start node {start[0]} and end node {end[0]} (disconnected graph regions)")
        except nx.NodeNotFound as e:
            raise ValueError(f"Node {e} not found in graph - route planning failed")

    def _successive_last_intersection_edge(self, index, route):
        """
        Returns the last successive intersection edge from a starting index.
        """
        last_intersection_edge = None
        last_node = None
        for node1, node2 in [(route[i], route[i+1]) for i in range(index, len(route)-1)]:
            candidate_edge = self._graph.edges[node1, node2]
            if node1 == route[index]:
                last_intersection_edge = candidate_edge
            if candidate_edge['type'] == RoadOption.LANEFOLLOW and candidate_edge['intersection']:
                last_intersection_edge = candidate_edge
                last_node = node2
            else:
                break

        return last_node, last_intersection_edge

    def _turn_decision(self, index, route, threshold=math.radians(35)):
        """
        Returns turn decision (RoadOption) for pair of edges around current index.
        """
        decision = None
        previous_node = route[index-1]
        current_node = route[index]
        next_node = route[index+1]
        next_edge = self._graph.edges[current_node, next_node]
        if index > 0:
            if self._previous_decision != RoadOption.VOID \
                    and self._intersection_end_node > 0 \
                    and self._intersection_end_node != previous_node \
                    and next_edge['type'] == RoadOption.LANEFOLLOW \
                    and next_edge['intersection']:
                decision = self._previous_decision
            else:
                self._intersection_end_node = -1
                current_edge = self._graph.edges[previous_node, current_node]
                calculate_turn = current_edge['type'] == RoadOption.LANEFOLLOW and not current_edge[
                    'intersection'] and next_edge['type'] == RoadOption.LANEFOLLOW and next_edge['intersection']
                if calculate_turn:
                    last_node, tail_edge = self._successive_last_intersection_edge(index, route)
                    self._intersection_end_node = last_node
                    if tail_edge is not None:
                        next_edge = tail_edge
                    cv, nv = current_edge['exit_vector'], next_edge['exit_vector']
                    if cv is None or nv is None:
                        return next_edge['type']
                    cross_list = []
                    for neighbor in self._graph.successors(current_node):
                        select_edge = self._graph.edges[current_node, neighbor]
                        if select_edge['type'] == RoadOption.LANEFOLLOW:
                            if neighbor != route[index+1]:
                                sv = select_edge['net_vector']
                                cross_list.append(np.cross(cv, sv)[2])
                    next_cross = np.cross(cv, nv)[2]
                    deviation = math.acos(np.clip(
                        np.dot(cv, nv)/(np.linalg.norm(cv)*np.linalg.norm(nv)), -1.0, 1.0))
                    if not cross_list:
                        cross_list.append(0)
                    if deviation < threshold:
                        decision = RoadOption.STRAIGHT
                    elif cross_list and next_cross < min(cross_list):
                        decision = RoadOption.LEFT
                    elif cross_list and next_cross > max(cross_list):
                        decision = RoadOption.RIGHT
                    elif next_cross < 0:
                        decision = RoadOption.LEFT
                    elif next_cross > 0:
                        decision = RoadOption.RIGHT
                else:
                    decision = next_edge['type']

        else:
            decision = next_edge['type']

        self._previous_decision = decision
        return decision

    def _find_closest_in_list(self, current_waypoint, waypoint_list):
        """
        Find closest waypoint in list to current waypoint.
        """
        min_distance = float('inf')
        closest_index = -1
        for i, waypoint in enumerate(waypoint_list):
            distance = waypoint.transform.location.distance(
                current_waypoint.transform.location)
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        return closest_index
