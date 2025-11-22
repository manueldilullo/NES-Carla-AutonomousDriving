"""
HUD (Heads-Up Display) for CARLA Training Visualization

This module provides a simple text-based HUD that displays real-time information
about the vehicle's position, destination, and navigation status during training.

Features:
- Vehicle coordinates (X, Y, Z)
- Destination coordinates
- Next waypoint position
- Distance to destination
- Current speed
- Episode step counter
- Generation/fitness information
- Opens in separate console window

Author: Manuel Di Lullo
Date: 2025
Dependencies: CARLA Python API
"""

import carla
from typing import Optional
import math
import os
import sys
import subprocess


class TrainingHUD:
    """
    Console-based HUD that opens in a separate terminal window.
    
    Displays vehicle telemetry and navigation information in real-time
    in a dedicated console window separate from the main training output.
    """
    
    def __init__(self, world: carla.World):
        """
        Initialize the HUD and open a separate console window.
        
        Args:
            world: CARLA world instance
        """
        self.world = world
        # Use absolute path for HUD file
        self.hud_file = os.path.abspath("training_hud_display.txt")
        self.terminal_process = None
        
        # Create initial HUD file
        with open(self.hud_file, 'w', encoding='utf-8') as f:
            f.write("╔════════════════════════════════════════════════════════════════════╗\n")
            f.write("║           Training HUD Initializing...                            ║\n")
            f.write("║           Waiting for first update...                             ║\n")
            f.write("╚════════════════════════════════════════════════════════════════════╝\n")
        
        print(f"HUD file created at: {self.hud_file}")
        
        # Open separate terminal window based on OS
        self._open_terminal_window()
    
    def _open_terminal_window(self):
        """Open a separate terminal window to display HUD."""
        # For Git Bash users: Just inform them about the file location
        # They can manually open it or use: watch -n 1 cat training_hud_display.txt
        print(f"\n{'='*70}")
        print(f"HUD file location: {self.hud_file}")
        print(f"To view HUD in real-time, open a new terminal and run:")
        print(f"  Windows CMD: for /L %i in (0,0,1) do @(cls && type \"{self.hud_file}\" && timeout /t 1 /nobreak > nul)")
        print(f"  Git Bash:    while true; do clear; cat '{self.hud_file}'; sleep 1; done")
        print(f"Or simply open the file in a text editor and refresh it")
        print(f"{'='*70}\n")
        
        # Don't try to auto-open from Git Bash - it causes path issues
        self.terminal_process = None
    
    def draw(self, 
             vehicle: carla.Vehicle,
             destination: Optional[carla.Location] = None,
             next_waypoint: Optional[carla.Location] = None,
             generation: int = 0,
             episode_step: int = 0,
             fitness: float = 0.0,
             additional_info: dict = None) -> None:
        """
        Update HUD display file with current information.
        
        Args:
            vehicle: CARLA vehicle instance
            destination: Target destination location
            next_waypoint: Next waypoint location on route
            generation: Current training generation
            episode_step: Current step in episode
            fitness: Current fitness/reward value
            additional_info: Dictionary with extra information to display
        """
        # Get vehicle information
        vehicle_transform = vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # m/s
        
        # Calculate distances
        distance_to_dest = 0.0
        waypoint_dist = 0.0
        
        if destination:
            distance_to_dest = vehicle_location.distance(destination)
        
        if next_waypoint:
            waypoint_dist = vehicle_location.distance(next_waypoint)
        
        # Build HUD display
        lines = []
        lines.append("╔" + "═" * 68 + "╗")
        lines.append(f"║ 🎮 TRAINING HUD - Generation {generation:3d} │ Step {episode_step:3d}" + " " * 17 + "║")
        lines.append("╠" + "═" * 68 + "╣")
        
        # Vehicle info
        lines.append(f"║ 🚗 Vehicle Pos: ({vehicle_location.x:7.1f}, {vehicle_location.y:7.1f}, {vehicle_location.z:5.1f})" + " " * 10 + "║")
        lines.append(f"║ 💨 Speed: {speed:5.1f} m/s ({speed*3.6:5.1f} km/h)" + " " * 28 + "║")
        
        # Navigation info
        if destination:
            lines.append("╟" + "─" * 68 + "╢")
            lines.append(f"║ 🎯 Destination: ({destination.x:7.1f}, {destination.y:7.1f}, {destination.z:5.1f})" + " " * 6 + "║")
            lines.append(f"║ 📏 Distance to Goal: {distance_to_dest:6.1f} m" + " " * 36 + "║")
        
        if next_waypoint:
            lines.append(f"║ 📍 Next Waypoint: ({next_waypoint.x:7.1f}, {next_waypoint.y:7.1f}, {next_waypoint.z:5.1f})" + " " * 3 + "║")
            lines.append(f"║ 📏 WP Distance: {waypoint_dist:6.1f} m" + " " * 40 + "║")
        
        # Training metrics
        lines.append("╟" + "─" * 68 + "╢")
        lines.append(f"║ 💯 Fitness: {fitness:8.1f}" + " " * 48 + "║")
        
        # Additional info
        if additional_info:
            for key, value in additional_info.items():
                if isinstance(value, float):
                    text = f"║ ➤ {key}: {value:6.3f}"
                else:
                    text = f"║ ➤ {key}: {value}"
                lines.append(text + " " * (70 - len(text)) + "║")
        
        lines.append("╚" + "═" * 68 + "╝")
        
        # Write to HUD file
        try:
            with open(self.hud_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))
                f.flush()  # Force write to disk
        except Exception as e:
            print(f"Warning: HUD file write error: {e}")
    
    def destroy(self):
        """Clean up HUD resources and close terminal window."""
        # Terminate terminal process
        if self.terminal_process is not None:
            try:
                self.terminal_process.terminate()
            except:
                pass
        
        # Remove HUD file
        try:
            if os.path.exists(self.hud_file):
                os.remove(self.hud_file)
        except:
            pass


def create_hud(world: carla.World) -> TrainingHUD:
    """
    Factory function to create a HUD instance.
    
    Args:
        world: CARLA world instance
    
    Returns:
        TrainingHUD: Initialized HUD object with separate console window
    """
    return TrainingHUD(world)

