"""
Training Engine - Common training logic for all training modes

Handles CARLA connection, vehicle setup, agent initialization, and evaluation.
Follows DRY principle to eliminate code duplication across training scripts.

Author: Manuel Di Lullo
Date: 2025
"""

import time
import carla
import random
import numpy as np
from typing import Optional, Tuple

from config import Config
from models.custom_ml_model import CustomMLModel
from models.simple_temporal_model import SimpleTemporalModel
from agents.nes_agent import NES
from train_temporal import TemporalMLAgent
from agents.custom_ml_agent import CustomAdvancedMLAgent
from utils.actors_handler import spawn_vehicles_and_pedestrians, cleanup_actors
from utils.hud import create_hud
from training.training_utils import reset_vehicle_physics


class TrainingEngine:
    """
    Manages common training infrastructure and logic.
    
    Responsibilities:
    - CARLA connection and world setup
    - Vehicle spawning and management
    - Agent initialization
    - Common evaluation logic
    """
    
    def __init__(self, config: Config):
        """
        Initialize training engine with configuration.
        
        Args:
            config: Complete training configuration
        """
        self.config = config
        self.client = None
        self.world = None
        self.vehicle = None
        self.agent = None
        self.nes = None
        self.hud = None
        self.npc_vehicles = []
        self.npc_pedestrians = []
        
    def connect_to_carla(self, host: str = '127.0.0.1', port: int = 2000, timeout: float = 10.0) -> bool:
        """
        Connect to CARLA simulator.
        
        Args:
            host: CARLA server hostname
            port: CARLA server port
            timeout: Connection timeout in seconds
            
        Returns:
            bool: True if connection successful
        """
        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(timeout)
            self.world = self.client.get_world()
            
            # Enable synchronous mode if not already enabled
            settings = self.world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                self.world.apply_settings(settings)
                print("✓ Enabled synchronous mode (0.05s timestep)")
            
            cleanup_actors(self.world)
            print(f"✓ Connected to CARLA simulator ({host}:{port})")
            return True
            
        except Exception as e:
            print(f"✗ Failed to connect to CARLA: {e}")
            return False
    
    def spawn_vehicle(self, spawn_point: Optional[carla.Transform] = None) -> bool:
        """
        Spawn training vehicle.
        
        Args:
            spawn_point: Optional specific spawn location
            
        Returns:
            bool: True if spawn successful
        """
        try:
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
            
            if spawn_point is None:
                spawn_points = self.world.get_map().get_spawn_points()
                spawn_point = random.choice(spawn_points)
            
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            
            # Give CARLA time to initialize vehicle
            for _ in range(10):
                self.world.tick()
            time.sleep(0.5)
            
            print("✓ Training vehicle spawned")
            return True
            
        except Exception as e:
            print(f"✗ Failed to spawn vehicle: {e}")
            return False
    
    def spawn_npcs(self) -> None:
        """Spawn NPC vehicles and pedestrians."""
        try:
            self.npc_vehicles, self.npc_pedestrians = spawn_vehicles_and_pedestrians(
                self.world,
                num_vehicles=self.config.training.num_npc_vehicles,
                num_pedestrians=self.config.training.num_npc_pedestrians
            )
            if self.npc_vehicles or self.npc_pedestrians:
                print(f"✓ Spawned {len(self.npc_vehicles)} NPC vehicles, {len(self.npc_pedestrians)} pedestrians")
        except Exception as e:
            print(f"Warning: Failed to spawn NPCs: {e}")
    
    def initialize_model(self):
        """
        Initialize neural network model based on configuration.
        
        Returns:
            Model instance
        """
        if self.config.model.model_type == 'linear':
            model = CustomMLModel(param_size=self.config.model.get_param_size())
        else:  # temporal
            model = SimpleTemporalModel(
                input_size=self.config.model.input_size,
                memory_size=self.config.model.memory_size,
                decay_factor=self.config.model.decay_factor
            )
        
        print(f"✓ {model.__class__.__name__} initialized ({self.config.model.get_param_size()} parameters)")
        return model
    
    def initialize_agent(self, model):
        """
        Initialize agent with model and vehicle.
        
        Args:
            model: Neural network model
            
        Returns:
            Agent instance
        """
        if self.config.model.model_type == 'temporal':
            self.agent = TemporalMLAgent(self.vehicle, model)
        else:
            self.agent = CustomAdvancedMLAgent(self.vehicle, model)
        
        # Give sensors time to initialize
        for _ in range(10):
            self.world.tick()
        time.sleep(0.3)
        
        print(f"✓ {self.agent.__class__.__name__} initialized with sensors")
        return self.agent
    
    def initialize_nes(self) -> NES:
        """
        Initialize NES optimizer.
        
        Returns:
            NES instance
        """
        self.nes = NES(
            agent=self.agent,
            population_size=self.config.nes.population_size,
            sigma=self.config.nes.sigma,
            learning_rate=self.config.nes.learning_rate
        )
        
        print(f"✓ NES optimizer initialized")
        return self.nes
    
    def initialize_hud(self) -> Optional[object]:
        """
        Initialize HUD for visualization.
        
        Returns:
            HUD instance or None if failed
        """
        try:
            self.hud = create_hud(self.world)
            print("✓ Training HUD initialized")
            return self.hud
        except Exception as e:
            print(f"Warning: HUD initialization failed: {e}")
            return None
    
    def cleanup(self) -> None:
        """Perform deep cleanup of all CARLA resources."""
        from training.training_utils import deep_cleanup
        
        print("\n" + "=" * 60)
        print("Performing deep cleanup...")
        print("=" * 60)
        
        try:
            # Destroy HUD
            if self.hud:
                try:
                    self.hud.destroy()
                    print("✓ HUD destroyed")
                except Exception as e:
                    print(f"  Warning: HUD cleanup: {e}")
            
            # Destroy agent (includes sensors) - check if not already destroyed
            if self.agent:
                try:
                    # Only destroy if not already destroyed
                    if not (hasattr(self.agent, '_destroyed') and self.agent._destroyed):
                        self.agent.destroy()
                        print("✓ Agent destroyed")
                except Exception as e:
                    print(f"  Warning: Agent cleanup: {e}")
            
            # Destroy vehicle
            if self.vehicle:
                try:
                    if self.vehicle.is_alive:
                        self.vehicle.destroy()
                    print("✓ Vehicle destroyed")
                except Exception as e:
                    print(f"  Warning: Vehicle cleanup: {e}")
            
            # Deep cleanup: remove all actors and orphaned sensors
            if self.world:
                try:
                    print("\nPerforming deep cleanup of world actors...")
                    deep_cleanup(self.world, active_vehicle=None)
                    print("✓ Deep cleanup completed")
                except Exception as e:
                    print(f"  Warning: Deep cleanup: {e}")
            
            print("\n" + "=" * 60)
            print("Cleanup completed successfully!")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n✗ Cleanup error: {e}")
            import traceback
            traceback.print_exc()
