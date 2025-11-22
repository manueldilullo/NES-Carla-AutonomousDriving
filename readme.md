# NES-CARLA: Natural Evolution Strategies for Autonomous Driving

Evolutionary optimization of neural networks for autonomous vehicle control in the CARLA simulator using Natural Evolution Strategies (NES).
Related paper [here](./NeuroEvolution%20of%20a%20Temporal%20Policy%20for%20Autonomous%20Driving%20in%20CARLA.pdf)
---

## 🚀 Quick Start

```bash
# 1. Start CARLA simulator
./CarlaUE4.sh

# 2. Run training (in new terminal)
python train.py --type temporal --preset standard

# Other options:
python train.py --type batch --batch-size 4     # Parallel training (4x faster)
python train.py --type nes --preset linear      # Simple linear model
python train.py --type temporal --preset quick_test  # Fast testing
```

---

## Table of Contents

1. [Overview](#overview)
2. [Technology Stack](#technology-stack)
3. [Theoretical Foundation](#theoretical-foundation)
4. [Architecture](#architecture)
5. [Reward Function](#reward-function)
6. [Setup Instructions](#setup-instructions)
7. [Training Guide](#training-guide)
8. [Model Variants](#model-variants)
9. [Performance Analysis](#performance-analysis)

---

## Overview

This project implements autonomous driving agents using **gradient-free evolutionary optimization**. Instead of backpropagation, the system evolves neural network parameters through Natural Evolution Strategies (NES), evaluating each candidate by having it drive in the CARLA simulator.

### Key Features

- **Gradient-Free Learning**: No backpropagation required - evolution through fitness evaluation
- **Multi-Modal Sensors**: RGB camera, LiDAR, collision detection
- **Temporal Models**: Simple recurrent architecture for sequential decision-making  
- **Batch Training**: Parallel evaluation of multiple vehicles for 4x speedup
- **Lightweight Models**: Linear and simple temporal networks optimized for NES
- **Comprehensive Logging**: Automatic tracking of training progress and model checkpoints

### Design Philosophy

Following **KISS** (Keep It Simple, Stupid) and **YAGNI** (You Aren't Gonna Need It):
- Simple linear/temporal models instead of deep networks
- Population-based optimization instead of gradient descent
- Minimal dependencies and modular architecture
- Focus on core autonomous driving without over-engineering

---

## Technology Stack

### Core Dependencies

- **CARLA 0.9.15**: Open-source autonomous driving simulator
- **Python 3.8+**: Main programming language
- **NumPy**: Numerical computations and array operations
- **NetworkX**: Graph algorithms for route planning (A*)

### Optional Dependencies

- **Matplotlib/Seaborn**: Visualization and training plots
- **Pandas**: Training log analysis
- **PyTorch**: Only for advanced models (not required for basic NES training)

### Hardware Requirements

- **CPU**: Multi-core recommended (4+ cores)
- **GPU**: NVIDIA GPU with 6GB+ VRAM (for batch training)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 20GB for CARLA + models

---

## Theoretical Foundation

### Natural Evolution Strategies (NES)

NES is a gradient-free optimization algorithm that estimates parameter gradients through fitness-based selection:

```
θ_{t+1} = θ_t + α · ∇_θ E[F(θ + σε)]

where:
  θ: neural network parameters
  F: fitness function (driving performance)
  σ: exploration noise (standard deviation)
  α: learning rate
  ε ~ N(0, I): Gaussian noise
```

#### Algorithm Steps

1. **Population Generation**: Create N parameter variations by adding Gaussian noise to base parameters
   ```python
   θ_i = θ_base + σ · ε_i,  where ε_i ~ N(0, I)
   ```

2. **Fitness Evaluation**: Test each individual by driving in simulation
   ```python
   F_i = evaluate_driving(θ_i)  # Higher = better performance
   ```

3. **Gradient Estimation**: Compute search gradient from fitness-weighted parameter differences
   ```python
   ∇F ≈ (1/(N·σ)) · Σ F_i · (θ_i - θ_base)
   ```

4. **Parameter Update**: Move base parameters toward better-performing regions
   ```python
   θ_base = θ_base + α · ∇F
   ```

5. **Iterate**: Generate new population around updated parameters

#### Why NES for Autonomous Driving?

**Advantages:**
- ✅ **No Gradient Computation**: Works with non-differentiable simulators and reward functions
- ✅ **Robust to Sparse Rewards**: Handles discrete events (collisions, waypoint reaching)
- ✅ **Natural Exploration**: Gaussian noise provides structured exploration
- ✅ **Parallel Evaluation**: Each individual can be evaluated independently
- ✅ **Simple Implementation**: No backpropagation or computational graph needed

**Trade-offs:**
- ⚠️ **Sample Inefficiency**: Requires more evaluations than gradient-based methods
- ⚠️ **Scalability**: Best for low-to-medium parameter counts (< 1000)
- ⚠️ **Convergence**: Can be slower than gradient descent for smooth objectives

### Neural Network Models

#### 1. Linear Model (CustomMLModel)

Simplest model mapping vehicle state directly to actions:

```
State → [W_steer, W_throttle, W_brake] → Actions

Input:  [speed, waypoint_dist, waypoint_angle, obstacle_dist, collision]
Output: [steering ∈ [-1,1], throttle ∈ [0,1], brake ∈ [0,1]]
```

- **Parameters**: 15 (5 inputs × 3 actions)
- **Activation**: Tanh for steering, clipped linear for throttle/brake
- **Use Case**: Fast inference, interpretable weights

#### 2. Temporal Model (SimpleTemporalModel)

Adds memory for sequential decision-making without full LSTM complexity:

```
Extended State = Concat(Current State, Memory)
Memory = [steer_{t-1}, throttle_{t-1}, brake_{t-1}, ..., steer_{t-N}]

Extended State → [W_steer, W_accel] → Actions
Memory Update: memory_new = (1-decay) · current + decay · memory_old
```

- **Parameters**: 40-80 depending on memory size (10 inputs + 3N memory elements) × 2 outputs
- **Temporal Awareness**: Exponential moving average of past actions
- **Recurrence**: Simple decay-based memory (not LSTM gates)
- **Mutual Exclusion**: Single acceleration output splits into throttle/brake

**Key Innovation**: Unified acceleration control prevents throttle+brake conflicts:
```python
accel = tanh(W_accel @ extended_state)
if accel >= 0:
    throttle, brake = accel, 0
else:
    throttle, brake = 0, -accel
```

### Navigation and Control

#### A* Global Planning

Uses road network topology for route planning:
```python
route = nx.astar_path(
    graph, 
    start_node, 
    end_node,
    heuristic=euclidean_distance,
    weight='length'
)
```

#### PID Local Control

Low-level vehicle control with PID controllers:

**Lateral (Steering):**
```
error = angle_to_waypoint
steering = K_P · error + K_D · d(error)/dt + K_I · ∫error dt
```

**Longitudinal (Speed):**
```
speed_error = target_speed - current_speed
acceleration = K_P · speed_error + K_D · d(speed_error)/dt
```

---

## Architecture

```
NES_Carla/
├── train.py                    # 🆕 UNIFIED TRAINING ENTRY POINT
├── config.py                   # 🆕 CENTRALIZED CONFIGURATION
│
├── agents/                     # ML agents and NES optimizer
│   ├── nes_agent.py           # Natural Evolution Strategies
│   ├── custom_ml_agent.py     # Advanced agent with sensors
│   └── __init__.py
│
├── models/                     # Neural network models
│   ├── custom_ml_model.py     # Linear model (15 params)
│   ├── simple_temporal_model.py # Temporal model (36-78 params)
│   └── __init__.py
│
├── navigation/                 # Path planning and control
│   ├── global_route_planner.py # A* route planning
│   ├── local_planner.py       # Waypoint following with PID
│   ├── controller.py          # PID controllers
│   └── __init__.py
│
├── training_logs/             # Auto-generated training outputs
│   └── [experiment_folders]/  # Timestamped training sessions
│       ├── config.json            # Saved configuration
│       ├── training_log.csv       # Generation statistics
│       ├── best_models/           # Best performing models
│       │   ├── best_model.npy
│       │   └── best_model_metadata.json
│       └── checkpoints/           # Training checkpoints
│           └── checkpoint_gen_*.pkl
│
├── utils/                      # Utility functions
│   ├── reward.py              # Reward function
│   ├── actors_handler.py      # Vehicle/pedestrian management
│   ├── camera.py              # Camera utilities
│   ├── logger.py              # Logging system
│   └── hud.py                 # HUD visualization
│
├── train_nes.py               # ⚠️ DEPRECATED: Use `train.py --type nes`
├── train_temporal.py          # ⚠️ DEPRECATED: Use `train.py --type temporal`
├── train_batch.py             # ⚠️ DEPRECATED: Use `train.py --type batch`
│
├── analyze_training.py        # Training analysis and visualization
├── requirements.txt           # Python dependencies
└── readme.md                  # This file
```

### Migration from Old Scripts

**Old way:**
```bash
python train_nes.py
python train_temporal.py
python train_batch.py --batch-size 4
```

**New way (recommended):**
```bash
python train.py --type nes
python train.py --type temporal
python train.py --type batch --batch-size 4
```

**Benefits:**
- ✅ Single entry point for all training modes
- ✅ Consistent configuration across modes
- ✅ Easy to tune hyperparameters via config presets
- ✅ Better code organization (DRY principle)
- ✅ Reproducible experiments with saved configs

---

## Reward Function

The reward function shapes learning through multi-objective optimization:

### Components

```python
reward = waypoint_progress + movement_reward - penalties

where:
  waypoint_progress: ONE-TIME reward per waypoint reached
  movement_reward: Small bonus for maintaining speed
  penalties: collision + off_road + stuck
```

### 1. Waypoint Progress (PRIMARY SIGNAL)

**Purpose**: Guide agent toward destination  
**Implementation**: One-time reward per new waypoint reached

```python
WAYPOINT_PROGRESS_SCALE = 100.0  # Very large reward dominates learning

if reached_new_waypoint:
    reward += num_new_waypoints * WAYPOINT_PROGRESS_SCALE
```

**Why One-Time?**
- Prevents reward "farming" by staying near waypoint
- Encourages continuous forward progress
- Clear signal: reach waypoint → get reward → move to next

### 2. Safety Penalties

**Collision**: Moderate penalty to discourage collisions while allowing learning
```python
COLLISION_PENALTY = 50.0
if collision_detected:
    reward -= COLLISION_PENALTY
```

**Off-Road**: Penalty for leaving drivable surface
```python
OFF_ROAD_PENALTY = 5.0
if distance_from_lane_center > 3.5:  # meters
    reward -= OFF_ROAD_PENALTY
```

### 3. Stuck Detection (Distance-Based)

**Purpose**: Prevent agent from stopping indefinitely  
**Implementation**: Check progress every N steps

```python
STUCK_CHECK_INTERVAL = 50        # steps
MIN_PROGRESS_DISTANCE = 3.0      # meters
STUCK_PENALTY = 20.0

if steps % STUCK_CHECK_INTERVAL == 0:
    distance_moved = current_pos.distance(start_pos)
    if distance_moved < MIN_PROGRESS_DISTANCE:
        reward -= STUCK_PENALTY
```

### 4. Movement Reward

**Purpose**: Encourage agent to keep moving  
**Implementation**: Small constant reward for any movement

```python
MOVEMENT_THRESHOLD = 0.5   # km/h
MOVEMENT_REWARD = 0.5
STANDING_STILL_PENALTY = 1.0

if speed > MOVEMENT_THRESHOLD:
    reward += MOVEMENT_REWARD
else:
    reward -= STANDING_STILL_PENALTY
```

### Reward Tuning Philosophy

**Design Principles:**
- **Dominant Signal**: Waypoint progress (50.0) >> penalties (5-20)
- **Allow Mistakes**: Small penalties enable exploratory learning
- **Simple Components**: Few, clear objectives avoid contradictory signals
- **One-Time Rewards**: Prevent exploitation and reward farming

**Tuning Guidelines:**
```python
# If agent doesn't move:
WAYPOINT_PROGRESS_SCALE += 20  # Increase primary signal

# If agent crashes too much:
COLLISION_PENALTY *= 2  # Increase safety importance

# If agent gets stuck:
STUCK_CHECK_INTERVAL -= 10  # Check more frequently
MIN_PROGRESS_DISTANCE += 1  # Require more progress
```

### Expected Reward Ranges

- **Early Training (gen 1-50)**: -100 to 200 (many collisions)
- **Mid Training (gen 50-150)**: 200 to 800 (completing some waypoints)
- **Late Training (gen 150-300)**: 800 to 2000+ (consistent completion)

---

## Setup Instructions

### 1. Install CARLA Simulator

Download CARLA 0.9.15 from the official releases:

```bash
# Linux
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xzf CARLA_0.9.15.tar.gz
cd CARLA_0.9.15

# Windows
# Download CARLA_0.9.15.zip from GitHub releases
# Extract to desired location
```

**Note**: CARLA requires ~20GB disk space and runs best on systems with dedicated GPU.

### 2. Clone Repository

```bash
git clone https://github.com/yourusername/NES_Carla.git
cd NES_Carla
```

### 3. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies:**
- `carla==0.9.15`: CARLA Python API
- `numpy>=1.21.0`: Numerical operations
- `matplotlib>=3.5.0`: Visualization
- `pandas>=1.4.0`: Data analysis

### 4. Configure CARLA Python API

The CARLA Python API should be installed automatically via pip. If you encounter issues:

```bash
# Manual installation
cd /path/to/CARLA_0.9.15/PythonAPI/carla/dist
pip install carla-0.9.15-py3-none-any.whl
```

### 5. Test CARLA Connection

```bash
# Terminal 1: Start CARLA server
cd /path/to/CARLA_0.9.15
./CarlaUE4.sh  # Linux
# or
CarlaUE4.exe  # Windows

# Terminal 2: Test Python connection
python -c "import carla; client = carla.Client('localhost', 2000); print('Connected!')"
```

### 6. Verify Setup

```bash
# Quick test: spawn a vehicle
python -c "
import carla
client = carla.Client('localhost', 2000)
world = client.get_world()
bp = world.get_blueprint_library().filter('vehicle.*')[0]
spawn = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(bp, spawn)
print(f'Spawned vehicle: {vehicle.type_id}')
vehicle.destroy()
"
```

---

## Training Guide

### Unified Training Interface

All training modes are now accessible through a single `train.py` script with configuration presets:

```bash
# Terminal 1: Start CARLA
./CarlaUE4.sh

# Terminal 2: Choose your training mode
```

### Quick Start Examples

**1. Standard Temporal Training (Recommended)**
```bash
python train.py --type temporal --preset standard
```

**2. Fast Batch Training (4x speedup)**
```bash
python train.py --type batch --batch-size 4
```

**3. Simple Linear Model**
```bash
python train.py --type nes --preset linear
```

**4. Quick Testing (50 generations)**
```bash
python train.py --type temporal --preset quick_test
```

### Configuration Presets

The system includes pre-tuned configurations:

- **`standard`**: Production-quality temporal training (300 generations)
- **`quick_test`**: Fast testing configuration (50 generations, small population)
- **`batch`**: Optimized for parallel batch training
- **`linear`**: Simple linear model baseline

### Advanced Usage

**Custom Hyperparameters:**
```bash
# Override specific parameters
python train.py --type temporal --generations 500 --population 20

# Adjust batch size
python train.py --type batch --batch-size 6
```

**Resume Training:**
```bash
python train.py --type temporal --resume training_logs/temporal_20251121_143022/checkpoints/checkpoint_gen_100.pkl
```

**Configuration Files:**
```bash
# Configurations are automatically saved to training_logs/*/config.json
# Modify and reuse saved configurations for reproducibility
```

### Performance Comparison

| Mode | Speed | GPU | Use Case |
|------|-------|-----|----------|
| **Sequential** | Baseline | CPU OK | Debugging, small experiments |
| **Batch (4x)** | 4x faster | 6GB+ VRAM | Production training |
| **Batch (6x)** | 6x faster | 8GB+ VRAM | Large-scale experiments |

### Configuration System

All hyperparameters are centralized in `config.py`. You can:

1. **Use presets** (recommended for most users)
2. **Modify existing presets** in `config.py`
3. **Create custom configurations** programmatically

**Key Parameters:**

```python
# Model Configuration
model_config = {
    'model_type': 'temporal',  # 'linear' or 'temporal'
    'input_size': 10,          # Number of input features (comprehensive state)
    'memory_size': 5,          # Timesteps to remember (temporal only)
    'decay_factor': 0.85       # Memory decay rate [0,1]
}

# NES Configuration
nes_config = {
    'population_size': 50,     # Individuals per generation
    'sigma': 0.1,              # Exploration noise
    'learning_rate': 0.01,     # Parameter update step size
    'min_sigma': 0.05,         # Minimum exploration
    'sigma_decay': 0.995       # Decay rate per generation
}

# Training Configuration
training_config = {
    'num_generations': 300,    # Total generations
    'max_steps': 600,          # Steps per episode
    'min_route_distance': 50,  # Minimum route length (meters)
    'max_route_distance': 150, # Maximum route length (meters)
    'completion_bonus': 500,   # Reward for completing route
    'save_interval': 10        # Checkpoint every N generations
}
```

**Tuning Guidelines:**

| Parameter | Small | Default | Large | Effect |
|-----------|-------|---------|-------|--------|
| **population_size** | 6-8 | 50 | 100+ | Gradient quality vs speed |
| **sigma** | 0.05 | 0.1 | 0.3-0.5 | Exploration vs exploitation |
| **learning_rate** | 0.005 | 0.01 | 0.05 | Convergence speed vs stability |
| **memory_size** | 2-3 | 5 | 8-10 | Temporal awareness (parameters) |
| **batch_size** | 2 | 4 | 6-8 | Training speed (needs more VRAM) |

### Monitoring Training

#### Real-Time Logs

Watch console output for generation statistics:
```
=== Generation 64/300 ===
Mean fitness: 456.78
Max fitness: 1243.91 (best overall)
Completion rate: 25.0%
Generation time: 68.3s
```

#### Training Log CSV

Automatically saved to `training/temporal_model_YYYYMMDD_HHMMSS/training_log.csv`:

| Column | Description |
|--------|-------------|
| generation | Generation number |
| mean_fitness | Average population fitness |
| max_fitness | Best individual this generation |
| best_fitness_overall | Best across all generations |
| completion_rate | % of individuals completing route |
| target_min_distance | Progressive difficulty (min) |
| target_max_distance | Progressive difficulty (max) |
| generation_time_s | Time taken (seconds) |

#### Analysis Script

Generate plots and statistics:

```bash
python analyze_training.py training/temporal_model_20251120_173541/training_log.csv
```

Outputs:
- `training_progress.png`: Fitness curves and completion rates
- `training_summary.json`: Statistical summary
- Console: Milestones and insights

### Expected Training Behavior

#### Early Generations (1-50)
- **Fitness**: Negative to low positive (-500 to 200)
- **Behavior**: Random steering, frequent collisions
- **Completion**: 0-5%
- **What to Look For**: Gradual fitness increase

#### Mid Training (50-150)
- **Fitness**: Steady improvement (200-800)
- **Behavior**: Following waypoints, fewer crashes
- **Completion**: 10-30%
- **What to Look For**: Completion rate rising

#### Late Training (150-300)
- **Fitness**: High plateau (800-2000+)
- **Behavior**: Smooth driving, consistent navigation
- **Completion**: 30-60%
- **What to Look For**: Stable high performance

### Checkpoints and Saving

Automatic saves during training:

```
training/temporal_model_YYYYMMDD_HHMMSS/
├── training_log.csv              # Updated every generation
├── training_config.json          # Saved at start
├── best_models/
│   ├── best_model.npy           # Updated when fitness improves
│   └── best_model_metadata.json # Model info and generation
└── checkpoints/
    ├── checkpoint_gen_25.pkl    # Full state every 25 generations
    ├── checkpoint_gen_50.pkl
    └── ...
```

### Loading Trained Models

```python
from config import Config
from models.simple_temporal_model import SimpleTemporalModel

# Load config from training directory
config = Config.from_json('training_logs/temporal_20251121_143022/config.json')

# Initialize model with saved config
model = SimpleTemporalModel(
    input_size=config.model.input_size,
    memory_size=config.model.memory_size,
    decay_factor=config.model.decay_factor
)

# Load best parameters
model.load('training_logs/temporal_20251121_143022/best_models/best_model.npy')

# Use for prediction
steering, throttle, brake = model.predict(state)
```

---

## Model Variants

All variants are accessible through the unified `train.py` interface:

### 1. Linear Model (`--type nes`)

**Best For**: Quick experiments, interpretable weights, CPU training

```bash
python train.py --type nes --preset linear
```

**Characteristics:**
- 15 parameters (5 inputs × 3 actions)
- No temporal awareness
- Fast training, easy to debug
- Good baseline for comparison

### 2. Temporal Model (`--type temporal`)

**Best For**: Production use, smooth driving, sequential tasks

```bash
python train.py --type temporal --preset standard
```

**Characteristics:**
- 40-80 parameters (depending on memory size)
- Temporal memory for sequential decisions
- Smoother control than linear
- Better at complex maneuvers
- Uses 10-feature comprehensive state vector

### 3. Batch Training (`--type batch`)

**Best For**: Fast iteration, GPU utilization, large-scale experiments

```bash
python train.py --type batch --batch-size 4
```

**Characteristics:**
- 4x faster wall-clock time (with batch_size=4)
- Parallel evaluation of multiple vehicles
- Requires 6GB+ VRAM
- Same accuracy as sequential training

---

## Performance Analysis

### Metrics

**Fitness Evolution:**
- Track mean, max, and standard deviation over generations
- Best fitness should reach 1000+ for competent driving

**Completion Rate:**
- Percentage of individuals completing route without collision
- Target: 30-50% by generation 300

**Smoothness:**
- Standard deviation of steering commands
- Lower = smoother driving

### Troubleshooting

**Issue**: Fitness stuck near 0
- **Cause**: Insufficient waypoint progress signal
- **Fix**: Increase `WAYPOINT_PROGRESS_SCALE` in reward.py

**Issue**: High variance in fitness
- **Cause**: High exploration (sigma) or inconsistent evaluations
- **Fix**: Reduce sigma or increase episode length

**Issue**: No completions by gen 100
- **Cause**: Routes too difficult or penalties too harsh
- **Fix**: Reduce initial route distance or decrease penalties

**Issue**: Training very slow
- **Cause**: Too many vehicles/pedestrians or high rendering resolution
- **Fix**: Reduce NPC count or use no-rendering mode

---

## Advanced Topics

### Custom Reward Functions

Modify `utils/reward.py` to add new objectives:

```python
# Add speed optimization
target_speed = 30  # km/h
speed_error = abs(speed - target_speed)
reward -= speed_error * 0.1

# Add lane centering
lateral_deviation = distance_from_lane_center
reward -= abs(lateral_deviation) * 2.0
```

### Feature Engineering

Extend state representation in `agents/custom_ml_agent.py`:

```python
def get_comprehensive_state(self):
    """
    Get comprehensive state vector for temporal model.
    
    Returns 10-feature vector:
    - speed (normalized)
    - distance to destination
    - angle to destination
    - distance to next waypoint
    - angle to next waypoint
    - collision flag
    - lane invasion flag
    - previous steering
    - previous throttle
    - obstacle distance
    """
    return [
        speed,
        distance_to_destination,     # Long-term goal
        angle_to_destination,         # Long-term direction
        waypoint_dist,                # Short-term goal
        waypoint_angle,               # Short-term direction
        collision_flag,               # Safety indicator
        lane_invasion_flag,           # Lane violation
        last_steering,                # Previous control
        last_throttle,                # Previous control
        obstacle_dist                 # Obstacle detection
    ]
```

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

- **CARLA Team**: Open-source autonomous driving simulator
- **OpenAI**: Natural Evolution Strategies research
- **Computer Vision Center (CVC)**: CARLA navigation components

---

## Contact

**Author**: Manuel Di Lullo  
**Institution**: Sapienza University of Rome  
**Course**: Automatic Verification of Intelligent Systems  
**Year**: 2025
