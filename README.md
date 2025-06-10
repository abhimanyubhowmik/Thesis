# Underwater Autonomous Vehicle Navigation System

## Overview
This repository contains a complete system for underwater autonomous vehicle (UAV) navigation using GB Planner with custom Voxblox integration and HoloOcean simulation. The system provides uncertainity aware mapping, path planning capabilities with visibility estimation and fog density analysis for underwater environments.

## Repository Structure
```
thesis/
├── README.md
├── dependencies/
│   ├── gbplanner_ros/          # Forked GB Planner (uam branch)
│   └── voxblox/                # Custom Voxblox implementation
├── simulation/             # Simulation components
├── voxel_viz/              # Confidence visualization tools
├── config/
│   └── OceanSimple-Hovering2CameraOnly-VisibilityEstimation.json
└── scripts/
    ├── install_dependencies.sh
    ├── build_workspaces.sh
    └── run_system.sh

```

## Prerequisites
- Ubuntu 18.04/20.04
- ROS Melodic/Noetic
- Python 3.x
- Git

## Installation

### 1. Clone the Repository
```bash
git clone <your-thesis-repo-url>
cd thesis_uav_navigation
```

### 2. Set Up Workspaces
Create the required workspace structure:
```bash
mkdir -p ~/gbplanner2_ws/src
mkdir -p ~/voxblox_ws/src
mkdir -p ~/simulation_ws/src
mkdir -p ~/confidence_ws/src
```

### 3. Install GB Planner
```bash
cd ~/gbplanner2_ws/src
git clone -b uam https://github.com/abhimanyubhowmik/gbplanner_ros.git
cd ~/gbplanner2_ws
# Follow installation instructions from the GB Planner repository
catkin build
```

### 4. Install Custom Voxblox
```bash
cd ~/voxblox_ws/src
git clone https://github.com/abhimanyubhowmik/voxblox.git
cd ~/voxblox_ws
catkin build
```

### 5. Replace Voxblox in GB Planner
```bash
# Remove automatically installed voxblox from gbplanner
rm -rf ~/gbplanner2_ws/src/voxblox

# Create symbolic link to custom voxblox
ln -s ~/voxblox_ws/src/voxblox ~/gbplanner2_ws/src/voxblox

# Rebuild GB Planner with custom voxblox
cd ~/gbplanner2_ws
catkin build
```

### 6. Install Simulation Components
```bash
cd ~/simulation_ws/src
# Copy simulation folder from this repository
cp -r /path/to/thesis_uav_navigation/dependencies/simulation .
cd ~/simulation_ws
catkin build
```

### 7. Install Confidence Visualization
```bash
cd ~/confidence_ws/src
# Copy voxel_viz folder from this repository
cp -r /path/to/thesis_uav_navigation/dependencies/voxel_viz .
cd ~/confidence_ws
catkin build
```

### 8. Install HoloOcean Simulator
```bash
pip3 install holoocean

# Install in development mode (if needed)
git clone https://github.com/BYU-PCCL/holodeck-engine.git
cd holodeck-engine
pip3 install -e .
```

### 9. Configure HoloOcean
```bash
# Create HoloOcean config directory
mkdir -p ~/.local/share/holoocean/0.5.0/worlds/OceanSimple

# Copy configuration file
cp config/OceanSimple-Hovering2CameraOnly-VisibilityEstimation.json \
   ~/.local/share/holoocean/0.5.0/worlds/OceanSimple/
```

## Usage

### Running the Complete System

#### Terminal 1: Start ROS Core
```bash
roscore
```

#### Terminal 2: Launch Simulation
```bash
cd ~/simulation_ws
source devel/setup.bash

# For basic simulation
roslaunch holoocean_ros holoocean_sim.launch

# For simulation with fog
roslaunch holoocean_ros fog_holoocean.launch
```

#### Terminal 3: Launch GB Planner
```bash
cd ~/gbplanner2_ws
source devel/setup.bash
roslaunch gbplanner auv_sim.launch
```

#### Terminal 4: Launch Confidence Visualization
```bash
cd ~/confidence_ws
source devel/setup.bash
roslaunch voxel_viz confidence_visualization.launch
```

### Monitoring System Status

#### Check Fog Density
```bash
rostopic echo /current_fog_density
```

#### Check Robot Position
```bash
rostopic echo /robot_position
```

#### View Available Topics
```bash
rostopic list
```

## System Architecture

The system consists of four main components:

1. **GB Planner (gbplanner2_ws)**: Core path planning algorithm with UAV-specific modifications
2. **Custom Voxblox (voxblox_ws)**: Enhanced voxel-based mapping with custom features
3. **HoloOcean Simulation (simulation_ws)**: Underwater environment simulation with fog effects
4. **Confidence Visualization (confidence_ws)**: Real-time visualization of navigation confidence

## Key Features

- **Underwater Navigation**: Specialized for UAV operations in marine environments
- **Fog Simulation**: Realistic underwater visibility conditions
- **Real-time Visualization**: Live confidence and density mapping
- **Modular Design**: Separate workspaces for easy maintenance and development

## Configuration

### Simulation Parameters
Modify `OceanSimple-Hovering2CameraOnly-VisibilityEstimation.json` to adjust:
- Camera settings
- Environmental conditions
- Vehicle dynamics
- Sensor configurations

### Planning Parameters
GB Planner parameters can be adjusted in the respective launch files within the gbplanner2_ws.

## Troubleshooting

### Common Issues

1. **Build Failures**: Ensure all dependencies are installed and workspaces are properly sourced
2. **HoloOcean Config**: Verify the config file is in the correct location
3. **Voxblox Integration**: Ensure the symbolic link is created correctly
4. **ROS Communication**: Check that all terminals are sourced with the correct workspace

### Logs and Debugging
```bash
# Check ROS logs
roscd log

# Monitor system performance
rostopic hz /topic_name
```

## Citation
If you use this work in your research, please cite:
```
[Your thesis citation information]
```

## Dependencies
- [GB Planner ROS](https://github.com/abhimanyubhowmik/gbplanner_ros/tree/uam)
- [Custom Voxblox](https://github.com/abhimanyubhowmik/voxblox)
- [HoloOcean](https://github.com/BYU-PCCL/holodeck-engine)

## License
[Specify your license here]

## Contact
[bhowmikabhimanyu@gmail.com](mailto:bhowmikabhimanyu@gmail.com)
