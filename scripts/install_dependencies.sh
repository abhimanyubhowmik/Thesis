#!/bin/bash

# install_dependencies.sh
# Script to install all dependencies for UAV Navigation System

set -e  # Exit on any error

echo "==========================================="
echo "UAV Navigation System - Dependency Install"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Ubuntu
if ! command -v apt &> /dev/null; then
    print_error "This script is designed for Ubuntu systems with apt package manager"
    exit 1
fi

# Update system packages
print_status "Updating system packages..."
sudo apt update

# Install basic dependencies
print_status "Installing basic dependencies..."
sudo apt install -y \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    pkg-config

# Install ROS dependencies (adjust for your ROS version)
print_status "Installing ROS dependencies..."
sudo apt install -y \
    ros-*-desktop-full \
    ros-*-catkin \
    python3-catkin-tools \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool

# Initialize rosdep if not already done
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    print_status "Initializing rosdep..."
    sudo rosdep init
fi
rosdep update

# Install Python packages for HoloOcean
print_status "Installing Python packages..."
pip3 install --user \
    numpy \
    opencv-python \
    matplotlib \
    holoocean

# Install additional dependencies for GB Planner
print_status "Installing GB Planner dependencies..."
sudo apt install -y \
    libeigen3-dev \
    libpcl-dev \
    libompl-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libgtest-dev

# Install Voxblox dependencies
print_status "Installing Voxblox dependencies..."
sudo apt install -y \
    protobuf-compiler \
    libprotobuf-dev \
    libprotoc-dev

# Create workspace directories
print_status "Creating workspace directories..."
mkdir -p ~/gbplanner2_ws/src
mkdir -p ~/voxblox_ws/src
mkdir -p ~/simulation_ws/src
mkdir -p ~/confidence_ws/src

# Create HoloOcean config directory
print_status "Creating HoloOcean config directory..."
mkdir -p ~/.local/share/holoocean/0.5.0/worlds/OceanSimple

# Get the script directory to find config files
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Copy HoloOcean configuration
if [ -f "$REPO_ROOT/config/OceanSimple-Hovering2CameraOnly-VisibilityEstimation.json" ]; then
    print_status "Copying HoloOcean configuration..."
    cp "$REPO_ROOT/config/OceanSimple-Hovering2CameraOnly-VisibilityEstimation.json" \
       ~/.local/share/holoocean/0.5.0/worlds/OceanSimple/
else
    print_warning "HoloOcean config file not found. Please copy it manually later."
fi

# Clone repositories
print_status "Cloning GB Planner repository..."
cd ~/gbplanner2_ws/src
if [ ! -d "gbplanner_ros" ]; then
    git clone -b uam https://github.com/abhimanyubhowmik/gbplanner_ros.git
else
    print_warning "GB Planner already exists, skipping clone"
fi

print_status "Cloning custom Voxblox repository..."
cd ~/voxblox_ws/src
if [ ! -d "voxblox" ]; then
    git clone https://github.com/abhimanyubhowmik/voxblox.git
else
    print_warning "Voxblox already exists, skipping clone"
fi

# Copy simulation and voxel_viz folders
print_status "Copying simulation components..."
if [ -d "$REPO_ROOT/dependencies/simulation" ]; then
    cp -r "$REPO_ROOT/dependencies/simulation" ~/simulation_ws/src/
else
    print_warning "Simulation folder not found in dependencies"
fi

print_status "Copying confidence visualization components..."
if [ -d "$REPO_ROOT/dependencies/voxel_viz" ]; then
    cp -r "$REPO_ROOT/dependencies/voxel_viz" ~/confidence_ws/src/
else
    print_warning "voxel_viz folder not found in dependencies"
fi

# Install rosdep dependencies for each workspace
print_status "Installing rosdep dependencies..."

for ws in gbplanner2_ws voxblox_ws simulation_ws confidence_ws; do
    if [ -d ~/$ws/src ]; then
        print_status "Installing dependencies for $ws..."
        cd ~/$ws
        rosdep install --from-paths src --ignore-src -r -y || print_warning "Some dependencies for $ws might not be available"
    fi
done

# Set up environment
print_status "Setting up environment..."
if ! grep -q "source /opt/ros" ~/.bashrc; then
    echo "source /opt/ros/*/setup.bash" >> ~/.bashrc
fi

# Create a source script for convenience
cat > ~/source_all_workspaces.sh << 'EOF'
#!/bin/bash
# Source all workspaces
source /opt/ros/*/setup.bash
if [ -f ~/voxblox_ws/devel/setup.bash ]; then
    source ~/voxblox_ws/devel/setup.bash
fi
if [ -f ~/gbplanner2_ws/devel/setup.bash ]; then
    source ~/gbplanner2_ws/devel/setup.bash
fi
if [ -f ~/simulation_ws/devel/setup.bash ]; then
    source ~/simulation_ws/devel/setup.bash
fi
if [ -f ~/confidence_ws/devel/setup.bash ]; then
    source ~/confidence_ws/devel/setup.bash
fi
EOF

chmod +x ~/source_all_workspaces.sh

print_status "Installation completed successfully!"
print_status "Next steps:"
echo "  1. Run './build_workspaces.sh' to build all workspaces"
echo "  2. Run 'source ~/source_all_workspaces.sh' to source all workspaces"
echo "  3. Run './run_system.sh' to start the system"

print_warning "Note: You may need to restart your terminal or run 'source ~/.bashrc' for ROS to be available."