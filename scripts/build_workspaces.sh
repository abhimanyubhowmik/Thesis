#!/bin/bash

# build_workspaces.sh
# Script to build all workspaces for UAV Navigation System

set -e  # Exit on any error

echo "======================================="
echo "UAV Navigation System - Workspace Build"
echo "======================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Function to check if a workspace exists
check_workspace() {
    local ws_path=$1
    local ws_name=$2
    
    if [ ! -d "$ws_path" ]; then
        print_error "$ws_name workspace not found at $ws_path"
        print_error "Please run install_dependencies.sh first"
        exit 1
    fi
}

# Function to build a workspace
build_workspace() {
    local ws_path=$1
    local ws_name=$2
    
    print_step "Building $ws_name workspace..."
    
    cd "$ws_path"
    
    # Source ROS
    source /opt/ros/*/setup.bash
    
    # Check if there are packages to build
    if [ -z "$(find src -name 'package.xml' 2>/dev/null)" ]; then
        print_warning "No ROS packages found in $ws_name/src, skipping build"
        return
    fi
    
    # Clean previous build (optional)
    if [ "$CLEAN_BUILD" = "true" ]; then
        print_status "Cleaning previous build for $ws_name..."
        catkin clean -y || rm -rf build devel
    fi
    
    # Build with catkin
    print_status "Running catkin build for $ws_name..."
    if command -v catkin &> /dev/null; then
        catkin build -j$(nproc) --cmake-args -DCMAKE_BUILD_TYPE=Release
    else
        # Fallback to catkin_make if catkin build is not available
        catkin_make -j$(nproc) -DCMAKE_BUILD_TYPE=Release
    fi
    
    if [ $? -eq 0 ]; then
        print_status "$ws_name built successfully!"
    else
        print_error "Failed to build $ws_name"
        exit 1
    fi
}

# Parse command line arguments
CLEAN_BUILD=false
WORKSPACE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --clean           Clean build (remove previous build artifacts)"
            echo "  --workspace NAME  Build only specific workspace (voxblox|gbplanner|simulation|confidence)"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Define workspace paths
VOXBLOX_WS=~/voxblox_ws
GBPLANNER_WS=~/gbplanner2_ws
SIMULATION_WS=~/simulation_ws
CONFIDENCE_WS=~/confidence_ws

print_status "Starting workspace build process..."

if [ -n "$WORKSPACE" ]; then
    # Build specific workspace
    case $WORKSPACE in
        voxblox)
            check_workspace "$VOXBLOX_WS" "Voxblox"
            build_workspace "$VOXBLOX_WS" "Voxblox"
            ;;
        gbplanner)
            check_workspace "$GBPLANNER_WS" "GB Planner"
            build_workspace "$GBPLANNER_WS" "GB Planner"
            ;;
        simulation)
            check_workspace "$SIMULATION_WS" "Simulation"
            build_workspace "$SIMULATION_WS" "Simulation"
            ;;
        confidence)
            check_workspace "$CONFIDENCE_WS" "Confidence"
            build_workspace "$CONFIDENCE_WS" "Confidence"
            ;;
        *)
            print_error "Unknown workspace: $WORKSPACE"
            print_error "Available workspaces: voxblox, gbplanner, simulation, confidence"
            exit 1
            ;;
    esac
else
    # Build all workspaces in correct order
    print_status "Building all workspaces in dependency order..."
    
    # Step 1: Build Voxblox first
    check_workspace "$VOXBLOX_WS" "Voxblox"
    build_workspace "$VOXBLOX_WS" "Voxblox"
    
    # Step 2: Build GB Planner (depends on Voxblox)
    check_workspace "$GBPLANNER_WS" "GB Planner"
    
    # Remove any existing voxblox in gbplanner and link to our custom version
    print_step "Linking custom Voxblox to GB Planner workspace..."
    if [ -d "$GBPLANNER_WS/src/voxblox" ]; then
        rm -rf "$GBPLANNER_WS/src/voxblox"
    fi
    ln -sf "$VOXBLOX_WS/src/voxblox" "$GBPLANNER_WS/src/voxblox"
    
    # Source voxblox before building gbplanner
    cd "$GBPLANNER_WS"
    source /opt/ros/*/setup.bash
    if [ -f "$VOXBLOX_WS/devel/setup.bash" ]; then
        source "$VOXBLOX_WS/devel/setup.bash"
    fi
    
    build_workspace "$GBPLANNER_WS" "GB Planner"
    
    # Step 3: Build Simulation workspace
    check_workspace "$SIMULATION_WS" "Simulation"
    build_workspace "$SIMULATION_WS" "Simulation"
    
    # Step 4: Build Confidence workspace
    check_workspace "$CONFIDENCE_WS" "Confidence"
    build_workspace "$CONFIDENCE_WS" "Confidence"
fi

print_status "All builds completed successfully!"

# Update the source script
print_step "Updating workspace source script..."
cat > ~/source_all_workspaces.sh << 'EOF'
#!/bin/bash
# Source all workspaces in correct order
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

echo "All workspaces sourced successfully!"
EOF

chmod +x ~/source_all_workspaces.sh

print_status "Build process completed!"
print_status "Next steps:"
echo "  1. Run 'source ~/source_all_workspaces.sh' to source all workspaces"
echo "  2. Run './run_system.sh' to start the system"

# Verification
print_step "Verifying builds..."
for ws in voxblox_ws gbplanner2_ws simulation_ws confidence_ws; do
    if [ -f ~/$ws/devel/setup.bash ]; then
        print_status "✓ $ws built successfully"
    else
        print_warning "✗ $ws build may have issues"
    fi
done