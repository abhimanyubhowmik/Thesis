#!/bin/bash

# run_system.sh
# Script to run the complete UAV Navigation System

echo "====================================="
echo "UAV Navigation System - System Launch"
echo "====================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

print_instruction() {
    echo -e "${CYAN}[INSTRUCTION]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Function to check if ROS is running
check_ros_core() {
    if ! pgrep -f "roscore\|rosmaster" > /dev/null; then
        return 1
    fi
    return 0
}

# Function to check if workspace is built
check_workspace_built() {
    local ws_path=$1
    local ws_name=$2
    
    if [ ! -f "$ws_path/devel/setup.bash" ]; then
        print_error "$ws_name workspace not built. Please run build_workspaces.sh first"
        return 1
    fi
    return 0
}

# Function to wait for user input
wait_for_user() {
    local message=$1
    print_instruction "$message"
    read -p "Press Enter to continue..."
}

# Function to open new terminal with command
open_terminal() {
    local title=$1
    local command=$2
    
    # Try different terminal emulators
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal --title="$title" -- bash -c "$command; exec bash"
    elif command -v xterm &> /dev/null; then
        xterm -title "$title" -e "$command; bash" &
    elif command -v konsole &> /dev/null; then
        konsole --title "$title" -e bash -c "$command; exec bash" &
    else
        print_error "No supported terminal emulator found"
        print_error "Please run the following command manually in a new terminal:"
        echo "$command"
        wait_for_user "After running the command, press Enter to continue"
    fi
}

# Parse command line arguments
SIMULATION_TYPE="basic"
AUTO_MODE=false
MANUAL_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --fog)
            SIMULATION_TYPE="fog"
            shift
            ;;
        --auto)
            AUTO_MODE=true
            shift
            ;;
        --manual)
            MANUAL_MODE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --fog       Use fog simulation instead of basic simulation"
            echo "  --auto      Auto-launch all components (requires terminal emulator)"
            echo "  --manual    Manual mode - provides step-by-step instructions"
            echo "  -h, --help  Show this help message"
            echo ""
            echo "System Components:"
            echo "  1. ROS Core"
            echo "  2. HoloOcean Simulation (basic or fog)"
            echo "  3. GB Planner"
            echo "  4. Confidence Visualization"
            echo ""
            echo "Monitoring Commands:"
            echo "  rostopic echo /current_fog_density  - Check fog density"
            echo "  rostopic echo /robot_position       - Check robot position"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if workspaces are built
print_step "Checking workspace builds..."
check_workspace_built ~/gbplanner2_ws "GB Planner" || exit 1
check_workspace_built ~/simulation_ws "Simulation" || exit 1
check_workspace_built ~/confidence_ws "Confidence" || exit 1

# Check if voxblox is built (optional)
if [ -f ~/voxblox_ws/devel/setup.bash ]; then
    print_status "✓ All workspaces are built"
else
    print_warning "Voxblox workspace not found, but continuing..."
fi

if [ "$AUTO_MODE" = true ]; then
    print_step "Starting system in auto mode..."
    
    # Start roscore
    print_status "Starting ROS Core..."
    if ! check_ros_core; then
        open_terminal "ROS Core" "roscore"
        sleep 3
    else
        print_status "ROS Core already running"
    fi
    
    # Start simulation
    print_status "Starting HoloOcean Simulation ($SIMULATION_TYPE mode)..."
    if [ "$SIMULATION_TYPE" = "fog" ]; then
        SIMULATION_CMD="cd ~/simulation_ws && source devel/setup.bash && roslaunch holoocean_ros fog_holoocean.launch"
    else
        SIMULATION_CMD="cd ~/simulation_ws && source devel/setup.bash && roslaunch holoocean_ros holoocean_sim.launch"
    fi
    open_terminal "HoloOcean Simulation" "$SIMULATION_CMD"
    
    wait_for_user "Wait for simulation to fully load, then press Enter to continue..."
    
    # Start GB Planner
    print_status "Starting GB Planner..."
    GBPLANNER_CMD="cd ~/gbplanner2_ws && source devel/setup.bash && roslaunch gbplanner auv_sim.launch"
    open_terminal "GB Planner" "$GBPLANNER_CMD"
    
    sleep 2
    
    # Start Confidence Visualization
    print_status "Starting Confidence Visualization..."
    CONFIDENCE_CMD="cd ~/confidence_ws && source devel/setup.bash && roslaunch voxel_viz confidence_visualization.launch"
    open_terminal "Confidence Visualization" "$CONFIDENCE_CMD"
    
    print_status "All components launched!"
    
elif [ "$MANUAL_MODE" = true ]; then
    print_step "Starting system in manual mode..."
    
    echo ""
    print_instruction "Please follow these steps to start the system:"
    echo ""
    
    print_step "Step 1: Start ROS Core"
    echo "Open a new terminal and run:"
    echo -e "${YELLOW}roscore${NC}"
    wait_for_user "After ROS Core starts, press Enter to continue"
    
    print_step "Step 2: Start HoloOcean Simulation"
    echo "Open a new terminal and run:"
    echo -e "${YELLOW}cd ~/simulation_ws${NC}"
    echo -e "${YELLOW}source devel/setup.bash${NC}"
    if [ "$SIMULATION_TYPE" = "fog" ]; then
        echo -e "${YELLOW}roslaunch holoocean_ros fog_holoocean.launch${NC}"
    else
        echo -e "${YELLOW}roslaunch holoocean_ros holoocean_sim.launch${NC}"
    fi
    wait_for_user "After simulation loads completely, press Enter to continue"
    
    print_step "Step 3: Start GB Planner"
    echo "Open a new terminal and run:"
    echo -e "${YELLOW}cd ~/gbplanner2_ws${NC}"
    echo -e "${YELLOW}source devel/setup.bash${NC}"
    echo -e "${YELLOW}roslaunch gbplanner auv_sim.launch${NC}"
    wait_for_user "After GB Planner starts, press Enter to continue"
    
    print_step "Step 4: Start Confidence Visualization"
    echo "Open a new terminal and run:"
    echo -e "${YELLOW}cd ~/confidence_ws${NC}"
    echo -e "${YELLOW}source devel/setup.bash${NC}"
    echo -e "${YELLOW}roslaunch voxel_viz confidence_visualization.launch${NC}"
    wait_for_user "After visualization starts, press Enter to continue"
    
    print_status "All components should now be running!"
    
else
    # Interactive mode (default)
    print_step "Starting system in interactive mode..."
    
    # Check if roscore is running
    if ! check_ros_core; then
        print_status "Starting ROS Core..."
        echo "Please run the following command in a new terminal:"
        echo -e "${YELLOW}roscore${NC}"
        wait_for_user "After ROS Core starts, press Enter to continue"
    else
        print_status "ROS Core is already running"
    fi
    
    # Simulation
    print_status "Starting HoloOcean Simulation..."
    echo "Please run the following commands in a new terminal:"
    echo -e "${YELLOW}cd ~/simulation_ws${NC}"
    echo -e "${YELLOW}source devel/setup.bash${NC}"
    if [ "$SIMULATION_TYPE" = "fog" ]; then
        echo -e "${YELLOW}roslaunch holoocean_ros fog_holoocean.launch${NC}"
        print_status "Using fog simulation mode"
    else
        echo -e "${YELLOW}roslaunch holoocean_ros holoocean_sim.launch${NC}"
        print_status "Using basic simulation mode"
    fi
    wait_for_user "After simulation loads completely, press Enter to continue"
    
    # GB Planner
    print_status "Starting GB Planner..."
    echo "Please run the following commands in a new terminal:"
    echo -e "${YELLOW}cd ~/gbplanner2_ws${NC}"
    echo -e "${YELLOW}source devel/setup.bash${NC}"
    echo -e "${YELLOW}roslaunch gbplanner auv_sim.launch${NC}"
    wait_for_user "After GB Planner starts, press Enter to continue"
    
    # Confidence Visualization
    print_status "Starting Confidence Visualization..."
    echo "Please run the following commands in a new terminal:"
    echo -e "${YELLOW}cd ~/confidence_ws${NC}"
    echo -e "${YELLOW}source devel/setup.bash${NC}"
    echo -e "${YELLOW}roslaunch voxel_viz confidence_visualization.launch${NC}"
    wait_for_user "After visualization starts, press Enter to continue"
fi

# System is now running
print_status "System is now running!"
echo ""
print_instruction "System Monitoring Commands:"
echo "To check fog density:"
echo -e "${YELLOW}rostopic echo /current_fog_density${NC}"
echo ""
echo "To check robot position:"
echo -e "${YELLOW}rostopic echo /robot_position${NC}"
echo ""
echo "To see all available topics:"
echo -e "${YELLOW}rostopic list${NC}"
echo ""
echo "To see system info:"
echo -e "${YELLOW}rosnode list${NC}"
echo ""

print_instruction "To stop the system:"
echo "Press Ctrl+C in each terminal to stop the respective components"
echo "Stop them in reverse order: Visualization -> GB Planner -> Simulation -> ROS Core"
echo ""

print_status "System launch completed!"
print_status "Check the terminals for any error messages and ensure all components are running properly."