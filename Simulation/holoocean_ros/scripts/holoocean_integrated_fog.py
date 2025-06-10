#!/usr/bin/env python3
import rospy
import holoocean
import numpy as np
import yaml
import time
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseArray

from holoocean_ros.publishers_conf_fog import ImagePublisher, SensorPublisher, PathPublisher
from holoocean_ros.control_system import ControlSystem
from holoocean_ros.callbacks import CallbackHandler
from holoocean_ros.utils import get_sim_time

# Import the fog density controller
from holoocean_ros.fog_density import FogDensityController

class HoloOceanNode:
    def __init__(self):
        rospy.init_node('holoocean_integrated', anonymous=True)
        self.ticks_count = 0
        self.ros_start_time = time.time()
        
        # Load PID gains and Camera intrinsics from config file
        self._load_pid_config()
        intrinsics_path = rospy.get_param('~camera_intrinsics_file', 
                                        'camera_intrinsics.yaml')

        # Initialize HoloOcean environment
        self.env = holoocean.make(
            rospy.get_param('~env_config', "OceanSimple-Hovering2CameraOnly-VisibilityEstimation"),
            show_viewport=rospy.get_param('~show_viewport', True)
        )
        
        # Set initial fog density (will be overridden by controller)
        initial_fog_density = rospy.get_param('~fog_density', 2.0)
        self.env.weather.set_fog_density(initial_fog_density)
        self.env.set_render_quality(rospy.get_param('~render_quality', 3))

        # Initialize fog density controller
        self.fog_controller = FogDensityController(self.env)

        # Initialize publishers
        self.image_publisher = ImagePublisher(intrinsics_path, initial_fog_density)
        self.sensor_publisher = SensorPublisher()
        self.path_publisher = PathPublisher()
        
        # Initialize controllers
        self.control_system = ControlSystem(self.pid_config)
        
        # Initialize callback handler
        self.callback_handler = CallbackHandler(self.control_system)
        
        # Setup subscribers
        self._init_subscribers()
        
        # Setup static transforms
        self.sensor_publisher.setup_static_tf()
        
        # Initialize actual path
        self.path_publisher.init_paths()
        
        # Timer for control loop
        self.trajectory_update_rate = rospy.get_param('~trajectory_update_rate', 10.0)
        self.timer = rospy.Timer(rospy.Duration(1.0/self.trajectory_update_rate), self.control_loop)
        
        rospy.loginfo("HoloOcean Integrated Node with Dynamic Fog Initialized")
        
        # Log fog controller information
        region_info = self.fog_controller.get_region_info()
        rospy.loginfo(f"Fog regions: {region_info['division_type']} division")
        rospy.loginfo(f"Low fog density: {region_info['low_fog_density']}, High fog density: {region_info['high_fog_density']}")
    
    def _load_pid_config(self):
        """Load PID parameters from YAML file"""
        pid_path = rospy.get_param('~pid_config', 'pid_config.yaml')
        with open(pid_path, 'r') as f:
            self.pid_config = yaml.safe_load(f)['pid_gains']
    
    def _init_subscribers(self):
        """Initialize all ROS subscribers"""
        rospy.Subscriber('/pci_command_path', PoseArray, 
                        self.callback_handler.path_callback)
    
    def get_sim_time(self):
        """Get current simulation time"""
        return rospy.Time.from_sec(self.ros_start_time + self.ticks_count/50)
    
    def control_loop(self, event):
        """Integrated control and publishing loop"""
        # Get environment state - This is the only place we call env.tick()
        state = self.env.tick()
        self.ticks_count += 1
        sim_time = self.get_sim_time()
        
        # Update control system
        self.control_system.update(state, sim_time)
        
        # Get command from control system
        command = self.control_system.get_command()
        
        # Send command to environment
        self.env.act("auv0", command)
        
        # Get current pose for path updates and fog density control
        if "ImuPoseSensor" in state:
            current_position = state["ImuPoseSensor"][0:3, 3]
            rotation = R.from_matrix(state["ImuPoseSensor"][0:3, 0:3]).as_quat()
            
            # Update fog density based on robot position
            self.fog_controller.update_fog_density(current_position)
            
            # Update path target and publish paths
            self.callback_handler.update_path_target(current_position)
            self.path_publisher.update_actual_path(current_position, rotation, sim_time)
            if self.callback_handler.current_path is not None:
                self.path_publisher.publish_target_path(self.callback_handler.current_path, sim_time)
        
        # Publish sensor data
        self.image_publisher.publish_images(state, sim_time)
        self.image_publisher.publish_freespace_pointcloud(state, sim_time)
        self.sensor_publisher.publish_imu(state, sim_time)
        self.sensor_publisher.publish_odometry(state, sim_time)
        
        # Publish PID errors
        self.control_system.publish_pid_errors()
    
    def shutdown(self):
        """Clean up before shutting down"""
        if hasattr(self, 'env'):
            self.env.close()


if __name__ == '__main__':
    try:
        node = HoloOceanNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.shutdown()