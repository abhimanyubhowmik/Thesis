#!/usr/bin/env python3
import rospy
import yaml
import os
import rospkg
import holoocean
import numpy as np
from sensor_msgs.msg import Image
from utils_shared_ros.utils import get_keyboard_listener, parse_keys, pressed_keys

class HolooceanControl:
    def __init__(self):
        self.load_config()
        self.init_ros()
        self.init_sim()
        self.init_control()

    def load_config(self):
        """Load parameters from YAML config"""
        rospack = rospkg.RosPack()
        config_path = os.path.join(rospack.get_path('holoocean_ros'), 'config', 'holo_config.yaml')
        
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        self.force = self.cfg['control']['force']

    def init_ros(self):
        """Initialize ROS components"""
        rospy.init_node('holocean_control')
        self.pub = rospy.Publisher(
            '/camera/image_raw', 
            Image, 
            queue_size=self.cfg['ros']['pub_queue_size']
        )
        self.rate = rospy.Rate(self.cfg['ros']['pub_rate'])

    def init_sim(self):
        """Initialize simulation environment"""
        sim_cfg = {
            "name": "Dam-HoveringCamera",
            "world": "SimpleUnderwater",
            "package_name": "Ocean",
            "main_agent": "auv0",
            "ticks_per_sec": self.cfg['simulation']['ticks_per_sec'],
            "agents": [{
                "agent_name": "auv0",
                "agent_type": "HoveringAUV",
                "sensors": [{
                    "sensor_type": "RGBCamera",
                    "sensor_name": "LeftCamera",
                    "socket": "CameraSocket",
                    "configuration": {
                        "CaptureWidth": self.cfg['simulation']['resolution'][0],
                        "CaptureHeight": self.cfg['simulation']['resolution'][1],
                        "FOV": 90,
                        "VSync": True  # Critical for flicker prevention
                    }
                }],
                "control_scheme": 0,
                "location": [0, 0, -10]
            }]
        }
        self.env = holoocean.make(scenario_cfg=sim_cfg)

    def init_control(self):
        """Initialize control systems"""
        self.listener = get_keyboard_listener()
        self.listener.daemon = True
        self.listener.start()

    def convert_frame(self, pixels):
        """Properly convert numpy array to ROS Image message"""
        # Ensure contiguous memory layout
        if not pixels.flags['C_CONTIGUOUS']:
            pixels = np.ascontiguousarray(pixels)

        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "holocean_camera"
        msg.height, msg.width = pixels.shape[:2]
        
        # Set encoding based on actual color format
        msg.encoding = "bgr8"  # HoloOcean uses BGR format by default
        
        # Calculate step from actual array stride
        msg.step = pixels.strides[0]  # This is critical for proper alignment
        
        # Verify dimensions match data size
        expected_size = msg.height * msg.step
        actual_size = len(pixels.tobytes())
        
        if expected_size != actual_size:
            rospy.logerr(f"Image size mismatch: {expected_size} vs {actual_size}")
            return None

        msg.data = pixels.tobytes()
        return msg

    def run(self):
        """Main control loop"""
        try:
            while not rospy.is_shutdown() and 'q' not in pressed_keys:
                # Process controls
                command = parse_keys(pressed_keys, self.force)
                self.env.act("auv0", command)
                
                # Get state and publish
                state = self.env.tick()
                
                if "LeftCamera" in state:
                    # Get raw pixel data
                    raw_pixels = state["LeftCamera"]
                    
                    # Convert BGR to RGB if needed by your consumers
                    # rgb_pixels = cv2.cvtColor(raw_pixels, cv2.COLOR_BGR2RGB)
                    
                    # Create message with proper formatting
                    img_msg = self.convert_frame(raw_pixels)
    
                    if img_msg is not None:
                        self.pub.publish(img_msg)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.env.__exit__()
            rospy.loginfo("Clean shutdown complete")

if __name__ == "__main__":
    controller = HolooceanControl()
    controller.run()