#!/usr/bin/env python3
import rospy
import holoocean
from sensor_msgs.msg import Image
from utils_shared_ros.utils import (
    get_keyboard_listener,
    parse_keys,
    load_config,
    pressed_keys,
    force
)

def main():
    rospy.init_node('holocean_control')
    pub = rospy.Publisher('/camera/image_raw', Image, queue_size=1)
    
    # Initialize keyboard listener
    listener = get_keyboard_listener()
    listener.daemon = True
    listener.start()
    
    # Load configuration
    config = load_config('holoocean_ros','sim.json')
    
    # Create environment
    env = holoocean.make(scenario_cfg=config)
    
    rate = rospy.Rate(30)
    
    try:
        while not rospy.is_shutdown() and 'q' not in pressed_keys:
            command = parse_keys(pressed_keys, force)
            env.act("auv0", command)
            state = env.tick()
            
            if "LeftCamera" in state:
                pixels = state["LeftCamera"][:, :, 0:3]
                img_msg = Image()
                img_msg.header.stamp = rospy.Time.now()
                img_msg.height, img_msg.width = pixels.shape[:2]
                img_msg.encoding = "rgb8"
                img_msg.step = 3 * img_msg.width
                img_msg.data = pixels.tobytes()
                pub.publish(img_msg)
            
            rate.sleep()
            
    except KeyboardInterrupt:
        pass
    finally:
        env.__exit__()
        rospy.loginfo("Shutdown complete!")

if __name__ == "__main__":
    main()