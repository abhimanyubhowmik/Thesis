#!/usr/bin/env python3

import rospy
import os
import time
from subprocess import Popen, PIPE

def run_cmd(cmd):
    process = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    stdout, stderr = process.communicate()
    return stdout.decode('utf-8')

def print_topic_info(topic_name):
    """Print detailed info about a topic including publishers, subscribers and message type"""
    print(f"\n===== TOPIC: {topic_name} =====")
    
    # Get topic type
    topic_type = run_cmd(f"rostopic type {topic_name}")
    print(f"Message Type: {topic_type.strip()}")
    
    # Get publishers
    publishers = run_cmd(f"rostopic info {topic_name} | grep 'Publishers:' -A 10")
    print(f"Publishers:\n{publishers.strip()}")
    
    # Get subscribers
    subscribers = run_cmd(f"rostopic info {topic_name} | grep 'Subscribers:' -A 10")
    print(f"Subscribers:\n{subscribers.strip()}")
    
    # Check if messages are being published
    try:
        print("Latest message (timeout 1 sec):")
        msg = run_cmd(f"timeout 1 rostopic echo {topic_name} -n 1")
        if msg:
            print(f"✅ Message received on {topic_name}")
        else:
            print(f"❌ No messages received on {topic_name} within timeout")
    except:
        print(f"❌ Error checking messages on {topic_name}")

def check_connections():
    """Check ROS connections for GB Planner and HoloOcean integration"""
    print("\n========== CHECKING ROS CONNECTIONS ==========")
    
    # List all nodes
    print("\n----- ACTIVE NODES -----")
    nodes = run_cmd("rosnode list")
    print(nodes)
    
    # Check critical topics
    critical_topics = [
        "/holoocean/current_pose",
        "/holoocean/pose_with_covariance",
        "/camera/depth/points",
        "/odometry",
        "/odometry_throttled"
    ]
    
    print("\n----- CRITICAL TOPICS -----")
    for topic in critical_topics:
        print_topic_info(topic)
        
    # Check TF tree
    print("\n----- TF TREE -----")
    tf_tree = run_cmd("rosrun tf view_frames")
    latest_tf = run_cmd("rosrun tf tf_echo world base_link 2>/dev/null")
    print(f"TF from world to base_link:\n{latest_tf}")
    
    print("\n========== CONNECTION CHECK COMPLETE ==========")

if __name__ == "__main__":
    rospy.init_node('connection_checker', anonymous=True)
    time.sleep(2)  # Give time for connections to establish
    check_connections()