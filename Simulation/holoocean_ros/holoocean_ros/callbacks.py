#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseArray
from trajectory_msgs.msg import MultiDOFJointTrajectory

class CallbackHandler:
    def __init__(self, control_system):
        """
        Initialize the callback handler
        
        Args:
            control_system: The control system to update with new target poses
        """
        self.control_system = control_system
        self.current_path = None
        self.path_index = 0
        rospy.loginfo("Callback Handler initialized")
    
    def trajectory_callback(self, msg):
        """
        Process MultiDOFJointTrajectory message
        
        Args:
            msg: The trajectory message
        """
        if not msg.points:
            return
            
        # Get the first point in the trajectory
        point = msg.points[0]
        
        # Extract position and orientation
        position = [point.transforms[0].translation.x,
                   point.transforms[0].translation.y,
                   point.transforms[0].translation.z]
                   
        orientation = [point.transforms[0].rotation.x,
                      point.transforms[0].rotation.y,
                      point.transforms[0].rotation.z,
                      point.transforms[0].rotation.w]
        
        # Update target pose in the control system
        self.control_system.set_target_pose(position, orientation)
        
        rospy.loginfo(f"New target: Position {position}, Orientation {orientation}")
    
    def path_callback(self, msg):
        """
        Process PoseArray message
        
        Args:
            msg: The PoseArray message containing a path
        """
        if not msg.poses:
            rospy.logwarn("Received empty PoseArray message")
            return
            
        rospy.loginfo(f"Path callback received {len(msg.poses)} poses")
        self.current_path = msg.poses
        self.path_index = 0
        
        # Set initial target to the first point in the path
        pose = self.current_path[self.path_index]
        position = [pose.position.x, pose.position.y, pose.position.z]
        orientation = [pose.orientation.x, pose.orientation.y, 
                    pose.orientation.z, pose.orientation.w]
                    
        # Update target pose in the control system
        self.control_system.set_target_pose(position, orientation)
        
        rospy.loginfo(f"New path received with {len(self.current_path)} points")
    
    def update_path_target(self, current_position, current_velocity=0.0):
        """
        Move to next waypoint if close enough to current target
        
        Args:
            current_position: Current position of the vehicle
            current_velocity: Current velocity magnitude (optional)
        
        Returns:
            bool: True if path target was updated, False otherwise
        """
        if self.current_path is None or self.path_index >= len(self.current_path) - 1:
            return False
            
        # Check distance to current target
        target_pos = self.control_system.current_target_pose['position']
        distance = np.linalg.norm(current_position - target_pos)
        
        # If we're close enough, move to the next point
        if distance < 0.1:  # 0.1m threshold, adjust as needed
            self.path_index += 1
            pose = self.current_path[self.path_index]
            position = [pose.position.x, pose.position.y, pose.position.z]
            orientation = [pose.orientation.x, pose.orientation.y, 
                          pose.orientation.z, pose.orientation.w]
                          
            # Update target pose in the control system
            self.control_system.set_target_pose(position, orientation)
            
            rospy.loginfo(f"Moving to next waypoint: {self.path_index}/{len(self.current_path)}")
            return True
            
        return False