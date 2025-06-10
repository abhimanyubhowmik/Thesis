#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
from std_msgs.msg import Header

class PoseConverter:
    def __init__(self):
        rospy.init_node('pose_converter', anonymous=True)
        rospy.loginfo("Converting and publishing pose messages")
        
        # Subscribe to the original pose topic from HoloOcean
        self.pose_sub = rospy.Subscriber('/holoocean/raw_pose', Pose, self.pose_callback)
        
        # Publisher for the PoseWithCovarianceStamped message that GB Planner expects
        self.pose_with_cov_pub = rospy.Publisher('/holoocean/pose_with_covariance', 
                                                PoseWithCovarianceStamped, queue_size=10)
        
    def pose_callback(self, msg):
        # Create a PoseWithCovarianceStamped message
        pose_with_cov = PoseWithCovarianceStamped()
        
        # Set header
        pose_with_cov.header = Header()
        pose_with_cov.header.stamp = rospy.Time.now()
        pose_with_cov.header.frame_id = "world"  # Set appropriate frame
        
        # Copy the pose data
        pose_with_cov.pose.pose = msg
        
        # Set covariance (default to low uncertainty)
        # This is a 6x6 covariance matrix (36 elements) for [x, y, z, roll, pitch, yaw]
        # Diagonal elements represent variance in each dimension
        low_covariance = 0.01
        pose_with_cov.pose.covariance = [low_covariance if i == j else 0.0 for i in range(36) for j in range(36)]
        
        # Publish the transformed message
        self.pose_with_cov_pub.publish(pose_with_cov)

if __name__ == '__main__':
    converter = PoseConverter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down pose converter")