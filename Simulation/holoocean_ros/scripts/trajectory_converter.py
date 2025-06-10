#!/usr/bin/env python3

import rospy
from trajectory_msgs.msg import MultiDOFJointTrajectory
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler
import numpy as np

class TrajectoryToPoseArrayConverter:
    def __init__(self):
        rospy.init_node('trajectory_to_pose_array_converter', anonymous=True)
        
        # Parameters
        self.input_topic = rospy.get_param('~input_topic', '/command/trajectory')
        self.output_topic = rospy.get_param('~output_topic', '/pci_command_path')
        self.frame_id = rospy.get_param('~frame_id', 'world')
        
        # Publisher and Subscriber
        self.pose_array_pub = rospy.Publisher(self.output_topic, PoseArray, queue_size=10)
        self.trajectory_sub = rospy.Subscriber(self.input_topic, MultiDOFJointTrajectory, self.trajectory_callback)
        
        rospy.loginfo(f"Trajectory converter initialized:")
        rospy.loginfo(f"  Input topic: {self.input_topic}")
        rospy.loginfo(f"  Output topic: {self.output_topic}")
        rospy.loginfo(f"  Frame ID: {self.frame_id}")

    def trajectory_callback(self, msg):
        """
        Convert MultiDOFJointTrajectory to PoseArray
        """
        try:
            pose_array = PoseArray()
            pose_array.header = msg.header
            pose_array.header.frame_id = self.frame_id
            pose_array.header.stamp = rospy.Time.now()
            
            # Extract poses from trajectory points
            for point in msg.points:
                if len(point.transforms) > 0:
                    # Use the first transform (assuming single DOF joint)
                    transform = point.transforms[0]
                    
                    pose = Pose()
                    
                    # Position
                    pose.position.x = transform.translation.x
                    pose.position.y = transform.translation.y
                    pose.position.z = transform.translation.z
                    
                    # Orientation
                    pose.orientation.x = transform.rotation.x
                    pose.orientation.y = transform.rotation.y
                    pose.orientation.z = transform.rotation.z
                    pose.orientation.w = transform.rotation.w
                    
                    pose_array.poses.append(pose)
                    
                elif len(point.velocities) > 0:
                    # If no transforms but velocities exist, create pose from velocities
                    # This is a fallback case - you might need to integrate velocities
                    velocity = point.velocities[0]
                    
                    pose = Pose()
                    pose.position.x = velocity.linear.x
                    pose.position.y = velocity.linear.y
                    pose.position.z = velocity.linear.z
                    
                    # Convert angular velocity to quaternion (simplified)
                    roll = velocity.angular.x
                    pitch = velocity.angular.y
                    yaw = velocity.angular.z
                    
                    q = quaternion_from_euler(roll, pitch, yaw)
                    pose.orientation.x = q[0]
                    pose.orientation.y = q[1]
                    pose.orientation.z = q[2]
                    pose.orientation.w = q[3]
                    
                    pose_array.poses.append(pose)
            
            # Publish the converted pose array
            if len(pose_array.poses) > 0:
                self.pose_array_pub.publish(pose_array)
                rospy.logdebug(f"Published PoseArray with {len(pose_array.poses)} poses")
            else:
                rospy.logwarn("Received trajectory with no valid poses")
                
        except Exception as e:
            rospy.logerr(f"Error converting trajectory: {str(e)}")

    def run(self):
        """
        Keep the node running
        """
        rospy.loginfo("Trajectory to PoseArray converter node is running...")
        rospy.spin()

if __name__ == '__main__':
    try:
        converter = TrajectoryToPoseArrayConverter()
        converter.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Trajectory converter node interrupted")
    except Exception as e:
        rospy.logerr(f"Error starting trajectory converter: {str(e)}")