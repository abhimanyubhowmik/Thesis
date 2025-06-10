#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import tf2_ros
import struct
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Imu, CameraInfo, PointCloud2, PointField
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseArray, Pose, Transform, Twist, Vector3, Quaternion, PoseStamped, TransformStamped

from holoocean_ros.utils import create_camera_info_from_params, create_pointcloud_msg, load_camera_intrinsics

class ImagePublisher:
    def __init__(self,intrinsics_path,left_intrinsics_path,right_intrinsics_path):
        self.bridge = CvBridge()

        # Load camera intrinsics
        self.camera_params = load_camera_intrinsics(intrinsics_path)

        # Load camera intrinsics for both cameras
        self.cam0_params = load_camera_intrinsics(left_intrinsics_path) 
        self.cam1_params = load_camera_intrinsics(right_intrinsics_path)
            
        # Publishers for camera data
        self.rgb_pub = rospy.Publisher('/camera/rgb/image_raw', Image, queue_size=10)
        self.depth_pub = rospy.Publisher('/camera/depth/image_raw', Image, queue_size=10)
        self.confidence_pub = rospy.Publisher('/camera/depth/confidence', Image, queue_size=10)
        self.rgb_info_pub = rospy.Publisher('/camera/rgb/camera_info', CameraInfo, queue_size=10)
        self.depth_info_pub = rospy.Publisher('/camera/depth/camera_info', CameraInfo, queue_size=10)
        self.freespace_pub = rospy.Publisher('/camera/depth/freespace_points', PointCloud2, queue_size=10)

         # Publishers for RAFT stereo
        self.cam0_pub = rospy.Publisher('/alphasense_driver_ros/cam0', Image, queue_size=10)
        self.cam1_pub = rospy.Publisher('/alphasense_driver_ros/cam1', Image, queue_size=10)
        self.cam0_info_pub = rospy.Publisher('/alphasense_driver_ros/cam0/camera_info', CameraInfo, queue_size=10)
        self.cam1_info_pub = rospy.Publisher('/alphasense_driver_ros/cam1/camera_info', CameraInfo, queue_size=10)
            
        # Camera info 
        self.depth_info = create_camera_info_from_params(self.camera_params, frame_id="camera_depth_optical_frame")
    
    def publish_images(self, state, sim_time):
        """Publish RGB and depth images with camera info"""
        # Publish left camera (Cam0RGBImg) - now as monochromatic
        if "Cam0RGBImg" in state:
            left_img = state["Cam0RGBImg"][:, :, 0:3]  # Remove alpha channel
            
            # Convert to grayscale for cam0
            left_mono = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
            
            # Resize to 240x320 if needed (current is likely 256x320)
            if left_mono.shape[0] != 240:
                left_mono = cv2.resize(left_mono, (320, 240))
                
            # Publish as mono8 image
            left_msg = self.bridge.cv2_to_imgmsg(left_mono, encoding="mono8")
            left_msg.header.stamp = sim_time
            left_msg.header.frame_id = "cam0_optical_frame"
            self.cam0_pub.publish(left_msg)
            
            # Publish camera info using left camera parameters
            cam0_info = create_camera_info_from_params(
                self.cam0_params, frame_id="cam0_optical_frame",
                baseline=-0.05) # Use cam0_params
            cam0_info.header.stamp = sim_time
            self.cam0_info_pub.publish(cam0_info)
        
        # Publish right camera (Cam1RGBImg) - now as monochromatic
        if "Cam1RGBImg" in state:
            right_img = state["Cam1RGBImg"][:, :, 0:3]  # Remove alpha channel
            
            # Convert to grayscale for cam1
            right_mono = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
            
            # Resize to 240x320 if needed
            if right_mono.shape[0] != 240:
                right_mono = cv2.resize(right_mono, (320, 240))
                
            # Publish as mono8 image
            right_msg = self.bridge.cv2_to_imgmsg(right_mono, encoding="mono8")
            right_msg.header.stamp = sim_time
            right_msg.header.frame_id = "cam1_optical_frame"
            self.cam1_pub.publish(right_msg)
            
            # Also publish the original RGB image for the regular RGB camera
            rgb_msg = self.bridge.cv2_to_imgmsg(right_img, encoding="rgb8")
            rgb_msg.header.stamp = sim_time
            rgb_msg.header.frame_id = "camera_rgb_optical_frame"
            self.rgb_pub.publish(rgb_msg)
            
            # Publish camera info for right camera with baseline
            cam1_info = create_camera_info_from_params(
                self.cam1_params, # Use cam1_params
                frame_id="cam1_optical_frame", 
                baseline=0.05) # You might want to load baseline from params too
            cam1_info.header.stamp = sim_time
            self.cam1_info_pub.publish(cam1_info)
            
            # Also publish regular RGB camera info
            rgb_info = create_camera_info_from_params(
                self.cam1_params, # Use cam1_params for the generic RGB info as well
                frame_id="camera_rgb_optical_frame")
            rgb_info.header.stamp = sim_time
            self.rgb_info_pub.publish(rgb_info)


        if "Cam0DepthImg" in state:
            depth_img = state["Cam0DepthImg"].astype(np.float32)
            
            # Check the shape of the depth image and handle accordingly
            if len(depth_img.shape) == 3:  # If it's (height, width, channels)
                # Take just the first channel if it's multi-channel
                depth_img = depth_img[:, :, 0]
            elif len(depth_img.shape) != 2:  # If it's not 2D at all
                rospy.logwarn(f"Unexpected depth image shape: {depth_img.shape}. Skipping confidence image.")
                return
            
            # Convert to meters and clip values to avoid divide-by-zero
            depth_meters = np.clip(depth_img, 0.1, 3000.0) / 100.0
            # Calculate confidence as 1/z²
            confidence = 1.0 / (depth_meters * depth_meters)

            # Normalize confidence to 0-1 range if needed
            max_confidence = np.max(confidence)
            if max_confidence > 0:
                confidence = confidence / max_confidence

            # Create a mask where confidence > 0.5
            confidence_mask = confidence > 0.1
            
            # Create filtered depth image (set depth values to NaN where confidence is too low)
            filtered_depth = depth_meters.copy()
            filtered_depth[~confidence_mask] = np.nan  # Use NaN for invalid points
            
            depth_msg = self.bridge.cv2_to_imgmsg(filtered_depth.astype(np.float32), encoding="32FC1")
            depth_msg.header.stamp = sim_time
            depth_msg.header.frame_id = "camera_depth_optical_frame"  
            self.depth_pub.publish(depth_msg)

            # Create and publish the confidence image
            confidence_msg = self.bridge.cv2_to_imgmsg(confidence.astype(np.float32), encoding="32FC1")
            confidence_msg.header.stamp = sim_time
            confidence_msg.header.frame_id = "camera_depth_optical_frame"
            self.confidence_pub.publish(confidence_msg)

            # Publish camera info with same timestamp
            self.depth_info.header.stamp = sim_time
            self.depth_info_pub.publish(self.depth_info)
    
    def publish_freespace_pointcloud(self, state, sim_time):
        """Generate and publish pointcloud data representing free space"""
        if "Cam0DepthImg" in state:
            depth_img = state["Cam0DepthImg"].astype(np.float32)
            
            # Check the shape of the depth image and handle accordingly
            if len(depth_img.shape) == 3:  # If it's (height, width, channels)
                # Take just the first channel if it's multi-channel
                depth_img = depth_img[:, :, 0]
            elif len(depth_img.shape) != 2:  # If it's not 2D at all
                rospy.logwarn(f"Unexpected depth image shape: {depth_img.shape}. Skipping freespace pointcloud.")
                return
                
            # Now depth_img should be 2D
            height, width = depth_img.shape
            
            # Clip depth to reasonable range
            clip_depth = np.clip(depth_img, 0, 3000.0) / 1000.0  # Convert to meter

            height, width = clip_depth.shape
            dims = (320, 256)
            fx = dims[0]/2  # 160.0
            fy = dims[0]/2  # 160.0
            cx = dims[0]/2  # 160.0
            cy = dims[1]/2  # 120.0
            
            # Sample points along each ray
            freespace_points = []
            sample_density = 10  # Number of points to sample along each ray
            
            # Use a smaller subset of pixels to keep point count manageable
            stride = 4
            for y in range(0, height, stride):
                for x in range(0, width, stride):
                    depth = clip_depth[y, x]
                    if depth > 0.1:  # Ignore invalid depth
                        # Calculate ray direction
                        ray_x = (x - cx) / fx
                        ray_y = -((y - cy) / fy)
                        ray_z = 1.0
                        ray_length = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2)
                        
                        # Normalize ray
                        ray_x /= ray_length
                        ray_y /= ray_length
                        ray_z /= ray_length
                        
                        # Sample points along the ray before the hit point
                        for i in range(1, sample_density):
                            sample_depth = depth * i / sample_density
                            px = ray_x * sample_depth
                            py = ray_y * sample_depth
                            pz = ray_z * sample_depth
                            freespace_points.append([px, py, pz])
            
            # Convert to numpy array
            if freespace_points:
                points_array = np.array(freespace_points)
                
                # Create and publish PointCloud2 message
                freespace_msg = create_pointcloud_msg(
                                points_array,
                                "camera_depth_optical_frame",  # frame_id
                                sim_time  # stamp
                            )
                freespace_msg.header.stamp = sim_time
                freespace_msg.header.frame_id = "camera_depth_optical_frame"
                self.freespace_pub.publish(freespace_msg)


class SensorPublisher:
    def __init__(self):
        # TF broadcasters
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        
        # Publishers for sensor data
        self.imu_pub = rospy.Publisher('/imu', Imu, queue_size=10)
        self.odom_pub = rospy.Publisher('/odometry', Odometry, queue_size=10)
        self.pose_pub = rospy.Publisher('/holoocean/current_pose', Pose, queue_size=10)
    
    def setup_static_tf(self):
        """Set up static transforms between frames"""
        static_transforms = []
        
        # Base link to camera transforms
        rgb_camera_transform = TransformStamped()
        rgb_camera_transform.header.stamp = rospy.Time.now()
        rgb_camera_transform.header.frame_id = "base_link"
        rgb_camera_transform.child_frame_id = "camera_rgb"
        rgb_camera_transform.transform.translation.x = 0.5
        rgb_camera_transform.transform.rotation.w = 1.0
        static_transforms.append(rgb_camera_transform)

        depth_camera_transform = TransformStamped()
        depth_camera_transform.header.stamp = rospy.Time.now()
        depth_camera_transform.header.frame_id = "base_link"
        depth_camera_transform.child_frame_id = "camera_depth"
        depth_camera_transform.transform.translation.x = 0.5
        depth_camera_transform.transform.rotation.w = 1.0
        static_transforms.append(depth_camera_transform)
        
        # Add transforms for stereo cameras
        cam0_transform = TransformStamped()
        cam0_transform.header.stamp = rospy.Time.now()
        cam0_transform.header.frame_id = "base_link"
        cam0_transform.child_frame_id = "cam0"
        cam0_transform.transform.translation.x = 0.5
        cam0_transform.transform.translation.y = -0.05  # Left camera (half of baseline)
        cam0_transform.transform.rotation.w = 1.0
        static_transforms.append(cam0_transform)
        
        cam1_transform = TransformStamped()
        cam1_transform.header.stamp = rospy.Time.now()
        cam1_transform.header.frame_id = "base_link"
        cam1_transform.child_frame_id = "cam1"
        cam1_transform.transform.translation.x = 0.5
        cam1_transform.transform.translation.y = 0.05  # Right camera (half of baseline)
        cam1_transform.transform.rotation.w = 1.0
        static_transforms.append(cam1_transform)
        
        # Add optical frames - with corrected orientation
        depth_optical_transform = TransformStamped()
        depth_optical_transform.header.stamp = rospy.Time.now()
        depth_optical_transform.header.frame_id = "camera_depth"
        depth_optical_transform.child_frame_id = "camera_depth_optical_frame"
        # Rotate to align with robot's forward direction
        # This rotates -90° around X and then -90° around Z
        q = R.from_euler('xyz', [90, 0, 90], degrees=True).as_quat()
        depth_optical_transform.transform.rotation.x = q[0]
        depth_optical_transform.transform.rotation.y = q[1]
        depth_optical_transform.transform.rotation.z = q[2]
        depth_optical_transform.transform.rotation.w = q[3]
        static_transforms.append(depth_optical_transform)

        # RGB optical frame with same correction
        rgb_optical_transform = TransformStamped()
        rgb_optical_transform.header.stamp = rospy.Time.now()
        rgb_optical_transform.header.frame_id = "camera_rgb"
        rgb_optical_transform.child_frame_id = "camera_rgb_optical_frame"
        rgb_optical_transform.transform.rotation.x = q[0]
        rgb_optical_transform.transform.rotation.y = q[1]
        rgb_optical_transform.transform.rotation.z = q[2]
        rgb_optical_transform.transform.rotation.w = q[3]
        static_transforms.append(rgb_optical_transform)
        
        # Add optical frames for stereo cameras
        cam0_optical_transform = TransformStamped()
        cam0_optical_transform.header.stamp = rospy.Time.now()
        cam0_optical_transform.header.frame_id = "cam0"
        cam0_optical_transform.child_frame_id = "cam0_optical_frame"
        cam0_optical_transform.transform.rotation.x = q[0]
        cam0_optical_transform.transform.rotation.y = q[1]
        cam0_optical_transform.transform.rotation.z = q[2]
        cam0_optical_transform.transform.rotation.w = q[3]
        static_transforms.append(cam0_optical_transform)
        
        cam1_optical_transform = TransformStamped()
        cam1_optical_transform.header.stamp = rospy.Time.now()
        cam1_optical_transform.header.frame_id = "cam1"
        cam1_optical_transform.child_frame_id = "cam1_optical_frame"
        cam1_optical_transform.transform.rotation.x = q[0]
        cam1_optical_transform.transform.rotation.y = q[1]
        cam1_optical_transform.transform.rotation.z = q[2]
        cam1_optical_transform.transform.rotation.w = q[3]
        static_transforms.append(cam1_optical_transform)

        # Send all static transforms at once
        self.static_tf_broadcaster.sendTransform(static_transforms)
    
    def publish_imu(self, state, sim_time):
        """Publish IMU sensor data"""
        if "IMUSensor" in state:
            imu_data = state["IMUSensor"]
            
            imu_msg = Imu()
            imu_msg.header.stamp = sim_time
            imu_msg.header.frame_id = "imu"
            
            # Convert to ROS conventions
            imu_msg.angular_velocity.x = imu_data[1, 0]
            imu_msg.angular_velocity.y = imu_data[1, 1]
            imu_msg.angular_velocity.z = imu_data[1, 2]
            
            imu_msg.linear_acceleration.x = imu_data[0, 0]
            imu_msg.linear_acceleration.y = imu_data[0, 1]
            imu_msg.linear_acceleration.z = imu_data[0, 2]

            self.imu_pub.publish(imu_msg)
    
    def publish_odometry(self, state, sim_time):
        """Publish odometry data and TF transform"""
        if "ImuPoseSensor" in state:
            imu_pose = state["ImuPoseSensor"]
            dynamics = state["DynamicsSensor"]

            odom_msg = Odometry()
            odom_msg.header.stamp = sim_time
            odom_msg.header.frame_id = "world"
            odom_msg.child_frame_id = "base_link"

            # Position
            position = imu_pose[0:3, 3]
            odom_msg.pose.pose.position.x = position[0]
            odom_msg.pose.pose.position.y = position[1]
            odom_msg.pose.pose.position.z = position[2]

            # Orientation
            rotation = R.from_matrix(imu_pose[0:3, 0:3]).as_quat()
            odom_msg.pose.pose.orientation.x = rotation[0]
            odom_msg.pose.pose.orientation.y = rotation[1]
            odom_msg.pose.pose.orientation.z = rotation[2]
            odom_msg.pose.pose.orientation.w = rotation[3]

            # Velocity
            odom_msg.twist.twist.linear.x = dynamics[3]
            odom_msg.twist.twist.linear.y = dynamics[4]
            odom_msg.twist.twist.linear.z = dynamics[5]
            
            self.odom_pub.publish(odom_msg)

            # Publish current pose
            pose_msg = Pose()
            pose_msg.position.x = position[0]
            pose_msg.position.y = position[1]
            pose_msg.position.z = position[2]
            pose_msg.orientation.x = rotation[0]
            pose_msg.orientation.y = rotation[1]
            pose_msg.orientation.z = rotation[2]
            pose_msg.orientation.w = rotation[3]
            self.pose_pub.publish(pose_msg)

            # Publish TF
            self.publish_tf(position, rotation, sim_time)
    
    def publish_tf(self, position, rotation, sim_time):
        """Publish transform from world to base_link"""
        # Normalize quaternion to prevent warnings
        quat_norm = np.sqrt(rotation[0]**2 + rotation[1]**2 + rotation[2]**2 + rotation[3]**2)
        if quat_norm > 0:
            normalized_rotation = [
                rotation[0]/quat_norm,
                rotation[1]/quat_norm,
                rotation[2]/quat_norm,
                rotation[3]/quat_norm
            ]
        else:
            # Default to identity quaternion if norm is zero
            normalized_rotation = [0, 0, 0, 1]

        transform = TransformStamped()
        transform.header.stamp = sim_time
        transform.header.frame_id = "world"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = position[0]
        transform.transform.translation.y = position[1]
        transform.transform.translation.z = position[2]
        transform.transform.rotation.x = normalized_rotation[0]
        transform.transform.rotation.y = normalized_rotation[1]
        transform.transform.rotation.z = normalized_rotation[2]
        transform.transform.rotation.w = normalized_rotation[3]
        self.tf_broadcaster.sendTransform(transform)


class PathPublisher:
    def __init__(self):
        # Publishers for path visualization
        self.target_path_pub = rospy.Publisher('/holoocean/target_path', Path, queue_size=10)
        self.actual_path_pub = rospy.Publisher('/holoocean/actual_path', Path, queue_size=10)
        self.target_path_poses_pub = rospy.Publisher('/holoocean/target_path_poses', PoseArray, queue_size=10)
        self.actual_path_poses_pub = rospy.Publisher('/holoocean/actual_path_poses', PoseArray, queue_size=10)
    
    def init_paths(self):
        """Initialize path objects"""
        self.actual_path = Path()
        self.actual_path.header.frame_id = "world"
        
        self.actual_path_poses = PoseArray()
        self.actual_path_poses.header.frame_id = "world"
    
    def update_actual_path(self, position, orientation, sim_time):
        """Update and publish actual path of the robot"""
        # Add to Path message
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = sim_time
        pose_stamped.header.frame_id = "world"
        pose_stamped.pose.position.x = position[0]
        pose_stamped.pose.position.y = position[1]
        pose_stamped.pose.position.z = position[2]
        pose_stamped.pose.orientation.x = orientation[0]
        pose_stamped.pose.orientation.y = orientation[1]
        pose_stamped.pose.orientation.z = orientation[2]
        pose_stamped.pose.orientation.w = orientation[3]
        
        self.actual_path.poses.append(pose_stamped)
        self.actual_path.header.stamp = sim_time
        self.actual_path_pub.publish(self.actual_path)

        # Add to PoseArray message
        pose = Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]
        pose.orientation.x = orientation[0]
        pose.orientation.y = orientation[1]
        pose.orientation.z = orientation[2]
        pose.orientation.w = orientation[3]

        self.actual_path_poses.poses.append(pose)
        self.actual_path_poses.header.stamp = sim_time
        self.actual_path_poses_pub.publish(self.actual_path_poses)
    
    def publish_target_path(self, path_poses, sim_time):
        """Convert and publish the target path"""
        if path_poses:
            # Path message
            path_msg = Path()
            path_msg.header.stamp = sim_time
            path_msg.header.frame_id = "world"
            
            # PoseArray message
            target_path_poses = PoseArray()
            target_path_poses.header.stamp = sim_time
            target_path_poses.header.frame_id = "world"
            
            for pose in path_poses:
                # For Path message
                pose_stamped = PoseStamped()
                pose_stamped.header.stamp = sim_time
                pose_stamped.header.frame_id = "world"
                pose_stamped.pose = pose
                path_msg.poses.append(pose_stamped)
                
                # For PoseArray message
                target_path_poses.poses.append(pose)
            
            # Publish both Path and PoseArray
            self.target_path_pub.publish(path_msg)
            self.target_path_poses_pub.publish(target_path_poses)