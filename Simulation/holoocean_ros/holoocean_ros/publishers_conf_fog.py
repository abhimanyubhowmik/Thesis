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
    def __init__(self,intrinsics_path,fog_density):
        self.bridge = CvBridge()

        # Load camera intrinsics
        self.camera_params = load_camera_intrinsics(intrinsics_path)
        self.fog_density = fog_density
        
        # Publishers for camera data
        self.rgb_pub = rospy.Publisher('/camera/rgb/image_raw', Image, queue_size=10)
        self.depth_pub = rospy.Publisher('/camera/depth/image_raw', Image, queue_size=10)
        self.confidence_pub = rospy.Publisher('/camera/depth/confidence', Image, queue_size=10)
        self.rgb_info_pub = rospy.Publisher('/camera/rgb/camera_info', CameraInfo, queue_size=10)
        self.depth_info_pub = rospy.Publisher('/camera/depth/camera_info', CameraInfo, queue_size=10)
        self.freespace_pub = rospy.Publisher('/camera/depth/freespace_points', PointCloud2, queue_size=10)
        
        # Camera info 
        self.depth_info = create_camera_info_from_params(self.camera_params, frame_id="camera_depth_optical_frame")
    
    def publish_images(self, state, sim_time):
        """Publish RGB and depth images with camera info"""
        if "Cam1RGBImg" in state:
            rgb_img = state["Cam1RGBImg"][:, :, 0:3]  # Remove alpha channel
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_img, encoding="rgb8")
            rgb_msg.header.stamp = sim_time
            rgb_msg.header.frame_id = "camera_rgb_optical_frame"
            self.rgb_pub.publish(rgb_msg)

            
            # Also publish regular RGB camera info
            rgb_info = create_camera_info_from_params(
                self.camera_params,
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
            
            # Determine the max depth value in raw data (likely represents "no return")
            max_raw_depth = np.max(depth_img)

            # Create a mask for points that are not at maximum depth or very close to it
            # Use a small threshold to account for potential floating-point imprecision
            threshold = 0.08 * max_raw_depth
            valid_depth_mask = depth_img < threshold
            
            valid_depth = depth_img * valid_depth_mask
            # Convert to meters
            depth_meters = np.clip(valid_depth, 0.1, 550.0) / 100.0
            depth_meters_mask = depth_meters < 5.4

            # depth_meters = np.clip(valid_depth, 0.1, 3000.0) / 100.0
            # depth_meters_mask = depth_meters < 29.9
            depth_meters = depth_meters * depth_meters_mask
            
            confidence = self.confidence_emulation(rgb_img)
            
            filtered_depth = depth_meters.copy()
            
            # Additional check: if almost everything (e.g., >95%) would be NaN after filtering, make everything NaN
            # This prevents sparse "floating walls" of noise points
            valid_points_percentage = np.sum(~np.isnan(filtered_depth)) / filtered_depth.size
            if valid_points_percentage < 0.02:
                rospy.loginfo("Almost no valid points remain after filtering. Setting all to NaN.")
                filtered_depth[:] = np.nan
            
            depth_msg = self.bridge.cv2_to_imgmsg(filtered_depth.astype(np.float32), encoding="32FC1")
            depth_msg.header.stamp = sim_time
            depth_msg.header.frame_id = "camera_depth_optical_frame"
            self.depth_pub.publish(depth_msg)
            
            # Create and publish the confidence image
            confidence_img = confidence.copy()
            #confidence_img[~valid_depth_mask] = 0  # Zero confidence for max depth points
            confidence_msg = self.bridge.cv2_to_imgmsg(confidence_img.astype(np.float32), encoding="32FC1")
            confidence_msg.header.stamp = sim_time
            confidence_msg.header.frame_id = "camera_depth_optical_frame"
            self.confidence_pub.publish(confidence_msg)
            
            # Publish camera info with same timestamp
            self.depth_info.header.stamp = sim_time
            self.depth_info_pub.publish(self.depth_info)

    def confidence_emulation(self,rgb_img):
        """Improved version of your original pipeline"""
        
        img_gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        
        # Edge detection using Sobel filters
        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Threshold and expand edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        dilated_mask = cv2.dilate(edge_magnitude.astype(np.uint8), kernel, iterations=2)
        
        # Smooth transitions and normalize
        gaussian = cv2.GaussianBlur(dilated_mask, (15,15), 5.0)
        #confidence = gaussian / np.max(gaussian) if np.max(gaussian) > 0 else gaussian
        
        # 2. Apply gamma correction to improve distribution
        #confidence = np.power(confidence, 0.5)  # Brightens the map

        # 3. Stretch contrast to use full dynamic range
        #min_val, max_val = np.percentile(confidence, [2, 98])
        #confidence = np.clip((confidence - min_val) / (max_val - min_val), 0, 1)

        confidence = gaussian * 1/255

        #confidence = confidence * np.ones_like(confidence) * 1/self.fog_density
        
        return confidence
    
    # Create a simple depth proxy based on vertical position
            
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
            clip_depth = np.clip(depth_img, 0, 1000.0) / 100.0  # Convert to meter

            height, width = clip_depth.shape
            dims = (320, 240)
            fx = 160.0  # 160.0
            fy = 160.0  # 160.0
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