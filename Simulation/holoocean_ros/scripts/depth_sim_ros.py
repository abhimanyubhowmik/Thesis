#!/usr/bin/env python3
import holoocean, cv2
import numpy as np
from pynput import keyboard
import time
import signal

# ROS imports
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Imu, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros
from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import struct


# Global variables
pressed_keys = []
force = 25
ticks_count = 0

class HoloOceanPublisher:

    def __init__(self):
        rospy.init_node('holocean_publisher', anonymous=True)
        self.bridge = CvBridge()
        
        # Initialize publishers
        self.rgb_pub = rospy.Publisher('/camera/rgb/image_raw', Image, queue_size=10)
        self.depth_pub = rospy.Publisher('/camera/depth/image_raw', Image, queue_size=10)
        self.confidence_pub = rospy.Publisher('/camera/depth/confidence', Image, queue_size=10)
        # camera info publishers
        self.rgb_info_pub = rospy.Publisher('/camera/rgb/camera_info', CameraInfo, queue_size=10)
        self.depth_info_pub = rospy.Publisher('/camera/depth/camera_info', CameraInfo, queue_size=10)
        self.imu_pub = rospy.Publisher('/imu', Imu, queue_size=10)
        self.odom_pub = rospy.Publisher('/odometry', Odometry, queue_size=10)
        self.freespace_pub = rospy.Publisher('/camera/depth/freespace_points', PointCloud2, queue_size=10)
        self.depth_info = self.create_camera_info()
        
        # TF setup
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        
        # Wait a bit before sending static transforms
        rospy.sleep(1.0)  # Give time for publishers to initialize
        self.setup_static_tf()


        # HoloOcean environment setup
        self.env = holoocean.make("OceanSimple-Hovering2CameraOnly-VisibilityEstimation")
        self.env.should_render_viewport(True)
        self.env.weather.set_fog_density(density=0.9)
        self.env.set_render_quality(1)
        
        # Spawn obstacles
        # self.sample_obstacle = GenerateObstacles(self.env)
        # self.sample_obstacle.random_spawner([0, 0, 10], [0, 0, 0], 2000)

        self.ros_start_time = time.time()

    def create_camera_info(self):
        # Create camera info message with correct parameters for point cloud generation
        info = CameraInfo()
        dims = (320, 240)
        info.height = dims[1]
        info.width = dims[0]
        info.distortion_model = "plumb_bob"
        
        # Set focal length and optical center based on FOV of 90 degrees
        # For 90 degree FOV, fx = width/2 and fy = width/2
        fx = dims[0]/2  # 160.0
        fy = dims[0]/2  # 160.0
        cx = dims[0]/2  # 160.0
        cy = dims[1]/2  # 120.0
        
        info.K = [fx, 0, cx,
                0, fy, cy,
                0, 0, 1]
        
        # Set projection matrix
        info.P = [fx, 0, cx, 0,
                0, fy, cy, 0,
                0, 0, 1, 0]
        
        # Set rectification matrix to identity
        info.R = [1, 0, 0,
                0, 1, 0,
                0, 0, 1]
        
        # Zero distortion
        info.D = [0, 0, 0, 0, 0]
        
        return info
    
    def create_pointcloud_msg(self, points):
        """Convert numpy array of points to PointCloud2 message."""
        msg = PointCloud2()
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12  # 3 * float32 (4 bytes)
        msg.row_step = msg.point_step * msg.width
        msg.data = points.astype(np.float32).tobytes()
        msg.is_dense = True
        return msg

    def setup_static_tf(self):
        static_transforms = []
        
        # World to odom transform
        world_transform = TransformStamped()
        world_transform.header.stamp = rospy.Time.now()
        world_transform.header.frame_id = "world"
        world_transform.child_frame_id = "odom"
        world_transform.transform.rotation.w = 1.0
        static_transforms.append(world_transform)

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

    def publish_images(self, state):
        current_time = self.get_sim_time()

        if "Cam1RGBImg" in state:
            rgb_img = state["Cam1RGBImg"][:, :, 0:3]  # Remove alpha channel
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_img, encoding="rgb8")
            rgb_msg.header.stamp = self.get_sim_time()
            rgb_msg.header.frame_id = "camera_rgb_optical_frame"
            self.rgb_pub.publish(rgb_msg)

            # Publish RGB camera info with same timestamp
            rgb_info = CameraInfo()
            rgb_info.header.stamp = current_time
            rgb_info.header.frame_id = "camera_rgb_optical_frame"
            dims = (320, 240)
            rgb_info.height = dims[1]
            rgb_info.width = dims[0]
            rgb_info.distortion_model = "plumb_bob"
            fx = dims[0]/2  # 160.0
            fy = dims[0]/2  # 160.0
            cx = dims[0]/2  # 160.0
            cy = dims[1]/2  # 120.0
            rgb_info.K = [fx, 0.0, cx,  # fx, 0, cx
                        0.0, fy, cy,   # 0, fy, cy
                        0.0, 0.0, 1.0]
            rgb_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            rgb_info.P = [fx, 0.0, cx, 0.0, 
                        0.0, fy, cy, 0.0, 
                        0.0, 0.0, 1.0, 0.0]
            rgb_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
            self.rgb_info_pub.publish(rgb_info)


        if "Cam0DepthImg" in state:
            depth_img = state["Cam0DepthImg"].astype(np.float32)
            
            # clip_depth = clip_depth / 3000.0
            # clip_depth = (clip_depth * 255)

             # Check the shape of the depth image and handle accordingly
            if len(depth_img.shape) == 3:  # If it's (height, width, channels)
                # Take just the first channel if it's multi-channel
                depth_img = depth_img[:, :, 0]
            elif len(depth_img.shape) != 2:  # If it's not 2D at all
                rospy.logwarn(f"Unexpected depth image shape: {depth_img.shape}. Skipping confidence image.")
                return
            
            # Convert to meters and clip values to avoid divide-by-zero
            depth_meters = np.clip(depth_img, 0.1, 3000.0) / 1000.0
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
            depth_msg.header.stamp = current_time
            depth_msg.header.frame_id = "camera_depth_optical_frame"  
            self.depth_pub.publish(depth_msg)

             # Create and publish the confidence image
            confidence_msg = self.bridge.cv2_to_imgmsg(confidence.astype(np.float32), encoding="32FC1")
            confidence_msg.header.stamp = self.get_sim_time()
            confidence_msg.header.frame_id = "camera_depth_optical_frame"
            self.confidence_pub.publish(confidence_msg)

            # Publish camera info with same timestamp
            self.depth_info.header.stamp = current_time
            self.depth_info.header.frame_id = "camera_depth_optical_frame"  
            self.depth_info_pub.publish(self.depth_info)

        # Publish camera info with same timestamp as images
        # self.publish_camera_info(current_time)

    def publish_freespace_pointcloud(self, state):
        if "Cam0DepthImg" in state:
            depth_img = state["Cam0DepthImg"].astype(np.float32)
            # Clip depth to reasonable range
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
            dims = (320, 240)
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
                freespace_msg = self.create_pointcloud_msg(points_array)
                freespace_msg.header.stamp = self.get_sim_time()
                freespace_msg.header.frame_id = "camera_depth_optical_frame"
                self.freespace_pub.publish(freespace_msg)

    def publish_imu(self, state):
        if "IMUSensor" in state:
            imu_data = state["IMUSensor"]
            imu_pose = state["ImuPoseSensor"]

            imu_msg = Imu()
            imu_msg.header.stamp = self.get_sim_time()
            imu_msg.header.frame_id = "imu"
            
            # Convert to ROS conventions
            imu_msg.angular_velocity.x = imu_data[1, 0]
            imu_msg.angular_velocity.y = imu_data[1, 1]
            imu_msg.angular_velocity.z = imu_data[1, 2]
            
            imu_msg.linear_acceleration.x = imu_data[0, 0]
            imu_msg.linear_acceleration.y = imu_data[0, 1]
            imu_msg.linear_acceleration.z = imu_data[0, 2]

            self.imu_pub.publish(imu_msg)

    def publish_odometry(self, state):
        if "ImuPoseSensor" in state:
            imu_pose = state["ImuPoseSensor"]
            dynamics = state["DynamicsSensor"]

            odom_msg = Odometry()
            odom_msg.header.stamp = self.get_sim_time()
            odom_msg.header.frame_id = "odom"
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

            # Publish TF
            self.publish_tf(position, rotation)

    def publish_tf(self, position, rotation):

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
        transform.header.stamp = self.get_sim_time()
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = position[0]
        transform.transform.translation.y = position[1]
        transform.transform.translation.z = position[2]
        transform.transform.rotation.x = normalized_rotation[0]
        transform.transform.rotation.y = normalized_rotation[1]
        transform.transform.rotation.z = normalized_rotation[2]
        transform.transform.rotation.w = normalized_rotation[3]
        self.tf_broadcaster.sendTransform(transform)


    def get_sim_time(self):
        return rospy.Time.from_sec(self.ros_start_time + ticks_count/50)

    def run(self):
        global ticks_count
        try:
            while not rospy.is_shutdown():
                start_time = time.time()
                
                if 'q' in pressed_keys:
                    break

                # Process control commands
                command = parse_keys(pressed_keys, force)
                self.env.act("auv0", command)
                
                # Get state data
                state = self.env.tick()
                ticks_count += 1

                # Publish all data
                self.publish_images(state)
                self.publish_imu(state)
                self.publish_odometry(state)
                self.publish_freespace_pointcloud(state)
                time.sleep(0.01)

        finally:
            self.env.__exit__()




def parse_keys(keys, val):
    command = np.zeros(8)
    if 'i' in keys:
        command[0:4] += val
    if 'k' in keys:
        command[0:4] -= val
    if 'j' in keys:
        command[[4,7]] += val
        command[[5,6]] -= val
    if 'l' in keys:
        command[[4,7]] -= val
        command[[5,6]] += val

    if 'w' in keys:
        command[4:8] += val
    if 'x' in keys:
        command[4:8] -= val
    if 'a' in keys:
        command[[4,6]] += val
        command[[5,7]] -= val
    if 'd' in keys:
        command[[4,6]] -= val
        command[[5,7]] += val
    if 'o' in keys:
        command[[0,3]] -= val
        command[[1,2]] += val
    if 'u' in keys:
        command[[0,3]] += val
        command[[1,2]] -= val
    if 'y' in keys:
        command[[0,1]] -= val
        command[[3,2]] += val
    if 'b' in keys:
        command[[0,1]] += val
        command[[3,2]] -= val
    return command


        
   
# Keyboard listener setup
def on_press(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.append(key.char)
        pressed_keys = list(set(pressed_keys))

def on_release(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.remove(key.char)

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
    
listener.start()

if __name__ == "__main__":
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        publisher = HoloOceanPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass