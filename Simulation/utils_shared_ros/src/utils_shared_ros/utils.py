# utils.py
import rospy
import numpy as np
from pynput import keyboard
import json
import os
from sensor_msgs.msg import CameraInfo, PointCloud2, PointField

# ================== Global State ==================
pressed_keys = []
force = 25

# ================== Keyboard Listener ==================
def get_keyboard_listener():
    return keyboard.Listener(on_press=on_press, on_release=on_release)

def on_press(key):
    if hasattr(key, 'char'):
        pressed_keys.append(key.char)
        pressed_keys[:] = list(set(pressed_keys))

def on_release(key):
    if hasattr(key, 'char') and key.char in pressed_keys:
        pressed_keys.remove(key.char)

# ================== Key Parsing Function ==================
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
    if 's' in keys:
        command[4:8] -= val
    if 'a' in keys:
        command[[4,6]] += val
        command[[5,7]] -= val
    if 'd' in keys:
        command[[4,6]] -= val
        command[[5,7]] += val
    return command

#=================== Load Config ========================


def load_config(package,file_name):
    """Load configuration from JSON file"""
    try:
        # Get path to this package
        import rospkg
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path(package)
        
        config_path = os.path.join(pkg_path, 'config', file_name)
        
        with open(config_path, 'r') as f:
            return json.load(f)
            
    except Exception as e:
        rospy.logerr(f"Failed to load config: {str(e)}")
        raise

#================Holoocean Controller Node Utlis ====================


def get_sim_time(start_time, ticks_count):
    """
    Calculate and return simulation time based on the start time and tick count
    
    Args:
        start_time: The time when the simulation started
        ticks_count: The current count of ticks
        
    Returns:
        rospy.Time: The current simulation time
    """
    return rospy.Time.from_sec(start_time + ticks_count/50)

def create_camera_info():
    """
    Create camera info message with correct parameters for point cloud generation
    
    Returns:
        CameraInfo: The configured camera info message
    """
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

def create_pointcloud_msg(points, frame_id, stamp):
    """
    Convert numpy array of points to PointCloud2 message
    
    Args:
        points: Numpy array of points
        frame_id: Frame ID for the message header
        stamp: Timestamp for the message header
        
    Returns:
        PointCloud2: The pointcloud message
    """
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
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

def get_freespace_points(depth_img, focal_length):
    """
    Sample points along rays to create a freespace pointcloud
    
    Args:
        depth_img: Depth image from camera
        focal_length: Focal length of the camera
        
    Returns:
        numpy.ndarray: Array of 3D points
    """
    # Check the shape of the depth image and handle accordingly
    if len(depth_img.shape) == 3:  # If it's (height, width, channels)
        # Take just the first channel if it's multi-channel
        depth_img = depth_img[:, :, 0]
    elif len(depth_img.shape) != 2:  # If it's not 2D at all
        rospy.logwarn(f"Unexpected depth image shape: {depth_img.shape}. Skipping freespace pointcloud.")
        return np.array([])
            
    # Now depth_img should be 2D
    height, width = depth_img.shape
    
    # Clip depth to reasonable range
    clip_depth = np.clip(depth_img, 0, 3000.0) / 1000.0  # Convert to meter

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
        return np.array(freespace_points)
    else:
        return np.array([])

def calculate_confidence_mask(depth_img):
    """
    Calculate a confidence mask for depth image
    
    Args:
        depth_img: Depth image from camera
        
    Returns:
        numpy.ndarray: Confidence values for each pixel
    """
    # Check shape of depth image
    if len(depth_img.shape) == 3:
        depth_img = depth_img[:, :, 0]
    
    # Convert to meters and clip values to avoid divide-by-zero
    depth_meters = np.clip(depth_img, 0.1, 3000.0) / 1000.0
    # Calculate confidence as 1/z²
    confidence = 1.0 / (depth_meters * depth_meters)

    # Normalize confidence to 0-1 range
    max_confidence = np.max(confidence)
    if max_confidence > 0:
        confidence = confidence / max_confidence

    return confidence, confidence > 0.1  # Return confidence values and mask