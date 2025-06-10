#!/usr/bin/env python3
import rospy
import numpy as np
import time
import yaml
import rospy
from sensor_msgs.msg import CameraInfo, PointCloud2, PointField


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

def load_camera_intrinsics(yaml_file_path):
    """Load camera intrinsics from YAML file"""
    try:
        with open(yaml_file_path, 'r') as file:
            camera_info = yaml.safe_load(file)
            print(camera_info)
        
        # Get intrinsics parameters
        intrinsics = camera_info.get('intrinsics', {})
        resolution = camera_info.get('resolution', {})
        distortion = camera_info.get('distortion', {})
        
        # Create a dictionary with all parameters
        params = {
            'fx': intrinsics.get('fx', 160.0),
            'fy': intrinsics.get('fy', 160.0),
            'cx': intrinsics.get('cx', 160.0),
            'cy': intrinsics.get('cy', 120.0),
            'width': resolution.get('width', 320),
            'height': resolution.get('height', 240),
            'distortion_model': distortion.get('model', 'radtan'),
            'k1': distortion.get('parameters', {}).get('k1', 0.0),
            'k2': distortion.get('parameters', {}).get('k2', 0.0),
            'p1': distortion.get('parameters', {}).get('p1', 0.0),
            'p2': distortion.get('parameters', {}).get('p2', 0.0)
        }
        
        return params
    except Exception as e:
        rospy.logerr(f"Failed to load camera intrinsics: {e}")
        # Return default values if file loading fails
        return {
            'fx': 525.0, 'fy': 525.0, 'cx': 160.0, 'cy': 120.0,
            'width': 320, 'height': 240, 'distortion_model': 'radtan',
            'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0
        }

def create_camera_info_from_params(params, frame_id=None, baseline=None):
    """Create a CameraInfo message from parameters"""
    info_msg = CameraInfo()
    if frame_id:
        info_msg.header.frame_id = frame_id
    
    # Fill in camera info
    info_msg.width = params['width']
    info_msg.height = params['height']
    info_msg.distortion_model = 'plumb_bob' if params['distortion_model'] == 'radtan' else params['distortion_model']
    
    # Intrinsic camera matrix
    fx, fy = params['fx'], params['fy']
    cx, cy = params['cx'], params['cy']
    info_msg.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    
    # Rectification matrix (identity for pinhole camera)
    info_msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    
    # Projection matrix
    if baseline is not None:
        # For stereo camera with baseline
        tx = -fx * baseline  # Baseline term
        info_msg.P = [fx, 0.0, cx, tx, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    else:
        # For monocular camera
        info_msg.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    
    # Distortion parameters
    info_msg.D = [params['k1'], params['k2'], params['p1'], params['p2'], 0.0]
    
    return info_msg

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