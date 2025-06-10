#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Vector3
import tf.transformations as tf_trans
import threading

class ConfidenceVolumeVisualizer:
    def __init__(self):
        rospy.init_node('confidence_volume_visualizer', anonymous=True)
        
        # Parameters
        self.cube_size = rospy.get_param('~cube_size', 5.0)  # Size of cubic volumes in meters
        self.confidence_topic = rospy.get_param('~confidence_topic', '/gbplanner_node/surface_pointcloud_confidence')
        self.frame_id = rospy.get_param('~frame_id', 'world')
        self.visualize_interval = rospy.get_param('~visualize_interval', 2.0)  # Seconds
        
        # FIXED: Don't use fixed confidence bounds - let them adapt dynamically
        self.use_fixed_bounds = rospy.get_param('~use_fixed_bounds', False)
        self.confidence_min = rospy.get_param('~confidence_min', 0.0) if self.use_fixed_bounds else None
        self.confidence_max = rospy.get_param('~confidence_max', 10.0) if self.use_fixed_bounds else None
        
        self.max_markers = rospy.get_param('~max_markers', 1000)  # Maximum number of markers to display
        self.min_points_per_voxel = rospy.get_param('~min_points_per_voxel', 5)  # Minimum points to consider voxel
        
        # NEW: Parameters for better surface area estimation
        self.use_point_density_area = rospy.get_param('~use_point_density_area', True)
        self.adaptive_color_scaling = rospy.get_param('~adaptive_color_scaling', True)
        
        rospy.loginfo(f"Subscribing to topic: {self.confidence_topic}")
        rospy.loginfo(f"Using fixed bounds: {self.use_fixed_bounds}")
        rospy.loginfo(f"Adaptive color scaling: {self.adaptive_color_scaling}")
        
        # Subscribers
        self.confidence_sub = rospy.Subscriber(
            self.confidence_topic, PointCloud2, self.confidence_callback, queue_size=1)
        
        # Add timer to check subscription status
        rospy.Timer(rospy.Duration(5.0), self.check_subscription_status)
        
        # Publishers
        self.marker_pub = rospy.Publisher(
            '/confidence_visualization/cubes', MarkerArray, queue_size=1)
        
        # Data containers with thread safety
        self.data_lock = threading.Lock()
        self.points_dict = {}  # Key: (x_idx, y_idx, z_idx), Value: List of (point, confidence)
        self.results_dict = {}  # Key: (x_idx, y_idx, z_idx), Value: (confidence/surface_area)
        self.published_marker_ids = set()  # Track published marker IDs for cleanup
        
        # NEW: Statistics tracking for better debugging
        self.stats_history = []
        self.max_history = 10
        
        # Timer for visualization
        rospy.Timer(rospy.Duration(self.visualize_interval), self.visualize_cubes)
        
        rospy.loginfo("Confidence Volume Visualizer initialized. Waiting for pointcloud data...")
    
    def check_subscription_status(self, event=None):
        """Check if we are receiving messages from the confidence topic."""
        try:
            all_topics = rospy.get_published_topics()
            topic_found = False
            topic_type = None
            
            for topic, msg_type in all_topics:
                if topic == self.confidence_topic:
                    topic_found = True
                    topic_type = msg_type
                    break
            
            if not topic_found:
                rospy.logwarn(f"Topic {self.confidence_topic} is not being published. Available topics:")
                for topic, msg_type in all_topics:
                    if "confidence" in topic.lower() or "pointcloud" in topic.lower():
                        rospy.logwarn(f"  {topic} [{msg_type}]")
            else:
                rospy.loginfo(f"Topic {self.confidence_topic} is available with type {topic_type}")
                
                # Check if we have any subscribers
                pub_count = self.marker_pub.get_num_connections()
                rospy.loginfo(f"Number of subscribers to our visualization: {pub_count}")
        except Exception as e:
            rospy.logwarn(f"Error checking subscription status: {e}")
    
    def point_to_grid_idx(self, point):
        """Convert a 3D point to grid indices based on cube_size."""
        x_idx = int(np.floor(point[0] / self.cube_size))  # Use floor instead of //
        y_idx = int(np.floor(point[1] / self.cube_size))
        z_idx = int(np.floor(point[2] / self.cube_size))
        return (x_idx, y_idx, z_idx)
    
    def confidence_callback(self, cloud_msg):
        """Process incoming confidence pointcloud."""
        try:
            rospy.loginfo("Received pointcloud with %d points", cloud_msg.width * cloud_msg.height)
            
            # Create new data containers
            new_points_dict = {}
            confidence_values = []  # Track all confidence values for statistics
            
            # Process points
            point_count = 0
            valid_point_count = 0
            
            for point in pc2.read_points(cloud_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
                x, y, z, confidence = point
                point_count += 1
                
                # Skip points with zero or invalid confidence
                if confidence <= 0 or not np.isfinite(confidence):
                    continue
                
                # Skip points with invalid coordinates
                if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                    continue
                
                valid_point_count += 1
                confidence_values.append(confidence)
                
                # Get grid indices
                idx = self.point_to_grid_idx((x, y, z))
                
                # Add point to grid
                if idx not in new_points_dict:
                    new_points_dict[idx] = []
                
                new_points_dict[idx].append((np.array([x, y, z]), confidence))
            
            # NEW: Log confidence statistics
            if confidence_values:
                conf_array = np.array(confidence_values)
                rospy.loginfo(f"Processed {point_count} total points, {valid_point_count} valid points")
                rospy.loginfo(f"Confidence stats - Min: {np.min(conf_array):.3f}, Max: {np.max(conf_array):.3f}, Mean: {np.mean(conf_array):.3f}, Std: {np.std(conf_array):.3f}")
                rospy.loginfo(f"Found {len(new_points_dict)} non-empty grid cells")
            
            # Update data with thread safety
            with self.data_lock:
                self.points_dict = new_points_dict
                # Process grid cells immediately after updating points
                self.process_grid_cells()
                
        except Exception as e:
            rospy.logerr(f"Error in confidence_callback: {e}")
    
    def process_grid_cells(self):
        """Calculate confidence per surface area for each grid cell."""
        try:
            new_results_dict = {}
            cell_stats = []
            
            for idx, points_data in self.points_dict.items():
                if len(points_data) < self.min_points_per_voxel:  # Use configurable minimum
                    continue
                
                # Extract points and confidences
                points = [p[0] for p in points_data]
                confidences = [p[1] for p in points_data]
                
                # Convert to numpy arrays
                points_array = np.array(points)
                confidences_array = np.array(confidences)
                
                # IMPROVED: Better surface area estimation
                surface_area = self.estimate_surface_area(points_array)
                
                # Calculate total confidence
                total_confidence = np.sum(confidences_array)
                
                # Calculate confidence per surface area
                conf_per_surface = total_confidence / surface_area
                
                # Store result
                new_results_dict[idx] = conf_per_surface
                
                # Track statistics
                cell_stats.append({
                    'idx': idx,
                    'num_points': len(points_data),
                    'total_confidence': total_confidence,
                    'surface_area': surface_area,
                    'conf_per_surface': conf_per_surface,
                    'mean_confidence': np.mean(confidences_array),
                    'std_confidence': np.std(confidences_array)
                })
            
            # NEW: Log detailed statistics
            if cell_stats:
                conf_per_surface_values = [s['conf_per_surface'] for s in cell_stats]
                cps_array = np.array(conf_per_surface_values)
                
                rospy.loginfo(f"Cell statistics - Processed {len(cell_stats)} cells:")
                rospy.loginfo(f"  Conf/Surface - Min: {np.min(cps_array):.3f}, Max: {np.max(cps_array):.3f}, Mean: {np.mean(cps_array):.3f}")
                rospy.loginfo(f"  Surface areas - Min: {min(s['surface_area'] for s in cell_stats):.3f}, Max: {max(s['surface_area'] for s in cell_stats):.3f}")
                
                # Log top and bottom cells for debugging
                sorted_cells = sorted(cell_stats, key=lambda x: x['conf_per_surface'])
                rospy.loginfo("Bottom 3 cells (lowest conf/surface):")
                for cell in sorted_cells[:3]:
                    rospy.loginfo(f"  Cell {cell['idx']}: {cell['conf_per_surface']:.3f} ({cell['num_points']} pts, surf: {cell['surface_area']:.3f})")
                
                rospy.loginfo("Top 3 cells (highest conf/surface):")
                for cell in sorted_cells[-3:]:
                    rospy.loginfo(f"  Cell {cell['idx']}: {cell['conf_per_surface']:.3f} ({cell['num_points']} pts, surf: {cell['surface_area']:.3f})")
            
            # Update results dictionary
            self.results_dict = new_results_dict
            
            # Store statistics for adaptive scaling
            if conf_per_surface_values:
                self.stats_history.append({
                    'timestamp': rospy.Time.now(),
                    'min': np.min(cps_array),
                    'max': np.max(cps_array),
                    'mean': np.mean(cps_array),
                    'std': np.std(cps_array),
                    'count': len(cps_array)
                })
                
                # Keep only recent history
                if len(self.stats_history) > self.max_history:
                    self.stats_history = self.stats_history[-self.max_history:]
            
        except Exception as e:
            rospy.logerr(f"Error in process_grid_cells: {e}")
    
    def estimate_surface_area(self, points_array):
        """Improved surface area estimation."""
        if self.use_point_density_area:
            # Method 1: Point density based estimation
            # Estimate surface area based on point density and spatial distribution
            num_points = len(points_array)
            
            # Get the bounding box
            min_coords = np.min(points_array, axis=0)
            max_coords = np.max(points_array, axis=0)
            ranges = max_coords - min_coords
            
            # If points are very clustered, use a minimum area
            min_area = 0.01  # 1 cm²
            
            # Estimate area based on the assumption that points are roughly uniformly distributed on a surface
            # Use the two largest dimensions to estimate surface area
            sorted_ranges = np.sort(ranges)
            if sorted_ranges[-1] > 0.01 and sorted_ranges[-2] > 0.01:  # If we have significant spread in 2 dimensions
                surface_area = sorted_ranges[-1] * sorted_ranges[-2]  # Rectangle approximation
            else:
                # If points are too clustered, use point density
                surface_area = num_points * 0.001  # Assume each point represents 1 mm²
            
            return max(surface_area, min_area)
        
        else:
            # Method 2: SVD-based estimation (original method but improved)
            if len(points_array) < 3:
                return 0.01  # Minimum area
            
            mean_point = np.mean(points_array, axis=0)
            centered_points = points_array - mean_point
            
            try:
                U, S, Vt = np.linalg.svd(centered_points, full_matrices=False)
                
                # If we have at least 2 significant dimensions
                if len(S) >= 2:
                    # Use the two largest singular values to estimate area
                    surface_area = np.pi * S[0] * S[1]  # Elliptical approximation
                else:
                    surface_area = S[0] ** 2  # Linear case
                
                return max(surface_area, 0.01)  # Minimum 1 cm²
                
            except Exception as svd_error:
                rospy.logdebug(f"SVD failed: {svd_error}")
                # Fallback to bounding box
                ranges = np.max(points_array, axis=0) - np.min(points_array, axis=0)
                surface_area = max(ranges[0] * ranges[1], ranges[0] * ranges[2], ranges[1] * ranges[2])
                return max(surface_area, 0.01)
    
    def get_adaptive_bounds(self):
        """Get adaptive color scaling bounds based on recent statistics."""
        if not self.stats_history or not self.adaptive_color_scaling:
            return None, None
        
        # Use recent statistics to set bounds
        recent_mins = [s['min'] for s in self.stats_history[-3:]]  # Last 3 measurements
        recent_maxs = [s['max'] for s in self.stats_history[-3:]]
        recent_means = [s['mean'] for s in self.stats_history[-3:]]
        recent_stds = [s['std'] for s in self.stats_history[-3:]]
        
        avg_min = np.mean(recent_mins)
        avg_max = np.mean(recent_maxs)
        avg_mean = np.mean(recent_means)
        avg_std = np.mean(recent_stds)
        
        # Use mean ± 2*std as bounds to capture most of the variation
        adaptive_min = max(avg_mean - 2*avg_std, avg_min)
        adaptive_max = min(avg_mean + 2*avg_std, avg_max)
        
        # Ensure we have some spread
        if adaptive_max - adaptive_min < avg_std:
            adaptive_min = avg_mean - avg_std
            adaptive_max = avg_mean + avg_std
        
        return adaptive_min, adaptive_max
    
    def visualize_cubes(self, event=None):
        """Publish marker array for visualizing confidence volumes."""
        try:
            # Create a local copy of the results dictionary to avoid threading issues
            with self.data_lock:
                results_copy = dict(self.results_dict)
            
            if not results_copy:
                rospy.logdebug("No results to visualize")
                # Still publish empty marker array to clear old markers
                marker_array = MarkerArray()
                self.marker_pub.publish(marker_array)
                return
            
            marker_array = MarkerArray()
            
            # Get min and max confidence for normalization
            confidence_values = list(results_copy.values())
            if not confidence_values:
                return
            
            # FIXED: Use adaptive bounds or current data bounds
            if self.use_fixed_bounds and self.confidence_min is not None and self.confidence_max is not None:
                min_conf = self.confidence_min
                max_conf = self.confidence_max
                rospy.loginfo(f"Using fixed bounds: [{min_conf:.3f}, {max_conf:.3f}]")
            else:
                # Try adaptive bounds first
                adaptive_min, adaptive_max = self.get_adaptive_bounds()
                if adaptive_min is not None and adaptive_max is not None:
                    min_conf = adaptive_min
                    max_conf = adaptive_max
                    rospy.loginfo(f"Using adaptive bounds: [{min_conf:.3f}, {max_conf:.3f}]")
                else:
                    # Fall back to current data bounds
                    min_conf = min(confidence_values)
                    max_conf = max(confidence_values)
                    rospy.loginfo(f"Using data bounds: [{min_conf:.3f}, {max_conf:.3f}]")
            
            # Avoid division by zero
            if max_conf <= min_conf:
                max_conf = min_conf + 1.0
                rospy.logwarn(f"Adjusted bounds to avoid division by zero: [{min_conf:.3f}, {max_conf:.3f}]")
            
            # Sort items by confidence (highest first) and limit to max_markers
            sorted_items = sorted(results_copy.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_items) > self.max_markers:
                rospy.logwarn(f"Limiting visualization to top {self.max_markers} markers (out of {len(sorted_items)} grid cells)")
                sorted_items = sorted_items[:self.max_markers]
            
            current_marker_ids = set()
            color_distribution = {'red': 0, 'yellow': 0, 'green': 0, 'cyan': 0, 'blue': 0}
            
            for i, (idx, conf_per_surface) in enumerate(sorted_items):
                x_idx, y_idx, z_idx = idx
                
                # Create unique marker ID based on grid position
                marker_id = hash((x_idx, y_idx, z_idx)) % (2**31 - 1)
                if marker_id < 0:
                    marker_id = abs(marker_id)
                
                current_marker_ids.add(marker_id)
                
                # Create marker
                marker = Marker()
                marker.header.frame_id = self.frame_id
                marker.header.stamp = rospy.Time.now()
                marker.ns = "confidence_volumes"
                marker.id = marker_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                
                # Set position
                marker.pose.position.x = (x_idx + 0.5) * self.cube_size
                marker.pose.position.y = (y_idx + 0.5) * self.cube_size
                marker.pose.position.z = (z_idx + 0.5) * self.cube_size
                
                # Set orientation (identity)
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                
                # Set scale
                marker.scale.x = self.cube_size * 0.95
                marker.scale.y = self.cube_size * 0.95
                marker.scale.z = self.cube_size * 0.95
                
                # FIXED: Better normalization and color mapping
                norm_conf = (conf_per_surface - min_conf) / (max_conf - min_conf)
                norm_conf = max(0.0, min(1.0, norm_conf))  # Clamp to [0, 1]
                
                # Set color and track distribution
                marker.color = self.get_color_for_value(norm_conf)
                color_cat = self.categorize_color(norm_conf)
                color_distribution[color_cat] += 1
                
                # Set transparency
                marker.color.a = 0.7  # Semi-transparent
                
                # Make markers persistent
                marker.lifetime = rospy.Duration(0)
                
                marker_array.markers.append(marker)
            
            # Add DELETE markers for cleanup
            markers_to_delete = self.published_marker_ids - current_marker_ids
            for marker_id in markers_to_delete:
                delete_marker = Marker()
                delete_marker.header.frame_id = self.frame_id
                delete_marker.header.stamp = rospy.Time.now()
                delete_marker.ns = "confidence_volumes"
                delete_marker.id = marker_id
                delete_marker.action = Marker.DELETE
                marker_array.markers.append(delete_marker)
            
            # Update published marker IDs
            self.published_marker_ids = current_marker_ids
            
            # Publish marker array
            self.marker_pub.publish(marker_array)
            
            # Log color distribution for debugging
            rospy.loginfo(f"Published {len(sorted_items)} cubes. Color distribution: {color_distribution}")
            
        except Exception as e:
            rospy.logerr(f"Error in visualize_cubes: {e}")
    
    def categorize_color(self, norm_value):
        """Categorize normalized value into color ranges for debugging."""
        if norm_value < 0.2:
            return 'blue'
        elif norm_value < 0.4:
            return 'cyan'
        elif norm_value < 0.6:
            return 'green'
        elif norm_value < 0.8:
            return 'yellow'
        else:
            return 'red'
    
    def get_color_for_value(self, value):
        """Map a value in [0,1] to a color from blue to green to red."""
        color = ColorRGBA()
        
        # Improved color mapping: Blue -> Cyan -> Green -> Yellow -> Red
        if value < 0.25:
            # Blue to Cyan
            ratio = value / 0.25
            color.r = 0.0
            color.g = ratio
            color.b = 1.0
        elif value < 0.5:
            # Cyan to Green
            ratio = (value - 0.25) / 0.25
            color.r = 0.0
            color.g = 1.0
            color.b = 1.0 - ratio
        elif value < 0.75:
            # Green to Yellow
            ratio = (value - 0.5) / 0.25
            color.r = ratio
            color.g = 1.0
            color.b = 0.0
        else:
            # Yellow to Red
            ratio = (value - 0.75) / 0.25
            color.r = 1.0
            color.g = 1.0 - ratio
            color.b = 0.0
        
        color.a = 0.5 # Semi-transparent
        return color

if __name__ == '__main__':
    try:
        node = ConfidenceVolumeVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
        raise