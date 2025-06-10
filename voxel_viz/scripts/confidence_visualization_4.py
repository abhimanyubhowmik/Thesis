#!/usr/bin/env python3

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
        
        # NEW: Parameters for persistence and data management
        self.data_persistence_time = rospy.get_param('~data_persistence_time', 30.0)  # Keep data for 30 seconds
        self.accumulate_data = rospy.get_param('~accumulate_data', True)  # Accumulate data over time
        self.update_mode = rospy.get_param('~update_mode', 'incremental')  # 'incremental' or 'replace'
        
        # NEW: Parameters for better surface area estimation
        self.use_point_density_area = rospy.get_param('~use_point_density_area', True)
        self.adaptive_color_scaling = rospy.get_param('~adaptive_color_scaling', True)
        
        rospy.loginfo(f"Subscribing to topic: {self.confidence_topic}")
        rospy.loginfo(f"Using fixed bounds: {self.use_fixed_bounds}")
        rospy.loginfo(f"Adaptive color scaling: {self.adaptive_color_scaling}")
        rospy.loginfo(f"Data persistence time: {self.data_persistence_time}s")
        rospy.loginfo(f"Update mode: {self.update_mode}")
        
        # Subscribers
        self.confidence_sub = rospy.Subscriber(
            self.confidence_topic, PointCloud2, self.confidence_callback, queue_size=1)
        
        # Add timer to check subscription status
        rospy.Timer(rospy.Duration(5.0), self.check_subscription_status)
        
        # Publishers
        self.marker_pub = rospy.Publisher(
            '/confidence_visualization/cubes', MarkerArray, queue_size=1)
        
        # FIXED: Enhanced data containers with timestamps for persistence
        self.data_lock = threading.Lock()
        self.points_dict = {}  # Key: (x_idx, y_idx, z_idx), Value: {'points': List of (point, confidence), 'timestamp': rospy.Time, 'last_seen': rospy.Time}
        self.results_dict = {}  # Key: (x_idx, y_idx, z_idx), Value: {'conf_per_surface': float, 'timestamp': rospy.Time}
        
        # FIXED: Better marker ID management - use sequential IDs instead of hash
        self.grid_to_marker_id = {}  # Map grid indices to marker IDs
        self.next_marker_id = 0
        self.active_marker_ids = set()  # Track currently active marker IDs
        
        # NEW: Statistics tracking for better debugging
        self.stats_history = []
        self.max_history = 10
        
        # Timer for visualization
        rospy.Timer(rospy.Duration(self.visualize_interval), self.visualize_cubes)
        
        # NEW: Timer for data cleanup
        rospy.Timer(rospy.Duration(5.0), self.cleanup_old_data)
        
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
                rospy.logdebug(f"Topic {self.confidence_topic} is available with type {topic_type}")
                
                # Check if we have any subscribers
                pub_count = self.marker_pub.get_num_connections()
                rospy.logdebug(f"Number of subscribers to our visualization: {pub_count}")
        except Exception as e:
            rospy.logwarn(f"Error checking subscription status: {e}")
    
    def point_to_grid_idx(self, point):
        """Convert a 3D point to grid indices based on cube_size."""
        x_idx = int(np.floor(point[0] / self.cube_size))  # Use floor instead of //
        y_idx = int(np.floor(point[1] / self.cube_size))
        z_idx = int(np.floor(point[2] / self.cube_size))
        return (x_idx, y_idx, z_idx)
    
    def get_marker_id_for_grid(self, grid_idx):
        """Get a unique marker ID for a grid cell, creating one if needed."""
        if grid_idx not in self.grid_to_marker_id:
            self.grid_to_marker_id[grid_idx] = self.next_marker_id
            self.next_marker_id += 1
        return self.grid_to_marker_id[grid_idx]
    
    def cleanup_old_data(self, event=None):
        """Remove old data that hasn't been updated recently."""
        if not self.accumulate_data:
            return
            
        current_time = rospy.Time.now()
        cutoff_time = current_time - rospy.Duration(self.data_persistence_time)
        
        with self.data_lock:
            # Clean up old points
            old_point_keys = []
            for idx, data in self.points_dict.items():
                if data['last_seen'] < cutoff_time:
                    old_point_keys.append(idx)
            
            for key in old_point_keys:
                del self.points_dict[key]
                rospy.logdebug(f"Removed old point data for grid cell {key}")
            
            # Clean up old results
            old_result_keys = []
            for idx, data in self.results_dict.items():
                if data['timestamp'] < cutoff_time:
                    old_result_keys.append(idx)
            
            for key in old_result_keys:
                del self.results_dict[key]
                rospy.logdebug(f"Removed old result data for grid cell {key}")
            
            if old_point_keys or old_result_keys:
                rospy.loginfo(f"Cleaned up {len(old_point_keys)} old point cells and {len(old_result_keys)} old result cells")
    
    def confidence_callback(self, cloud_msg):
        """Process incoming confidence pointcloud."""
        try:
            rospy.loginfo("Received pointcloud with %d points", cloud_msg.width * cloud_msg.height)
            
            current_time = rospy.Time.now()
            
            # Process points
            new_data = {}  # Temporary storage for new points
            confidence_values = []  # Track all confidence values for statistics
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
                
                # Add point to temporary storage
                if idx not in new_data:
                    new_data[idx] = []
                
                new_data[idx].append((np.array([x, y, z]), confidence))
            
            # NEW: Log confidence statistics
            if confidence_values:
                conf_array = np.array(confidence_values)
                rospy.loginfo(f"Processed {point_count} total points, {valid_point_count} valid points")
                rospy.loginfo(f"Confidence stats - Min: {np.min(conf_array):.3f}, Max: {np.max(conf_array):.3f}, Mean: {np.mean(conf_array):.3f}, Std: {np.std(conf_array):.3f}")
                rospy.loginfo(f"Found {len(new_data)} non-empty grid cells in new data")
            
            # FIXED: Update data with proper persistence and thread safety
            with self.data_lock:
                if self.update_mode == 'replace' or not self.accumulate_data:
                    # Replace mode: completely replace old data
                    self.points_dict = {}
                    for idx, points_data in new_data.items():
                        self.points_dict[idx] = {
                            'points': points_data,
                            'timestamp': current_time,
                            'last_seen': current_time
                        }
                else:
                    # Incremental mode: merge with existing data
                    for idx, new_points_data in new_data.items():
                        if idx in self.points_dict:
                            # Update existing cell - merge points or replace based on preference
                            existing_data = self.points_dict[idx]
                            if len(new_points_data) >= len(existing_data['points']):
                                # If new data has more points, replace
                                self.points_dict[idx] = {
                                    'points': new_points_data,
                                    'timestamp': current_time,
                                    'last_seen': current_time
                                }
                                rospy.logdebug(f"Updated grid cell {idx} with {len(new_points_data)} points")
                            else:
                                # Just update the last_seen timestamp
                                self.points_dict[idx]['last_seen'] = current_time
                                rospy.logdebug(f"Refreshed timestamp for grid cell {idx}")
                        else:
                            # New cell
                            self.points_dict[idx] = {
                                'points': new_points_data,
                                'timestamp': current_time,
                                'last_seen': current_time
                            }
                            rospy.logdebug(f"Added new grid cell {idx} with {len(new_points_data)} points")
                
                # Process grid cells after updating points
                self.process_grid_cells()
                
                rospy.loginfo(f"Total active grid cells: {len(self.points_dict)}")
                
        except Exception as e:
            rospy.logerr(f"Error in confidence_callback: {e}")
    
    def process_grid_cells(self):
        """Calculate confidence per surface area for each grid cell."""
        try:
            current_time = rospy.Time.now()
            cell_stats = []
            updated_cells = 0
            
            for idx, cell_data in self.points_dict.items():
                points_data = cell_data['points']
                
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
                
                # FIXED: Store result with timestamp
                self.results_dict[idx] = {
                    'conf_per_surface': conf_per_surface,
                    'timestamp': current_time
                }
                updated_cells += 1
                
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
                
                rospy.loginfo(f"Cell statistics - Processed {len(cell_stats)} cells (updated {updated_cells}):")
                rospy.loginfo(f"  Conf/Surface - Min: {np.min(cps_array):.3f}, Max: {np.max(cps_array):.3f}, Mean: {np.mean(cps_array):.3f}")
                rospy.loginfo(f"  Surface areas - Min: {min(s['surface_area'] for s in cell_stats):.3f}, Max: {max(s['surface_area'] for s in cell_stats):.3f}")
                
                # Store statistics for adaptive scaling
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
            # FIXED: Create a safer copy of the results dictionary
            with self.data_lock:
                results_copy = {}
                for idx, data in self.results_dict.items():
                    results_copy[idx] = data['conf_per_surface']
            
            if not results_copy:
                rospy.logdebug("No results to visualize")
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
                rospy.logdebug(f"Using fixed bounds: [{min_conf:.3f}, {max_conf:.3f}]")
            else:
                # Try adaptive bounds first
                adaptive_min, adaptive_max = self.get_adaptive_bounds()
                if adaptive_min is not None and adaptive_max is not None:
                    min_conf = adaptive_min
                    max_conf = adaptive_max
                    rospy.logdebug(f"Using adaptive bounds: [{min_conf:.3f}, {max_conf:.3f}]")
                else:
                    # Fall back to current data bounds
                    min_conf = min(confidence_values)
                    max_conf = max(confidence_values)
                    rospy.logdebug(f"Using data bounds: [{min_conf:.3f}, {max_conf:.3f}]")
            
            # Avoid division by zero
            if max_conf <= min_conf:
                max_conf = min_conf + 1.0
                rospy.logwarn(f"Adjusted bounds to avoid division by zero: [{min_conf:.3f}, {max_conf:.3f}]")
            
            # Sort items by confidence (highest first) and limit to max_markers
            sorted_items = sorted(results_copy.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_items) > self.max_markers:
                rospy.logwarn(f"Limiting visualization to top {self.max_markers} markers (out of {len(sorted_items)} grid cells)")
                sorted_items = sorted_items[:self.max_markers]
            
            current_active_ids = set()
            color_distribution = {'red': 0, 'yellow': 0, 'green': 0, 'cyan': 0, 'blue': 0}
            
            # FIXED: Use consistent marker IDs and better cleanup
            for i, (idx, conf_per_surface) in enumerate(sorted_items):
                x_idx, y_idx, z_idx = idx
                
                # Get consistent marker ID for this grid cell
                marker_id = self.get_marker_id_for_grid(idx)
                current_active_ids.add(marker_id)
                
                # Create marker
                marker = Marker()
                marker.header.frame_id = self.frame_id
                marker.header.stamp = rospy.Time.now()
                marker.ns = "confidence_volumes"
                marker.id = marker_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                
                # Set position (center of the grid cell)
                marker.pose.position.x = (x_idx + 0.5) * self.cube_size
                marker.pose.position.y = (y_idx + 0.5) * self.cube_size
                marker.pose.position.z = (z_idx + 0.5) * self.cube_size
                
                # Set orientation (identity)
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                
                # Set scale (slightly smaller than grid cell to avoid overlap)
                marker.scale.x = self.cube_size * 0.9
                marker.scale.y = self.cube_size * 0.9
                marker.scale.z = self.cube_size * 0.9
                
                # FIXED: Better normalization and color mapping
                norm_conf = (conf_per_surface - min_conf) / (max_conf - min_conf)
                norm_conf = max(0.0, min(1.0, norm_conf))  # Clamp to [0, 1]
                
                # Set color and track distribution
                marker.color = self.get_color_for_value(norm_conf)
                color_cat = self.categorize_color(norm_conf)
                color_distribution[color_cat] += 1
                
                # Set transparency
                marker.color.a = 0.3  # Semi-transparent
                
                # Make markers persistent but with a reasonable lifetime
                marker.lifetime = rospy.Duration(self.data_persistence_time * 2)
                
                marker_array.markers.append(marker)
            
            # FIXED: Clean up only markers that are no longer active
            markers_to_delete = self.active_marker_ids - current_active_ids
            for marker_id in markers_to_delete:
                delete_marker = Marker()
                delete_marker.header.frame_id = self.frame_id
                delete_marker.header.stamp = rospy.Time.now()
                delete_marker.ns = "confidence_volumes"
                delete_marker.id = marker_id
                delete_marker.action = Marker.DELETE
                marker_array.markers.append(delete_marker)
            
            # Update active marker IDs
            self.active_marker_ids = current_active_ids
            
            # FIXED: Add a small delay to ensure frame is ready
            rospy.sleep(0.01)
            
            # Publish marker array
            self.marker_pub.publish(marker_array)
            
            # Enhanced logging with marker IDs for debugging
            rospy.loginfo(f"Published {len(sorted_items)} cubes (total active: {len(results_copy)}). Color distribution: {color_distribution}")
            rospy.loginfo(f"Active marker IDs: {sorted(list(current_active_ids))}")
            if markers_to_delete:
                rospy.loginfo(f"Deleted {len(markers_to_delete)} old markers: {sorted(list(markers_to_delete))}")
            
            # ADDITIONAL DEBUG: Print position and confidence for each marker
            for i, (idx, conf_per_surface) in enumerate(sorted_items[:5]):  # First 5 only
                x_idx, y_idx, z_idx = idx
                pos_x = (x_idx + 0.5) * self.cube_size
                pos_y = (y_idx + 0.5) * self.cube_size
                pos_z = (z_idx + 0.5) * self.cube_size
                marker_id = self.get_marker_id_for_grid(idx)
                rospy.loginfo(f"  Marker {marker_id}: Grid{idx} -> Pos({pos_x:.1f}, {pos_y:.1f}, {pos_z:.1f}) Conf={conf_per_surface:.1f}")
            
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
        
        color.a = 0.3  # Semi-transparent
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