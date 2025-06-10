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
        self.confidence_min = rospy.get_param('~confidence_min', 0.0)  # For color scaling
        self.confidence_max = rospy.get_param('~confidence_max', 10.0)  # For color scaling
        self.max_markers = rospy.get_param('~max_markers', 1000)  # Maximum number of markers to display
        
        rospy.loginfo(f"Subscribing to topic: {self.confidence_topic}")
        
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
        x_idx = int(point[0] // self.cube_size)
        y_idx = int(point[1] // self.cube_size)
        z_idx = int(point[2] // self.cube_size)
        return (x_idx, y_idx, z_idx)
    
    def confidence_callback(self, cloud_msg):
        """Process incoming confidence pointcloud."""
        try:
            rospy.loginfo("Received pointcloud with %d points", cloud_msg.width * cloud_msg.height)
            
            # Create new data containers
            new_points_dict = {}
            
            # Process points
            point_count = 0
            for point in pc2.read_points(cloud_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
                x, y, z, confidence = point
                point_count += 1
                
                # Skip points with zero or invalid confidence
                if confidence <= 0 or not np.isfinite(confidence):
                    continue
                
                # Skip points with invalid coordinates
                if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                    continue
                
                # Get grid indices
                idx = self.point_to_grid_idx((x, y, z))
                
                # Add point to grid
                if idx not in new_points_dict:
                    new_points_dict[idx] = []
                
                new_points_dict[idx].append((np.array([x, y, z]), confidence))
            
            rospy.loginfo(f"Processed {point_count} points, found {len(new_points_dict)} non-empty grid cells")
            
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
            
            for idx, points_data in self.points_dict.items():
                if len(points_data) < 3:  # Need at least 3 points to estimate a surface
                    continue
                
                # Extract points and confidences
                points = [p[0] for p in points_data]
                confidences = [p[1] for p in points_data]
                
                # Convert to numpy arrays
                points_array = np.array(points)
                confidences_array = np.array(confidences)
                
                # Estimate surface area using point density and distribution
                # This is a simple approximation - can be improved with more sophisticated methods
                
                # Get the principal components to estimate the surface normal
                mean_point = np.mean(points_array, axis=0)
                centered_points = points_array - mean_point
                
                # Use SVD to find the principal components
                try:
                    U, S, Vt = np.linalg.svd(centered_points)
                    # The smallest singular value corresponds to the normal direction
                    # If it's much smaller than the others, we can approximate a 2D surface
                    if len(S) >= 3 and S[2] < 0.1 * S[1]:  # Check if points roughly form a 2D surface
                        # Approximate surface area based on the spread in the two principal directions
                        surface_area = np.pi * S[0] * S[1]  # Elliptical approximation
                    else:
                        # If points don't form a clear surface, use the bounding box as approximation
                        ranges = np.max(points_array, axis=0) - np.min(points_array, axis=0)
                        surface_area = 2 * (ranges[0]*ranges[1] + ranges[0]*ranges[2] + ranges[1]*ranges[2])
                except Exception as svd_error:
                    rospy.logwarn(f"SVD failed for cell {idx}: {svd_error}")
                    # Fallback: use bounding box approximation
                    ranges = np.max(points_array, axis=0) - np.min(points_array, axis=0)
                    surface_area = 2 * (ranges[0]*ranges[1] + ranges[0]*ranges[2] + ranges[1]*ranges[2])
                
                # Avoid division by zero
                if surface_area < 1e-6:
                    surface_area = 1e-6
                
                # Calculate total confidence
                total_confidence = np.sum(confidences_array)
                
                # Calculate confidence per surface area
                conf_per_surface = total_confidence / surface_area
                
                # Store result
                new_results_dict[idx] = conf_per_surface
            
            # Update results dictionary (this assignment is atomic in Python)
            self.results_dict = new_results_dict
            rospy.loginfo(f"Processed {len(self.results_dict)} grid cells")
            
        except Exception as e:
            rospy.logerr(f"Error in process_grid_cells: {e}")
    
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
                
            min_conf = min(confidence_values)
            max_conf = max(confidence_values)
            
            # Use parameter values if specified
            if self.confidence_min is not None:
                min_conf = self.confidence_min
            if self.confidence_max is not None:
                max_conf = self.confidence_max
            
            # Avoid division by zero
            if max_conf <= min_conf:
                max_conf = min_conf + 1.0
            
            # Sort items by confidence (highest first) and limit to max_markers
            sorted_items = sorted(results_copy.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_items) > self.max_markers:
                rospy.logwarn(f"Limiting visualization to top {self.max_markers} markers (out of {len(sorted_items)} grid cells)")
                sorted_items = sorted_items[:self.max_markers]
            
            current_marker_ids = set()
            
            for i, (idx, conf_per_surface) in enumerate(sorted_items):
                x_idx, y_idx, z_idx = idx
                
                # Create unique marker ID based on grid position
                # This ensures the same grid cell always gets the same marker ID
                marker_id = hash((x_idx, y_idx, z_idx)) % (2**31 - 1)  # Ensure positive int32
                if marker_id < 0:  # Handle edge case where hash might be negative
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
                
                # Set scale (slightly smaller than cube size for visualization)
                marker.scale.x = self.cube_size * 0.95
                marker.scale.y = self.cube_size * 0.95
                marker.scale.z = self.cube_size * 0.95
                
                # Set color based on confidence per surface
                norm_conf = (conf_per_surface - min_conf) / (max_conf - min_conf)
                norm_conf = max(0.0, min(1.0, norm_conf))  # Clamp to [0, 1]
                
                # Color mapping from blue (low confidence) to red (high confidence)
                marker.color = self.get_color_for_value(norm_conf)
                
                # Set transparency
                marker.color.a = 0.6  # Semi-transparent
                
                # Make markers persistent (no automatic deletion)
                marker.lifetime = rospy.Duration(0)  # 0 means never expire
                
                marker_array.markers.append(marker)
            
            # Add DELETE markers for any markers that are no longer needed
            markers_to_delete = self.published_marker_ids - current_marker_ids
            for marker_id in markers_to_delete:
                delete_marker = Marker()
                delete_marker.header.frame_id = self.frame_id
                delete_marker.header.stamp = rospy.Time.now()
                delete_marker.ns = "confidence_volumes"
                delete_marker.id = marker_id
                delete_marker.action = Marker.DELETE
                marker_array.markers.append(delete_marker)
            
            # Update the set of published marker IDs
            self.published_marker_ids = current_marker_ids
            
            # Publish marker array
            self.marker_pub.publish(marker_array)
            rospy.loginfo(f"Published {len(sorted_items)} visualization cubes (from {len(results_copy)} grid cells), deleted {len(markers_to_delete)} old markers")
            
        except Exception as e:
            rospy.logerr(f"Error in visualize_cubes: {e}")
    
    def get_color_for_value(self, value):
        """Map a value in [0,1] to a color from blue to green to red."""
        color = ColorRGBA()
        
        # Blue (0.0) to Green (0.5) to Red (1.0)
        if value < 0.5:
            # Blue to Green
            ratio = value / 0.5
            color.r = 0.0
            color.g = ratio
            color.b = 1.0 - ratio
        else:
            # Green to Red
            ratio = (value - 0.5) / 0.5
            color.r = ratio
            color.g = 1.0 - ratio
            color.b = 0.0
        
        color.a = 0.6  # Semi-transparent
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