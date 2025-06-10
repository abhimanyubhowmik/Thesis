#!/usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray

class FogRegionVisualizer:
    def __init__(self):
        rospy.init_node('fog_region_visualizer', anonymous=True)
        
        # Get parameters
        self.world_size = rospy.get_param('~world_size', 2000.0)
        self.world_center = rospy.get_param('~world_center', [0.0, 0.0])
        self.division_type = rospy.get_param('~division_type', 'vertical')
        self.fog_density_low = rospy.get_param('~fog_density_low', 2.0)
        self.fog_density_high = rospy.get_param('~fog_density_high', 5.0)
        
        # Publishers
        self.marker_pub = rospy.Publisher('/fog_regions_visualization', MarkerArray, queue_size=1)
        
        # Subscribers
        self.robot_pos_sub = rospy.Subscriber('/robot_position', Point, self.robot_position_callback)
        self.fog_density_sub = rospy.Subscriber('/current_fog_density', Float32, self.fog_density_callback)
        
        # Current robot position
        self.robot_position = None
        self.current_fog_density = self.fog_density_low
        
        # Timer for publishing markers
        self.timer = rospy.Timer(rospy.Duration(1.0), self.publish_markers)
        
        rospy.loginfo("Fog Region Visualizer initialized")
    
    def robot_position_callback(self, msg):
        """Update robot position"""
        self.robot_position = [msg.x, msg.y, msg.z]
    
    def fog_density_callback(self, msg):
        """Update current fog density"""
        self.current_fog_density = msg.data
    
    def get_region_color(self, is_high_fog):
        """Get color for region based on fog density"""
        if is_high_fog:
            return [0.7, 0.7, 0.7, 0.5]  # Gray for high fog
        else:
            return [0.5, 0.8, 1.0, 0.3]  # Light blue for low fog
    
    def create_region_markers(self):
        """Create markers to visualize fog regions"""
        markers = MarkerArray()
        
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        markers.markers.append(clear_marker)
        
        if self.division_type == 'vertical':
            # Left region (low fog)
            left_marker = self.create_box_marker(
                0, 
                [-self.world_size/4, 0, 0],
                [self.world_size/2, self.world_size, 50],
                self.get_region_color(False),
                f"Low Fog Region ({self.fog_density_low})"
            )
            markers.markers.append(left_marker)
            
            # Right region (high fog)
            right_marker = self.create_box_marker(
                1,
                [self.world_size/4, 0, 0],
                [self.world_size/2, self.world_size, 50],
                self.get_region_color(True),
                f"High Fog Region ({self.fog_density_high})"
            )
            markers.markers.append(right_marker)
            
        elif self.division_type == 'horizontal':
            # Bottom region (low fog)
            bottom_marker = self.create_box_marker(
                0,
                [0, -self.world_size/4, 0],
                [self.world_size, self.world_size/2, 50],
                self.get_region_color(False),
                f"Low Fog Region ({self.fog_density_low})"
            )
            markers.markers.append(bottom_marker)
            
            # Top region (high fog)
            top_marker = self.create_box_marker(
                1,
                [0, self.world_size/4, 0],
                [self.world_size, self.world_size/2, 50],
                self.get_region_color(True),
                f"High Fog Region ({self.fog_density_high})"
            )
            markers.markers.append(top_marker)
            
        elif self.division_type == 'circular':
            # Inner circle (low fog)
            inner_marker = self.create_cylinder_marker(
                0,
                [0, 0, 0],
                self.world_size/4,
                50,
                self.get_region_color(False),
                f"Low Fog Region ({self.fog_density_low})"
            )
            markers.markers.append(inner_marker)
            
            # Outer ring visualization (approximate with boxes)
            for i in range(4):
                angle = i * np.pi / 2
                x = np.cos(angle) * self.world_size * 0.375
                y = np.sin(angle) * self.world_size * 0.375
                
                outer_marker = self.create_box_marker(
                    i + 2,
                    [x, y, 0],
                    [self.world_size/4, self.world_size/4, 50],
                    self.get_region_color(True),
                    f"High Fog Region ({self.fog_density_high})" if i == 0 else ""
                )
                markers.markers.append(outer_marker)
        
        # Add robot position marker if available
        if self.robot_position is not None:
            robot_marker = self.create_robot_marker()
            markers.markers.append(robot_marker)
        
        # Add boundary line marker
        boundary_marker = self.create_boundary_marker()
        if boundary_marker:
            markers.markers.append(boundary_marker)
        
        return markers
    
    def create_box_marker(self, marker_id, position, scale, color, text=""):
        """Create a box marker"""
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        marker.pose.position.x = position[0] + self.world_center[0]
        marker.pose.position.y = position[1] + self.world_center[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]
        
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        
        return marker
    
    def create_cylinder_marker(self, marker_id, position, radius, height, color, text=""):
        """Create a cylinder marker"""
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        marker.pose.position.x = position[0] + self.world_center[0]
        marker.pose.position.y = position[1] + self.world_center[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = radius * 2
        marker.scale.y = radius * 2
        marker.scale.z = height
        
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        
        return marker
    
    def create_robot_marker(self):
        """Create robot position marker"""
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.id = 100
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = self.robot_position[0]
        marker.pose.position.y = self.robot_position[1]
        marker.pose.position.z = self.robot_position[2]
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 20.0
        marker.scale.y = 20.0
        marker.scale.z = 20.0
        
        # Color based on current fog density
        if abs(self.current_fog_density - self.fog_density_high) < 0.1:
            marker.color.r = 1.0  # Red for high fog
            marker.color.g = 0.0
            marker.color.b = 0.0
        else:
            marker.color.r = 0.0  # Green for low fog
            marker.color.g = 1.0
            marker.color.b = 0.0
        marker.color.a = 1.0
        
        return marker
    
    def create_boundary_marker(self):
        """Create boundary line marker"""
        if self.division_type not in ['vertical', 'horizontal']:
            return None
            
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.id = 101
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        marker.pose.orientation.w = 1.0
        marker.scale.x = 5.0  # Line width
        
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # Create boundary line points
        if self.division_type == 'vertical':
            # Vertical line at x = 0
            p1 = Point()
            p1.x = self.world_center[0]
            p1.y = self.world_center[1] - self.world_size/2
            p1.z = 0
            
            p2 = Point()
            p2.x = self.world_center[0]
            p2.y = self.world_center[1] + self.world_size/2
            p2.z = 0
            
        else:  # horizontal
            # Horizontal line at y = 0
            p1 = Point()
            p1.x = self.world_center[0] - self.world_size/2
            p1.y = self.world_center[1]
            p1.z = 0
            
            p2 = Point()
            p2.x = self.world_center[0] + self.world_size/2
            p2.y = self.world_center[1]
            p2.z = 0
        
        marker.points = [p1, p2]
        return marker
    
    def publish_markers(self, event):
        """Publish visualization markers"""
        markers = self.create_region_markers()
        self.marker_pub.publish(markers)
    
    def plot_regions_matplotlib(self):
        """Create a matplotlib plot of the fog regions (for debugging)"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot world boundary
        world_x = [-self.world_size/2, self.world_size/2, self.world_size/2, -self.world_size/2, -self.world_size/2]
        world_y = [-self.world_size/2, -self.world_size/2, self.world_size/2, self.world_size/2, -self.world_size/2]
        ax.plot(world_x, world_y, 'k-', linewidth=2, label='World Boundary')
        
        if self.division_type == 'vertical':
            ax.axvline(x=0, color='yellow', linewidth=3, label='Fog Boundary')
            ax.fill_between([-self.world_size/2, 0], -self.world_size/2, self.world_size/2, 
                          alpha=0.3, color='lightblue', label=f'Low Fog ({self.fog_density_low})')
            ax.fill_between([0, self.world_size/2], -self.world_size/2, self.world_size/2, 
                          alpha=0.3, color='gray', label=f'High Fog ({self.fog_density_high})')
            
        elif self.division_type == 'horizontal':
            ax.axhline(y=0, color='yellow', linewidth=3, label='Fog Boundary')
            ax.fill_between([-self.world_size/2, self.world_size/2], -self.world_size/2, 0, 
                          alpha=0.3, color='lightblue', label=f'Low Fog ({self.fog_density_low})')
            ax.fill_between([-self.world_size/2, self.world_size/2], 0, self.world_size/2, 
                          alpha=0.3, color='gray', label=f'High Fog ({self.fog_density_high})')
            
        elif self.division_type == 'circular':
            circle = plt.Circle((0, 0), self.world_size/4, alpha=0.3, color='lightblue', 
                              label=f'Low Fog ({self.fog_density_low})')
            ax.add_patch(circle)
            # Add text for outer region
            ax.text(self.world_size/3, self.world_size/3, f'High Fog\n({self.fog_density_high})', 
                   fontsize=12, ha='center', va='center', 
                   bbox=dict(boxstyle='round', facecolor='gray', alpha=0.3))
        
        # Plot robot position if available
        if self.robot_position is not None:
            color = 'red' if abs(self.current_fog_density - self.fog_density_high) < 0.1 else 'green'
            ax.plot(self.robot_position[0], self.robot_position[1], 'o', 
                   color=color, markersize=10, label='Robot Position')
        
        ax.set_xlim(-self.world_size/2 - 100, self.world_size/2 + 100)
        ax.set_ylim(-self.world_size/2 - 100, self.world_size/2 + 100)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title(f'Fog Density Regions - {self.division_type.title()} Division')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.show()


if __name__ == '__main__':
    try:
        visualizer = FogRegionVisualizer()
        
        # Optional: Create matplotlib plot for debugging
        if rospy.get_param('~show_plot', False):
            visualizer.plot_regions_matplotlib()
        
        rospy.spin()
    except rospy.ROSInterruptException:
        pass