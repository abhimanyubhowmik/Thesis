#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Point
from std_msgs.msg import Float32

class FogDensityController:
    def __init__(self, env):
        """
        Initialize fog density controller
        
        Args:
            env: HoloOcean environment instance
        """
        self.env = env
        
        # Define world boundaries (2km x 2km area)
        self.world_size = rospy.get_param('~world_size', 2000.0)  # 2km in meters
        self.world_center = rospy.get_param('~world_center', [0.0, 0.0])  # World center coordinates
        
        # Define fog density regions
        self.fog_density_low = rospy.get_param('~fog_density_low', 2.0)
        self.fog_density_high = rospy.get_param('~fog_density_high', 5.0)
        
        # Division type: 'vertical', 'horizontal', 'diagonal', or 'circular'
        self.division_type = rospy.get_param('~division_type', 'vertical')
        
        # Current fog density
        self.current_fog_density = self.fog_density_low
        
        # Hysteresis to prevent rapid switching
        self.hysteresis_margin = rospy.get_param('~hysteresis_margin', 0.0)  # 10 meters
        self.last_region = None
        
        # Publishers for debugging
        self.fog_density_pub = rospy.Publisher('/current_fog_density', Float32, queue_size=1)
        self.robot_position_pub = rospy.Publisher('/robot_position', Point, queue_size=1)
        
        rospy.loginfo(f"Fog Density Controller initialized with {self.division_type} division")
        rospy.loginfo(f"World size: {self.world_size}m x {self.world_size}m")
        rospy.loginfo(f"Low fog density: {self.fog_density_low}, High fog density: {self.fog_density_high}")
    
    def get_region(self, position):
        """
        Determine which region the robot is in based on position
        
        Args:
            position: [x, y, z] coordinates of the robot
            
        Returns:
            str: 'low_fog' or 'high_fog'
        """
        x, y = position[0], position[1]
        
        # Adjust coordinates relative to world center
        rel_x = x - self.world_center[0]
        rel_y = y - self.world_center[1]
        
        if self.division_type == 'vertical':
            # Divide vertically (left/right)
            if rel_x < 0:
                return 'low_fog'
            else:
                return 'high_fog'
                
        elif self.division_type == 'horizontal':
            # Divide horizontally (top/bottom)
            if rel_y < 0:
                return 'low_fog'
            else:
                return 'high_fog'
                
        elif self.division_type == 'diagonal':
            # Divide diagonally (along y = x line)
            if rel_y < rel_x:
                return 'low_fog'
            else:
                return 'high_fog'
                
        elif self.division_type == 'circular':
            # Circular division (inner circle vs outer ring)
            distance_from_center = np.sqrt(rel_x**2 + rel_y**2)
            radius = self.world_size / 4  # Quarter of world size as radius
            if distance_from_center < radius:
                return 'low_fog'
            else:
                return 'high_fog'
        
        # Default to low fog
        return 'low_fog'
    
    def update_fog_density(self, position):
        """
        Update fog density based on robot position
        
        Args:
            position: [x, y, z] coordinates of the robot
        """
        current_region = self.get_region(position)
        
        # Apply hysteresis to prevent rapid switching
        if self.last_region is not None and self.last_region != current_region:
            # Check if we're close to the boundary
            if self._near_boundary(position):
                # Don't change region if we're within hysteresis margin
                current_region = self.last_region
        
        # Determine target fog density
        if current_region == 'low_fog':
            target_density = self.fog_density_low
        else:
            target_density = self.fog_density_high
        
        # Update fog density if it has changed
        if abs(target_density - self.current_fog_density) > 0.1:
            self.env.weather.set_fog_density(target_density)
            self.current_fog_density = target_density
            self.last_region = current_region
            
            rospy.loginfo(f"Fog density changed to {target_density} (region: {current_region})")
        
        # Publish current state for debugging
        self._publish_debug_info(position, target_density)
    
    def _near_boundary(self, position):
        """
        Check if robot is near region boundary (for hysteresis)
        
        Args:
            position: [x, y, z] coordinates of the robot
            
        Returns:
            bool: True if near boundary
        """
        x, y = position[0], position[1]
        rel_x = x - self.world_center[0]
        rel_y = y - self.world_center[1]
        
        if self.division_type == 'vertical':
            return abs(rel_x) < self.hysteresis_margin
        elif self.division_type == 'horizontal':
            return abs(rel_y) < self.hysteresis_margin
        elif self.division_type == 'diagonal':
            # Distance from diagonal line y = x
            distance_to_line = abs(rel_y - rel_x) / np.sqrt(2)
            return distance_to_line < self.hysteresis_margin
        elif self.division_type == 'circular':
            distance_from_center = np.sqrt(rel_x**2 + rel_y**2)
            radius = self.world_size / 4
            return abs(distance_from_center - radius) < self.hysteresis_margin
        
        return False
    
    def _publish_debug_info(self, position, fog_density):
        """
        Publish debug information
        
        Args:
            position: Robot position
            fog_density: Current fog density
        """
        # Publish robot position
        pos_msg = Point()
        pos_msg.x = position[0]
        pos_msg.y = position[1]
        pos_msg.z = position[2]
        self.robot_position_pub.publish(pos_msg)
        
        # Publish fog density
        fog_msg = Float32()
        fog_msg.data = fog_density
        self.fog_density_pub.publish(fog_msg)
    
    def get_region_info(self):
        """
        Get information about current region division for visualization
        
        Returns:
            dict: Region information
        """
        return {
            'division_type': self.division_type,
            'world_size': self.world_size,
            'world_center': self.world_center,
            'low_fog_density': self.fog_density_low,
            'high_fog_density': self.fog_density_high,
            'current_fog_density': self.current_fog_density
        }