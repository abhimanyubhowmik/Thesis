#!/usr/bin/env python3
import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Vector3
from holoocean_ros.controllers import PIDController

class ControlSystem:
    def __init__(self, pid_config):
        """Initialize the control system with PID controllers."""
        self._init_controllers(pid_config)
        self.current_command = np.zeros(8)
        self.current_target_pose = None
        self.path_index = 0
        
        # Initialize the publishers for PID errors
        self.pos_error_pub = rospy.Publisher('/control/position_error', Vector3, queue_size=10)
        self.vel_error_pub = rospy.Publisher('/control/velocity_error', Vector3, queue_size=10)
        self.att_error_pub = rospy.Publisher('/control/attitude_error', Vector3, queue_size=10)

    def _init_controllers(self, pid_config):
        """Initialize PID controllers using config values"""
        # Velocity controller
        vel_Kp = np.diag(pid_config['vel_Kp'])
        vel_Kd = np.diag(pid_config['vel_Kd'])
        vel_drag_coeff_p1 = np.diag([1.5, 1.5, 1.5, 0.1, 0.1, 0.1])
        vel_drag_coeff_p2 = np.diag([7.0, 7.0, 7.0, 1.2, 1.2, 1.2])    

        # Position controller
        position_Kp = np.diag(pid_config['position_Kp'])
        position_Ki = np.diag(pid_config['position_Ki'])
        position_Kd = np.diag(pid_config['position_Kd'])

        # Attitude controller
        att_Kp = np.diag(pid_config['att_Kp'])

        self.vel_controller = PIDController(
            Kp=vel_Kp, Kd=vel_Kd,
            drag_coeff_p1=vel_drag_coeff_p1,
            drag_coeff_p2=vel_drag_coeff_p2
        )
        self.position_controller = PIDController(
            Kp=position_Kp, Ki=position_Ki, Kd=position_Kd, state_dim=3
        )
        self.attitude_controller = PIDController(Kp=att_Kp, state_dim=3)

    def update(self, state, sim_time):
        """Update controllers and compute control command"""
        if not self.current_target_pose:
            return
            
        dt = state["t"] - self.vel_controller.prev_time
        
        # Get current pose and dynamics
        pose_world = state["PoseSensor"]
        angular_rates = state["IMUSensor"][1, :]
        rotmat = pose_world[0:3, 0:3]
        pose_rot = pose_world[0:3, 0:3]
        current_position = pose_world[0:3, 3]

        # Convert to NWU frame
        conversion_matrix = np.diag([1, -1, -1])  # Flip Y (East->West) and Z (Down->Up)
        R_NWU = conversion_matrix @ pose_rot @ conversion_matrix.T  # Now in NWU frame
        current_attitude = R.from_matrix(R_NWU).as_euler("XYZ", degrees=False)
        
        # Get current velocity
        dynamics_world = state["DynamicsSensor"]
        vel_body = np.dot(np.linalg.inv(rotmat), dynamics_world[3:6])
        vel_current = np.array([
            vel_body[0], vel_body[1], vel_body[2],
            angular_rates[0], angular_rates[1], angular_rates[2]
        ])
        
        # Reset command
        self.current_command = np.zeros(8)
        
        # Set target values
        position_setpoint = self.current_target_pose['position']
        
        # Convert quaternion to Euler angles for attitude control
        if len(self.current_target_pose['orientation']) == 4:  # Quaternion
            quat = self.current_target_pose['orientation']
            # Get rotation matrix from quaternion
            target_rot = R.from_quat(quat).as_matrix()
            
            # # Apply the same NWU transformation 
            # conversion_matrix = np.diag([1, -1, -1])
            # target_rot_NWU = conversion_matrix @ target_rot @ conversion_matrix.T
            
            # Convert to Euler angles
            attitude_setpoint = R.from_matrix(target_rot).as_euler("XYZ", degrees=False)
        else:  # Already Euler
            target_rot = R.from_euler("XYZ", self.current_target_pose['orientation'], degrees=False).as_matrix()
            
            # # Apply NWU transformation
            # conversion_matrix = np.diag([1, -1, -1])
            # target_rot_NWU = conversion_matrix @ target_rot @ conversion_matrix.T
            
            # Convert back to Euler angles
            attitude_setpoint = R.from_matrix(target_rot).as_euler("XYZ", degrees=False)
        
        # Default velocity setpoint
        yaw_rate_setpoint = 0.0
        vel_setpoint = np.array([0.0, 0.0, 0.0, 0.0, 0.0, yaw_rate_setpoint])
        
        # Generate control forces
        vel_control_force = self.vel_controller.update(
            vel_current, vel_setpoint, rotmat, dt, state["t"], clip_force=20.0)
            
        position_control_force = self.position_controller.update(
            current_position, position_setpoint, rotmat, dt, state["t"], 
            rotate_error_to_body=True, clip_force=30.0)
            
        attitude_control_force = self.attitude_controller.update(
            current_attitude, attitude_setpoint, rotmat, dt, state["t"], 
            attitude=True, clip_force=40.0)
        
        # Apply control forces to actuators
        self.current_command[4:8] += vel_control_force[0] + position_control_force[0]  # x

        self.current_command[[5,7]] += vel_control_force[1] + position_control_force[1]  # y
        self.current_command[[4,6]] -= vel_control_force[1] + position_control_force[1]  # -y

        self.current_command[0:4] -= vel_control_force[2] + position_control_force[2]  # z
                    
        # yaw
        self.current_command[[5,6]] += vel_control_force[5] - attitude_control_force[2]
        self.current_command[[4,7]] -= vel_control_force[5] - attitude_control_force[2]
        
        # Update controller states
        self.vel_controller.step(state["t"])
        self.position_controller.step(state["t"])
        self.attitude_controller.step(state["t"])

    def set_target_pose(self, position, orientation):
        """Set a new target pose"""
        self.current_target_pose = {
            'position': np.array(position),
            'orientation': np.array(orientation)
        }
        
    def get_command(self):
        """Return the current command"""
        return self.current_command
        
    def publish_pid_errors(self):
        """Publish PID controller errors for visualization"""
        # Position error
        if hasattr(self.position_controller, 'error') and self.position_controller.error is not None:
            pos_err_msg = Vector3()
            pos_err_msg.x = self.position_controller.error[0]
            pos_err_msg.y = self.position_controller.error[1]
            pos_err_msg.z = self.position_controller.error[2]
            self.pos_error_pub.publish(pos_err_msg)
        
        # Velocity error
        if hasattr(self.vel_controller, 'error') and self.vel_controller.error is not None:
            vel_err_msg = Vector3()
            # Extract the first 3 components (linear velocities)
            vel_err_msg.x = self.vel_controller.error[0]
            vel_err_msg.y = self.vel_controller.error[1]
            vel_err_msg.z = self.vel_controller.error[2]
            self.vel_error_pub.publish(vel_err_msg)
        
        # Attitude error
        if hasattr(self.attitude_controller, 'error') and self.attitude_controller.error is not None:
            att_err_msg = Vector3()
            att_err_msg.x = self.attitude_controller.error[0]  * (180 / np.pi)  # roll
            att_err_msg.y = self.attitude_controller.error[1]  * (180 / np.pi)  # pitch
            att_err_msg.z = self.attitude_controller.error[2]  * (180 / np.pi) # yaw
            self.att_error_pub.publish(att_err_msg)