import numpy as np

class PIDController:
    def __init__(self, Kp=None, Ki=None, Kd=None, drag_coeff_p1=None, drag_coeff_p2=None, state_dim = 6):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.drag_coeff_p1 = drag_coeff_p1
        self.drag_coeff_p2 = drag_coeff_p2
        
        self.state_dim = state_dim
        self.error = np.zeros(self.state_dim)
        self.prev_error = np.zeros(self.state_dim)
        self.prev_time = 0.0
        self.error_buffer = np.zeros((self.state_dim, 1000))
        self.state_buffer = np.zeros((self.state_dim+1, 500))
        self.setpoint_buffer = np.zeros((self.state_dim+1, 500))
        
    def update(self, current_state, setpoint_state, rot_mat, dt, time, rotate_error_to_body = False, attitude = False, clip_force = None):

        # store the velocity in a buffer
        self.state_buffer[:, 1:] = self.state_buffer[:, :-1]
        self.state_buffer[:-1, 0], self.state_buffer[-1, 0] = current_state, time
        
        # store the setpoint in a buffer
        self.setpoint_buffer[:, 1:] = self.setpoint_buffer[:, :-1]
        self.setpoint_buffer[:-1, 0], self.setpoint_buffer[-1, 0] = setpoint_state, time

        # error term
        self.error = setpoint_state - current_state
        
        if rotate_error_to_body:
            self.error = np.dot(rot_mat, self.error)
        elif attitude:
            self.error = np.arctan2(np.sin(self.error), np.cos(self.error))

        # derivative term
        error_vel_dot = (self.error - self.prev_error )/dt       
        
        # integral term: fill the buffer with the error
        self.error_buffer[:, 1:] = self.error_buffer[:, :-1]                      # rolling buffer
        self.error_buffer[:, 0] = self.error
        
        error_int = np.sum(self.error_buffer*dt, axis=1)
        
        if self.Kp is not None:
            control_force = np.dot(self.Kp, self.error)
        if self.Ki is not None:
            control_force += np.dot(self.Ki, error_int)
        if self.Kd is not None:
            control_force += np.dot(self.Kd, error_vel_dot)
        if self.drag_coeff_p1 is not None:
            control_force += np.dot(self.drag_coeff_p1, current_state)
        if self.drag_coeff_p2 is not None:
            control_force += np.dot(self.drag_coeff_p2, (current_state**3)/np.abs(current_state))
        
        # control allocation
        if clip_force is not None:
            control_force = np.clip(control_force, -clip_force, clip_force)
            return control_force
        else:
            return control_force

    def step(self, prev_time):
        self.prev_error = self.error
        self.prev_time = prev_time
