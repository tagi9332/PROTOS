import numpy as np
from utils.numerical_methods.rk54 import rk54

def q_step(dt: float, state: dict, config: dict):
    """
    Propagate the attitude state one time step using quaternion integration with RK54.
    """
    def omega_dot(omega, inertia, control_torque):
        """Rigid body rotational dynamics: I*omega_dot = -omega x (I*omega) + u"""
        return np.linalg.inv(inertia) @ (control_torque - np.cross(omega, inertia @ omega))

    def attitude_derivative(y, inertia, control_torque):
        """
        Compute dy/dt for [q0, q1, q2, q3, omega_x, omega_y, omega_z]
        """
        q = y[0:4]
        omega = y[4:7]

        # Quaternion kinematics
        Omega = np.array([
            [0.0, -omega[0], -omega[1], -omega[2]],
            [omega[0], 0.0, omega[2], -omega[1]],
            [omega[1], -omega[2], 0.0, omega[0]],
            [omega[2], omega[1], -omega[0], 0.0]
        ])
        q_dot = 0.5 * Omega @ q

        # Angular velocity dynamics
        w_dot = omega_dot(omega, inertia, control_torque)

        return np.concatenate([q_dot, w_dot])

    # -----------------------------
    # Extract state and inertia
    # -----------------------------
    chief_y = np.concatenate([state['chief_q_BN'], state['chief_omega_BN']])
    deputy_y = np.concatenate([state['deputy_q_BN'], state['deputy_omega_BN']])

    chief_inertia = np.array(config['satellite_properties']['chief']['inertia_matrix'])
    deputy_inertia = np.array(config['satellite_properties']['deputy']['inertia_matrix'])

    # Placeholder torques (replace with attitude control)
    torque_cmd_chief = np.array(state.get('torque_cmd_chief', [0,0,0]))
    torque_cmd_deputy = np.array(state.get('torque_cmd_deputy', [0,0,0]))

    # -----------------------------
    # Integrate using RK54
    # -----------------------------
    chief_y_next = rk54(lambda y: attitude_derivative(y, chief_inertia, torque_cmd_chief), chief_y, dt)
    deputy_y_next = rk54(lambda y: attitude_derivative(y, deputy_inertia, torque_cmd_deputy), deputy_y, dt)

    # Normalize quaternions
    chief_q_next = chief_y_next[0:4] / np.linalg.norm(chief_y_next[0:4])
    chief_w_next = chief_y_next[4:7]

    deputy_q_next = deputy_y_next[0:4] / np.linalg.norm(deputy_y_next[0:4])
    deputy_w_next = deputy_y_next[4:7]

    # -----------------------------
    # Return updated attitude state
    # -----------------------------
    attitude_state = {
        'chief_q_BN': chief_q_next,
        'chief_omega_BN': chief_w_next,
        'deputy_q_BN': deputy_q_next,
        'deputy_omega_BN': deputy_w_next
    }

    return attitude_state
