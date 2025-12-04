import numpy as np
from utils.numerical_methods.rk54 import rk54

def q_step(dt: float, state: dict, config: dict):
    # 1. Safety Check: Timestep vs Inertia
    # If dt is too large, the ZOH torque assumption causes divergence.
    if dt > 0.1: 
        print(f"WARNING: q_step dt is {dt}s. Large steps with constant torque cause instability in 6DOF mode.")

    # 2. Pre-compute Inverse Inertia (Optimization)
    J_chief = np.array(config['satellite_properties']['chief']['inertia_matrix'])
    J_inv_chief = np.linalg.inv(J_chief)
    
    J_deputy = np.array(config['satellite_properties']['deputy']['inertia_matrix'])
    J_inv_deputy = np.linalg.inv(J_deputy)

    def dynamics_func(y, J, J_inv, torque):
        q = y[0:4]
        omega = y[4:7]
        
        # Kinematics
        Omega = np.array([
            [0.0, -omega[0], -omega[1], -omega[2]],
            [omega[0], 0.0, omega[2], -omega[1]],
            [omega[1], -omega[2], 0.0, omega[0]],
            [omega[2], omega[1], -omega[0], 0.0]
        ])
        q_dot = 0.5 * Omega @ q
        
        # Dynamics
        # torque must be in BODY frame
        w_dot = J_inv @ (torque - np.cross(omega, J @ omega))
        
        return np.concatenate([q_dot, w_dot])

    # -----------------------------
    # Extract State
    # -----------------------------
    chief_y = np.concatenate([state['chief_q_BN'], state['chief_omega_BN']])
    deputy_y = np.concatenate([state['deputy_q_BN'], state['deputy_omega_BN']])
    
    torque_cmd_chief = np.array(state.get('torque_cmd_chief', [0,0,0]))
    torque_cmd_deputy = np.array(state.get('torque_cmd_deputy', [0,0,0]))

    # -----------------------------
    # Integrate
    # -----------------------------
    chief_y_next = rk54(
        lambda y: dynamics_func(y, J_chief, J_inv_chief, torque_cmd_chief), 
        chief_y, dt
    )
    
    deputy_y_next = rk54(
        lambda y: dynamics_func(y, J_deputy, J_inv_deputy, torque_cmd_deputy), 
        deputy_y, dt
    )

    # -----------------------------
    # Normalize & Return
    # -----------------------------
    attitude_state = {
        'chief_q_BN': chief_y_next[0:4] / np.linalg.norm(chief_y_next[0:4]),
        'chief_omega_BN': chief_y_next[4:7],
        'deputy_q_BN': deputy_y_next[0:4] / np.linalg.norm(deputy_y_next[0:4]),
        'deputy_omega_BN': deputy_y_next[4:7]
    }

    return attitude_state