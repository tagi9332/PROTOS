import numpy as np
from scipy.integrate import solve_ivp

def q_step(dt: float, sat_state: dict, config: dict, **kwargs):
    """
    Propagates the attitude kinematics and dynamics for a single satellite.
    """
    # Extract State
    q_BN = np.array(sat_state.get('q_BN', [1.0, 0.0, 0.0, 0.0]))
    omega_BN = np.array(sat_state.get('omega_BN', [0.0, 0.0, 0.0]))
    
    # Extract Control and Properties
    torque_cmd = np.array(sat_state.get('torque_cmd', [0.0, 0.0, 0.0]))
    J = np.array(sat_state.get('inertia_matrix', np.eye(3))) 
    J_inv = np.linalg.inv(J)

    # Dynamics Function
    def dynamics_func(t, y):
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
        w_dot = J_inv @ (torque_cmd - np.cross(omega, J @ omega))
        
        return np.concatenate([q_dot, w_dot])

    # Integrate
    y0 = np.concatenate([q_BN, omega_BN])
    
    sol = solve_ivp(
        dynamics_func, 
        t_span=(0, dt), 
        y0=y0, 
        method='RK45', 
        rtol=1e-12, 
        atol=1e-12
    )
    
    y_next = sol.y[:, -1]

    # Normalize & Return
    q_next = y_next[0:4]
    q_next = q_next / np.linalg.norm(q_next)
    omega_next = y_next[4:7]

    return {
        'q_BN': q_next.tolist(),
        'omega_BN': omega_next.tolist()
    }