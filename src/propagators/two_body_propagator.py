import numpy as np
from scipy.integrate import solve_ivp
from src.propagators.perturbation_accel import compute_perturb_accel
from data.resources.constants import MU_EARTH, J2000_EPOCH

def step_2body(sat_state: dict, dt: float, config: dict, **kwargs):
    """
    Generic 2-body propagator. Moves a single satellite forward by dt.
    """
    # Extract state for satellite
    r_initial = np.array(sat_state["r"])
    v_initial = np.array(sat_state["v"])
    u_ctrl = np.array(sat_state.get("accel_cmd", [0, 0, 0]))
    
    # Extract properties for satellite
    mass = sat_state.get("mass", 500.0)
    drag_params = {
        "Cd": sat_state.get("Cd", 2.2),
        "area": sat_state.get("area", 1.0)
    }

    # Global sim settings
    sim = getattr(config, "simulation", {})
    perturb_config = getattr(sim, "perturbations", {})
    epoch = getattr(sat_state, "epoch", J2000_EPOCH)

    # Define the Dynamics
    def dynamics(t, y):
        r = y[:3]
        v = y[3:]
        r_mag = np.linalg.norm(r)

        # 2-body gravity
        a = -MU_EARTH * r / r_mag**3
        
        # Perturbations
        a += compute_perturb_accel(r, v, perturb_config, drag_params, mass, epoch)
        
        # Add control
        a += u_ctrl 

        return np.hstack((v, a))

    # Integration
    y_0 = np.hstack((r_initial, v_initial))
    sol = solve_ivp(
        dynamics,
        (0, dt),
        y_0,
        method='RK45',
        rtol=1e-12,
        atol=1e-12
    )

    y_final = sol.y[:, -1]

    # Return updated vectors
    return {
        "r": y_final[:3].tolist(),
        "v": y_final[3:].tolist()
    }