import numpy as np
from utils.frame_convertions.rel_to_inertial_functions import rel_vector_to_inertial 

# Earth constants
MU_EARTH = 398600.4418  # km^3/s^2
R_EARTH = 6378.137      # km
J2 = 1.08263e-3

# -------------------------------
# Main step function for time-stepping
# -------------------------------
def step(state: dict, dt: float, config: dict):
    """
    Propagate the deputy state one time step using the selected scheme.
    
    state: 'chief_r', 'chief_v', 'deputy_r', 'deputy_v', 'deputy_rho', 'deputy_rho_dot'
    dt: timestep [s]
    config: dict from io_utils.parse_input["dynamics"]
    """
    propagator = config.get("simulation", {}).get("propagator", "CWH").upper()
    
    if propagator == "CWH":
        return _step_cwh(state, dt, config)
    elif propagator == "TH":
        return _step_th(state, dt, config)
    elif propagator == "2BODY":
        return _step_2body(state, dt, config)
    else:
        raise ValueError(f"Unknown propagator '{propagator}'")

# -------------------------------
# CWH step implementation
# -------------------------------
def _step_cwh(state: dict, dt: float, config: dict):
    chief_r = state["chief_r"]
    chief_v = state["chief_v"]
    deputy_r = state["deputy_r"]
    deputy_v = state["deputy_v"]
    deputy_rho = state["deputy_rho"]
    deputy_rho_dot = state["deputy_rho_dot"]
    sim = config.get("simulation", {})
    perturb = sim.get("perturbations", {})

    # Mean motion (assume circular chief orbit)
    r_mag = np.linalg.norm(chief_r)
    n = np.sqrt(MU_EARTH / r_mag**3)

    x, y, z = deputy_rho
    vx, vy, vz = deputy_rho_dot

    # CWH accelerations
    ax = 3 * n**2 * x + 2 * n * vy
    ay = -2 * n * vx
    az = -n**2 * z

    # Optional J2 perturbation
    if perturb.get("J2", False):
        r_total = np.linalg.norm(chief_r + deputy_rho)
        z2 = (chief_r[2] + z) ** 2
        factor = 1.5 * J2 * MU_EARTH * R_EARTH ** 2 / r_total ** 5
        ax -= factor * (1 - 5 * z2 / r_total ** 2) * (chief_r[0] + x)
        ay -= factor * (1 - 5 * z2 / r_total ** 2) * (chief_r[1] + y)
        az -= factor * (3 - 5 * z2 / r_total ** 2) * (chief_r[2] + z)

    # Euler integration of relative state for one step
    deputy_rho_next = deputy_rho + deputy_rho_dot * dt
    deputy_rho_dot_next = deputy_rho_dot + np.array([ax, ay, az]) * dt

    # Propagate chief using keplerian motion (circular orbit)
    theta = n * dt
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    STM = np.array([[cos_theta, -sin_theta, 0],
                  [sin_theta,  cos_theta, 0],
                  [0,          0,         1]])
    chief_r_next = STM @ chief_r
    chief_v_next = STM @ chief_v

    # Update deputy inertial position and velocity
    deputy_r_next, deputy_v_next = rel_vector_to_inertial(deputy_rho_next, deputy_rho_dot_next, chief_r_next, chief_v_next)

    return {
        "chief_r": chief_r_next,
        "chief_v": chief_v_next,
        "deputy_r": deputy_r_next,
        "deputy_v": deputy_v_next,
        "deputy_rho": deputy_rho_next,
        "deputy_rho_dot": deputy_rho_dot_next
    }

# -------------------------------
# TH step implementation (shell)
# -------------------------------
def _step_th(state: dict, dt: float, config: dict):
    """
    Placeholder for TH step function.
    """
    print("TH propagator step selected, not yet implemented.")
    return state.copy()


# -------------------------------
# 2-Body step implementation (shell)
# -------------------------------
def _step_2body(state: dict, dt: float, config: dict):
    chief_r = state["chief_r"]
    chief_v = state["chief_v"]
    deputy_r = state["deputy_r"]
    deputy_v = state["deputy_v"]
    sim = config.get("simulation", {})
    perturb = sim.get("perturbations", {})

    # Convert deputy position to inertial frame

    # Propagate both chief and deputy using 2-body dynamics
    





    return state.copy()