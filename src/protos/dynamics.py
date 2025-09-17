import numpy as np
from utils.frame_convertions.rel_to_inertial_functions import LVLH_DCM, rel_vector_to_inertial 

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
# 2-Body Step Implementation with RK4
# -------------------------------
def _step_2body(state: dict, dt: float, config: dict):
    """
    Propagate chief and deputy using two-body dynamics in inertial frame with RK4.
    state: dict containing chief and deputy positions/velocities
    dt: time step [s]
    config: simulation config dict
    """
    chief_r = state["chief_r"]
    chief_v = state["chief_v"]
    deputy_r = state["deputy_r"]
    deputy_v = state["deputy_v"]
    sim = config.get("simulation", {})
    perturb = sim.get("perturbations", {})

    # Acceleration function
    def accel(r):
        r_mag = np.linalg.norm(r)
        a = -MU_EARTH * r / r_mag**3

        # Optional J2 perturbation
        if perturb.get("J2", False):
            z2 = r[2] ** 2
            factor = 1.5 * J2 * MU_EARTH * R_EARTH**2 / r_mag**5
            a[0] -= factor * (1 - 5 * z2 / r_mag**2) * r[0]
            a[1] -= factor * (1 - 5 * z2 / r_mag**2) * r[1]
            a[2] -= factor * (3 - 5 * z2 / r_mag**2) * r[2]
        return a

    # RK4 integration for chief
    k1_vc = accel(chief_r) * dt
    k1_rc = chief_v * dt

    k2_vc = accel(chief_r + 0.5 * k1_rc) * dt
    k2_rc = (chief_v + 0.5 * k1_vc) * dt

    k3_vc = accel(chief_r + 0.5 * k2_rc) * dt
    k3_rc = (chief_v + 0.5 * k2_vc) * dt

    k4_vc = accel(chief_r + k3_rc) * dt
    k4_rc = (chief_v + k3_vc) * dt

    chief_r_next = chief_r + (k1_rc + 2*k2_rc + 2*k3_rc + k4_rc) / 6
    chief_v_next = chief_v + (k1_vc + 2*k2_vc + 2*k3_vc + k4_vc) / 6

    # RK4 integration for deputy
    k1_vd = accel(deputy_r) * dt
    k1_rd = deputy_v * dt

    k2_vd = accel(deputy_r + 0.5 * k1_rd) * dt
    k2_rd = (deputy_v + 0.5 * k1_vd) * dt

    k3_vd = accel(deputy_r + 0.5 * k2_rd) * dt
    k3_rd = (deputy_v + 0.5 * k2_vd) * dt

    k4_vd = accel(deputy_r + k3_rd) * dt
    k4_rd = (deputy_v + k3_vd) * dt

    deputy_r_next = deputy_r + (k1_rd + 2*k2_rd + 2*k3_rd + k4_rd) / 6
    deputy_v_next = deputy_v + (k1_vd + 2*k2_vd + 2*k3_vd + k4_vd) / 6

    # Compute updated relative state in LVLH frame
    C_HN = LVLH_DCM(chief_r_next, chief_v_next) 
    deputy_rho_next = C_HN @ (deputy_r_next - chief_r_next)
    omega = np.array([0, 0, np.linalg.norm(np.cross(chief_r_next, chief_v_next)) / np.dot(chief_r_next, chief_r_next)])
    deputy_rho_dot_next = C_HN @ (deputy_v_next - chief_v_next) - np.cross(omega, deputy_rho_next)  # assuming zero angular velocity for simplicity

    # Return updated state
    return {
        "chief_r": chief_r_next,
        "chief_v": chief_v_next,
        "deputy_r": deputy_r_next,
        "deputy_v": deputy_v_next,
        "deputy_rho": deputy_rho_next,
        "deputy_rho_dot": deputy_rho_dot_next
    }

