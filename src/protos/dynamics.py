import numpy as np
import os

os.chdir("H:/tanne/Documents/PROTOS")

# Import helper functions
from utils.frame_convertions.rel_to_inertial_functions import LVLH_basis_vectors, LVLH_DCM, inertial_to_LVLH, inertial_to_rel_vector


# Earth constants
MU_EARTH = 398600.4418  # km^3/s^2
R_EARTH = 6378.137      # km
J2 = 1.08263e-3

# -------------------------------
# Main step
# -------------------------------
def step(state: dict, dt: float, config: dict):
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
# CWH propagator
# -------------------------------
def _step_cwh(state: dict, dt: float, config: dict):
    """
    CWH linearized relative motion.
    Chief in ECI, deputy in LVLH relative to chief.
    """
    chief_r = state["chief_r"]
    chief_v = state["chief_v"]
    deputy_r = state["deputy_r_LVLH"] if "deputy_r_LVLH" in state else state["deputy_r"]
    deputy_v = state["deputy_v_LVLH"] if "deputy_v_LVLH" in state else state["deputy_v"]

    sim = config.get("simulation", {})
    perturb = sim.get("perturbations", {})

    # Mean motion
    r_mag = np.linalg.norm(chief_r)
    n = np.sqrt(MU_EARTH / r_mag**3)

    x, y, z = deputy_r
    vx, vy, vz = deputy_v

    # CWH accelerations
    ax = 3 * n**2 * x + 2 * n * vy
    ay = -2 * n * vx
    az = -n**2 * z

    # Optional J2 in LVLH
    if perturb.get("J2", False):
        r_total = np.linalg.norm(chief_r + deputy_r)
        z2 = (chief_r[2] + z)**2
        factor = 1.5 * J2 * MU_EARTH * R_EARTH**2 / r_total**5
        ax -= factor * (1 - 5*z2/r_total**2) * (chief_r[0] + x)
        ay -= factor * (1 - 5*z2/r_total**2) * (chief_r[1] + y)
        az -= factor * (3 - 5*z2/r_total**2) * (chief_r[2] + z)

    # Euler integration for deputy LVLH
    next_r_LVLH = deputy_r + deputy_v * dt
    next_v_LVLH = deputy_v + np.array([ax, ay, az]) * dt

    # Chief propagation in circular orbit (ECI)
    theta = n * dt
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R = np.array([[cos_theta, -sin_theta, 0],
                  [sin_theta,  cos_theta, 0],
                  [0,          0,         1]])
    chief_r_next = R @ chief_r
    chief_v_next = R @ chief_v

    # Convert deputy back to ECI
    C = LVLH_DCM(chief_r_next, chief_v_next)
    deputy_r_next = chief_r_next + C @ next_r_LVLH
    deputy_v_next = chief_v_next + C @ next_v_LVLH

    return {
        "chief_r": chief_r_next,
        "chief_v": chief_v_next,
        "deputy_r": deputy_r_next,
        "deputy_v": deputy_v_next,
        "deputy_r_LVLH": next_r_LVLH,
        "deputy_v_LVLH": next_v_LVLH
    }

# -------------------------------
# TH propagator (placeholder)
# -------------------------------
def _step_th(state: dict, dt: float, config: dict):
    """
    TH propagator with LVLH output.
    Currently a placeholder: copies LVLH state if present.
    """
    chief_r = state["chief_r"]
    chief_v = state["chief_v"]
    deputy_r = state.get("deputy_r", np.zeros(3))
    deputy_v = state.get("deputy_v", np.zeros(3))

    rho, rho_dot = inertial_to_rel_vector(deputy_r, deputy_v, chief_r, chief_v)

    return {
        "chief_r": chief_r,
        "chief_v": chief_v,
        "deputy_r": deputy_r,
        "deputy_v": deputy_v,
        "deputy_r_LVLH": rho,
        "deputy_v_LVLH": rho_dot
    }

# -------------------------------
# 2-body propagator with RK4 + LVLH
# -------------------------------
def _step_2body(state: dict, dt: float, config: dict):
    """
    2-body RK4 propagator.
    Outputs both inertial and LVLH relative deputy states.
    """
    chief_r = state["chief_r"]
    chief_v = state["chief_v"]
    deputy_r = state["deputy_r"]
    deputy_v = state["deputy_v"]
    sim = config.get("simulation", {})
    perturb = sim.get("perturbations", {})

    def accel(r):
        r_norm = np.linalg.norm(r)
        a = -MU_EARTH * r / r_norm**3
        if perturb.get("J2", False):
            z2 = r[2]**2
            factor = 1.5 * J2 * MU_EARTH * R_EARTH**2 / r_norm**5
            a[0] -= factor*(1 - 5*z2/r_norm**2)*r[0]
            a[1] -= factor*(1 - 5*z2/r_norm**2)*r[1]
            a[2] -= factor*(3 - 5*z2/r_norm**2)*r[2]
        return a

    def rk4_step(r, v, dt):
        k1_r, k1_v = v, accel(r)
        k2_r, k2_v = v + 0.5*dt*k1_v, accel(r + 0.5*dt*k1_r)
        k3_r, k3_v = v + 0.5*dt*k2_v, accel(r + 0.5*dt*k2_r)
        k4_r, k4_v = v + dt*k3_v, accel(r + dt*k3_r)
        r_next = r + (dt/6)*(k1_r + 2*k2_r + 2*k3_r + k4_r)
        v_next = v + (dt/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
        return r_next, v_next

    chief_r_next, chief_v_next = rk4_step(chief_r, chief_v, dt)
    deputy_r_next, deputy_v_next = rk4_step(deputy_r, deputy_v, dt)

    rho, rho_dot = inertial_to_rel_vector(deputy_r_next, deputy_v_next, chief_r_next, chief_v_next)

    return {
        "chief_r": chief_r_next,
        "chief_v": chief_v_next,
        "deputy_r": deputy_r_next,
        "deputy_v": deputy_v_next,
        "deputy_r_LVLH": rho,
        "deputy_v_LVLH": rho_dot
    }


