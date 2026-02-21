"""Quiz 8 controller implementing PD control with gravitational feedforward"""

import numpy as np
from data.resources.constants import MU_EARTH as mu_e, R_EARTH as r_e
from utils.orbital_element_conversions.oe_conversions import inertial_to_oes, ta_to_m, normalize_angle
from utils.frame_conversions.rel_to_inertial_functions import LVLH_DCM

def _B_matrix(a, e, i, argp, ta):
    """
    [B(oe)] matrix mapping accelerations [a_r, a_t, a_n] into 
    classical orbital element rates [a, e, i, Ω, ω, M].

    Inputs:
    - a     : semi-major axis
    - e     : eccentricity
    - f     : true anomaly
    - i     : inclination
    - argp  : argument of perigee
    - ta    : true anomaly
    """

    # Basic parameters
    p = a * (1 - e**2)              # semi-latus rectum
    h = np.sqrt(mu_e * p)           # specific angular momentum
    r = p / (1 + e * np.cos(ta))     # orbit radius
    true_lat = ta + argp         # true latitude

    # Define eta = sqrt(1 - e^2)
    eta = np.sqrt(1 - e**2)

    # ----- Build the full 6×3 B matrix -----
    B = np.zeros((6, 3))

    # Row 1: da/dt terms
    B[0, 0] = 2 * a**2 * e * np.sin(ta) / (h * r_e)
    B[0, 1] = 2 * a**2 * p / (h * r * r_e)
    B[0, 2] = 0

    # Row 2: de/dt terms
    B[1, 0] = p * np.sin(ta) / h
    B[1, 1] = ((p + r) * np.cos(ta) + r * e) / h
    B[1, 2] = 0

    # Row 3: di/dt terms
    B[2, 0] = 0
    B[2, 1] = 0
    B[2, 2] = (r * np.cos(true_lat)) / h

    # Row 4: dΩ/dt terms
    B[3, 0] = 0
    B[3, 1] = 0
    B[3, 2] = (r * np.sin(true_lat)) / (h * np.sin(i))

    # Row 5: dω/dt terms
    B[4, 0] = -p * np.cos(ta) / (h * e)
    B[4, 1] = (p + r) * np.sin(ta) / (h * e)
    B[4, 2] = -(r * np.sin(true_lat) * np.cos(i)) / (h * np.sin(i))

    # Row 6: dθ/dt terms
    B[5, 0] = eta * (p * np.cos(ta) - 2 * r * e) / (h * e)
    B[5, 1] = -eta * (p + r) * np.sin(ta) / (h * e)
    B[5, 2] = 0

    return B

def mean_oe_step(state: dict, config: dict, sat_name: str) -> dict:
    """
    GNC step implementing orbit element control law u = - [P]*([B])^T * ([A(oe_actual)] - [A(oe_desired)])
    Returns command acceleration in inertial frame for dynamics propagation.
    """
    # Extract chief inertial position
    r_chief = np.array(state["chief"]["r"])
    v_chief = np.array(state["chief"]["v"])

    # Extract deputy inertial position
    deputy_state = state["deputies"][sat_name]
    r_d = np.array(deputy_state["r"])
    v_d = np.array(deputy_state["v"])

    # Convert chief and deputy current inertial to classical orbital elements
    oe_c = np.array(inertial_to_oes(r_chief, v_chief))
    oe_d = np.array(inertial_to_oes(r_d, v_d))

    # Extract guidance configuration and desired state
    guidance_rpo = config.get("guidance", {}).get("rpo", {})
    desired_state_dict = guidance_rpo.get("desired_relative_state", {})

    # If desired state is given in LVLH, return error message
    if desired_state_dict.get("frame", "").upper() != "DOES":
        raise ValueError("Desired state for mean_OEs control must be given in differential orbital elements (dOEs).")
    
    # Extract deputy control state vector
    del_oe_des = np.array(desired_state_dict["state"])
    del_oe_des[3:] = np.radians(del_oe_des[3:])

    # Compute desired orbit elements
    oe_d_des = oe_c + del_oe_des

    # Compute B matrix
    B = _B_matrix(oe_d[0], oe_d[1], oe_d[2], oe_d[4], oe_d[5])

    # Convert true anomalies to mean anomalies for both deputy and desired deputy
    oe_d[5] = ta_to_m(oe_d[5], oe_d[1])
    oe_d_des[5] = ta_to_m(oe_d_des[5], oe_c[1])

    # Compute error terms
    error_oe = (oe_d - oe_d_des).reshape(-1,1)
    error_oe[0] /= r_e # normalize semi-major axis error by Earth radius

    # Angle wrapping for i, RAAN, ARGP, mean anomaly
    for idx in range(3,6):
        error_oe[idx] = normalize_angle(error_oe[idx])

    # Get gains matrix from config
    control_gains = config.get("control", {}).get("gains", {})
    P = np.zeros((3, 3))
    P[0,0] = control_gains.get("K1", 1.0)
    P[1,1] = control_gains.get("K2", 1.0)
    P[2,2] = control_gains.get("K3", 1.0)

    # PD control
    u = - P @ (B.T @ error_oe).flatten()

    # Compute DCM from LVLH to inertial
    C_H_N = LVLH_DCM(r_d, v_d)

    # Transform command acceleration to inertial frame
    u = C_H_N.T @ u 

    return {
        "accel_cmd": u.tolist()
    }