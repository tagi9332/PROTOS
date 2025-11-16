"""Quiz 8 controller implementing PD control with gravitational feedforward"""

import numpy as np
from data.resources.constants import MU_EARTH as mu_e, R_EARTH as r_e
from utils.orbital_element_conversions.oe_conversions import inertial_to_orbital_elements, ta_to_m, normalize_angle
from utils.frame_conversions.rel_to_inertial_functions import LVLH_DCM
from utils.differential_orbital_elements.B_matrix import B_matrix

def mean_oe_step(state: dict, config: dict) -> dict:
    """
    GNC step implementing orbit element control law u = - [P]*([B])^T * ([A(oe_actual)] - [A(oe_desired)])
    Returns command acceleration in inertial frame for dynamics propagation.
    """

    # Extract chief inertial position
    r_chief = np.array(state["chief_r"])
    v_chief = np.array(state["chief_v"])

    # Extract deputy inertial position
    r_deputy = np.array(state["deputy_r"])
    v_deputy = np.array(state["deputy_v"])

    # Extract deputy control state vector
    del_oe_desired = np.array(config["desired_relative_state"]["state"])

    # If desired state is given in LVLH, return error message
    if config["desired_relative_state"].get("frame").upper() != "DOES":
        raise ValueError("Desired state for mean_OEs control must be given in differential orbital elements (dOEs).")

    # Extract relative vectors
    deputy_rho = np.array(state["deputy_rho"])
    deputy_rho_dot = np.array(state["deputy_rho_dot"])

    # Guidance
    # Convert chief and deputy current inertial to classical orbital elements
    oe_chief = np.array(inertial_to_orbital_elements(r_chief, v_chief))
    oe_deputy = np.array(inertial_to_orbital_elements(r_deputy, v_deputy))
    
    B = B_matrix(oe_deputy[0], oe_deputy[1], oe_deputy[2], oe_deputy[4], oe_deputy[5])

    # Get P matrix from config
    P = np.zeros((3, 3))
    P[0,0] = config.get("control", {}).get("gains", {}).get("K1", 1.0)
    P[1,1] = config.get("control", {}).get("gains", {}).get("K2", 1.0)
    P[2,2] = config.get("control", {}).get("gains", {}).get("K3", 1.0)

    # Convert del_oe_desired angles from degrees to radians
    del_oe_desired[3:] = np.radians(del_oe_desired[3:])

    # Compute desired orbit elements
    oe_deputy_desired = oe_chief + del_oe_desired

    # Convert true anomalies to mean anomalies for both deputy and desired deputy
    oe_deputy[5] = ta_to_m(oe_deputy[5], oe_deputy[1])
    oe_deputy_desired[5] = ta_to_m(oe_deputy_desired[5], oe_chief[1])

    # Compute error terms
    delta_oe = oe_deputy - oe_deputy_desired
    delta_oe = delta_oe.reshape(-1,1)  # ensure column vector
    delta_oe[0] = delta_oe[0] / r_e  # normalize semi-major axis error by Earth radius

    # Angle wrapping for inclination, RAAN, argument of perigee, mean anomaly
    for idx in range(3,6):
        delta_oe[idx] = normalize_angle(delta_oe[idx])

    # PD control
    u = - P @ (B.T @ delta_oe)

    # Compute DCM from LVLH to inertial
    C_H_N = LVLH_DCM(r_deputy, v_deputy)

    # Transform command acceleration to inertial frame
    u = C_H_N.T @ u  # from LVLH to inertial
    u = u.flatten()

    # # Enforce saturation limit **TODO**
    # max_thrust = config.get("control", {}).get("max_thrust", None)  # in Newtons
    # if max_thrust is not None:
    #     mass = config.get("spacecraft", {}).get("deputy_mass", 1.0)  # in kg
    #     max_accel = max_thrust / mass  # in km/s^2 assuming inputs in km/s^2
    #     norm_u = np.linalg.norm(u)
    #     if norm_u > max_accel:
    #         u = (u / norm_u) * max_accel

    return {
        "status": "lvlh",
        "chief_r": state["chief_r"].tolist(),
        "chief_v": state["chief_v"].tolist(),
        "deputy_r": state["deputy_r"].tolist(),
        "deputy_v": state["deputy_v"].tolist(),
        "deputy_rho": deputy_rho.tolist(),
        "deputy_rho_dot": deputy_rho_dot.tolist(),
        "accel_cmd": u.tolist()
    }
