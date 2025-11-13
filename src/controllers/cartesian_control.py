import numpy as np
from utils.frame_convertions.rel_to_inertial_functions import LVLH_DCM, rel_vector_to_inertial, compute_omega

def grav_accel(r):
    MU_EARTH = 398600.4418  # km^3/s^2
    r_mag = np.linalg.norm(r)
    return -MU_EARTH * r / r_mag**3

def cartesian_step(state: dict, config: dict) -> dict:
    """
    GNC step implementing u = -(f(r_d) - f(r_dd)) - K1*Δr - K2*Δr_dot + u_d
    Returns command acceleration in inertial frame for dynamics propagation.
    """

    # Extract chief inertial position
    r_chief = np.array(state["chief_r"])
    v_chief = np.array(state["chief_v"])

    # Extract deputy inertial position
    r_deputy = np.array(state["deputy_r"])
    v_deputy = np.array(state["deputy_v"])

    # Extract relative vectors
    deputy_rho = np.array(state["deputy_rho"])
    deputy_rho_dot = np.array(state["deputy_rho_dot"])

    ## Guidance
    # Extract desired relative position and velocity
    desired_state = config.get("desired_relative_state", {})
    deputy_rho_des = np.array(desired_state.get("deputy_rho_des", [0, 0, 0]))
    deputy_rho_dot_des = np.array(desired_state.get("deputy_rho_dot_des", [0, 0, 0]))

    # Convert desired relative state to desired inertial deputy state
    r_deputy_des, v_deputy_des = rel_vector_to_inertial(deputy_rho_des, deputy_rho_dot_des, r_chief, v_chief)

    # Get Kp and Kd from config (scalar or list), default to 1s
    Kp = config.get("control", {}).get("pd", {}).get("Kp", 1.0)
    Kd = config.get("control", {}).get("pd", {}).get("Kd", 1.0)

    # Compute error terms
    delta_r = r_deputy - r_deputy_des
    delta_r_dot = v_deputy - v_deputy_des

    # Compute gravitational acceleration terms
    f_d = grav_accel(r_deputy)
    f_dd = grav_accel(r_deputy_des)

    # Compute DCM from the LVLH (orbit) frame to the inertial frame
    C_H_N = LVLH_DCM(r_chief, v_chief)

    # Compute the chief satellite's specific angular momentum vector
    h = np.cross(r_chief, v_chief)

    # Compute the LVLH frame angular velocity vector in inertial coordinates
    omega = h / np.linalg.norm(r_chief)**2

    # Compute the radial velocity of the chief
    rdot = np.dot(r_chief, v_chief) / np.linalg.norm(r_chief)

    # Compute the time derivative of the LVLH angular velocity (frame acceleration)
    omegadot = (-2 * rdot / np.linalg.norm(r_chief)**3) * h

    # Transform the desired relative position from LVLH to inertial frame
    deputy_rho_I_des = C_H_N.T @ deputy_rho_des

    # Compute the inertial acceleration contribution due to LVLH frame rotation
    rho_ddot_I = np.cross(omegadot, deputy_rho_I_des) + np.cross(omega, np.cross(omega, deputy_rho_I_des))

    # Compute control feedforward terms
    u_d = grav_accel(r_chief) - grav_accel(r_deputy_des) + rho_ddot_I

    # PD control
    u = -(f_d - f_dd) - Kp * delta_r - Kd * delta_r_dot + u_d

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
