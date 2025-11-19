import numpy as np
from data.resources.constants import MU_EARTH
from utils.frame_conversions.rel_to_inertial_functions import LVLH_DCM, rel_vector_to_inertial, inertial_to_rel_LVLH
from utils.orbital_dynamics.orbital_accel import grav_accel
from utils.orbital_element_conversions.oe_conversions import lroes_to_inertial

def cwh_step(state: dict, config: dict) -> dict:
    """
    GNC step implementing u = v_dot_star - A1*Δr - A2*Δr_dot - Kp*Δr - Kd*Δr_dot
    Returns command acceleration in inertial frame for dynamics propagation.
    """

    # Extract simulation time
    sim_time = state.get("sim_time", 0.0)

    # Extract chief inertial position
    r_chief = np.array(state["chief_r"])
    v_chief = np.array(state["chief_v"])

    # Extract relative vectors
    rho_deputy = np.array(state["deputy_rho"])
    rho_dot_deputy = np.array(state["deputy_rho_dot"])

    ## Guidance
    # Extract desired relative position and velocity
    desired_state = np.array(config["desired_relative_state"]["state"])

    # Check for correct frame definition
    if config["desired_relative_state"].get("frame").upper() != "LROES":
        raise ValueError("Desired state for CWH control must be given in LROES frame.")

    # Convert LROEs to deputy relative position and velocity in LVLH frame
    deputy_r_des, deputy_v_des = lroes_to_inertial(sim_time, r_chief, v_chief, desired_state)
    deputy_rho_des, deputy_rho_dot_des = inertial_to_rel_LVLH(deputy_r_des, deputy_v_des, r_chief, v_chief)


    def compute_theta_dot_and_ddot(r_c, v_c):
        """
        Computes θ̇ and θ̈ for the chief orbit.
        Assumes orbit is near circular so ω_LVLH = h / R^2.
        """
        r = np.asarray(r_c)
        v = np.asarray(v_c)

        R = np.linalg.norm(r)
        h_vec = np.cross(r, v)
        h = np.linalg.norm(h_vec)

        # Define q1 and q2 for near-circular orbit
        # q1 = e * cos(ω + Ω)
        # q2 = e * sin(ω + Ω)

        # θ̇ = h / R^2
        theta_dot = h / (R**2)

        # radial rate Ṙ = (r · v) / R
        R_dot = np.dot(r, v) / R

        # θ̈ = -2*(Ṙ/R)*θ̇
        theta_ddot = -2.0 * (MU_EARTH / R**3) * (q1*np.sin(theta) - q2*np.cos(theta))

        return theta_dot, theta_ddot

    def compute_A1(r_c, v_c, MU_EARTH):
        R = np.linalg.norm(r_c)
        theta_dot, theta_ddot = compute_theta_dot_and_ddot(r_c, v_c)

        A1 = np.array([
            [2*MU_EARTH/R**3 + theta_dot**2,   theta_ddot,             0.0],
            [-theta_ddot,                theta_dot**2 - MU_EARTH/R**3, 0.0],
            [0.0,                        0.0,                   -MU_EARTH/R**3]
        ])

        return A1
    
    def compute_A2(r_c, v_c):
        theta_dot, _ = compute_theta_dot_and_ddot(r_c, v_c)

        A2 = np.array([
            [0.0,          2*theta_dot, 0.0],
            [-2*theta_dot, 0.0,         0.0],
            [0.0,          0.0,         0.0]
        ])

        return A2

    # Get Kp and Kd from config (scalar or list), default to 1s
    Kp = config.get("control", {}).get("gains", {}).get("K1", 1.0)
    Kd = config.get("control", {}).get("gains", {}).get("K2", 1.0)

    # Compute error terms
    delta_r = rho_deputy - deputy_rho_des
    delta_r_dot = rho_dot_deputy - deputy_rho_dot_des

    # Compute DCM from the LVLH (orbit) frame to the inertial frame
    C_H_N = LVLH_DCM(r_chief, v_chief)

    A1 = compute_A1(r_chief, v_chief, MU_EARTH)
    A2 = compute_A2(r_chief, v_chief)

    # PD control
    u = A1 @ deputy_rho_des + A2 @ deputy_rho_dot_des - A1 @ rho_deputy - A2 @ rho_dot_deputy - Kp * delta_r - Kd * delta_r_dot

    # Convert command acceleration to inertial frame
    u = C_H_N.T @ u  # transpose to go from LVLH to inert

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
        "deputy_rho": rho_deputy.tolist(),
        "deputy_rho_dot": rho_dot_deputy.tolist(),
        "accel_cmd": u.tolist()
    }
