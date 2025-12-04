import numpy as np
from data.resources.constants import MU_EARTH
from utils.frame_conversions.rel_to_inertial_functions import LVLH_DCM, inertial_to_rel_LVLH
from utils.orbital_element_conversions.oe_conversions import lroes_to_inertial
def _get_desired_state_LVLH(frame: str, desired_state: list, r_c, v_c, sim_time):
    """
    Extract desired relative state in LVLH frame from configuration.
    """

    if frame == "LVLH":
        rho_des      = np.array(desired_state[:3])
        rho_dot_des  = np.array(desired_state[3:])
    elif frame == "LROES":
        lroes_des = desired_state
        r_d_des, v_d_des = lroes_to_inertial(
            sim_time,
            r_c,
            v_c,
            lroes_des
        )
        rho_des, rho_dot_des = inertial_to_rel_LVLH(
            r_d_des,
            v_d_des,
            r_c,
            v_c
        )
    else:
        raise ValueError(f"Frame '{frame}' not recognized for desired state. Use 'LVLH' or 'LROES'.")

    return rho_des, rho_dot_des

def _compute_A1(n):

    A1 = np.array([
        [3*n**2, 0,   0  ],
        [0,      0,   0  ],
        [0,      0, -n**2]
    ])

    return A1

def _compute_A2(n):

    A2 = np.array([
        [ 0  , 2*n, 0],
        [-2*n,  0 , 0],
        [ 0  ,  0 , 0]
    ])

    return A2

def cwh_step(state: dict, config: dict) -> dict:
    """
    GNC step implementing u = v_dot_star - A1*Δr - A2*Δr_dot - Kp*Δr - Kd*Δr_dot
    Returns command acceleration in inertial frame for dynamics propagation.
    """

    # Extract chief inertial position
    r_chief = np.array(state["chief_r"])
    v_chief = np.array(state["chief_v"])

    # Extract relative vectors
    rho_deputy = np.array(state["deputy_rho"])
    rho_dot_deputy = np.array(state["deputy_rho_dot"])

    ## Guidance
    # Extract desired relative position and velocity
    rho_des, rho_dot_des = _get_desired_state_LVLH(
        config.get("guidance", {}).get("rpo", {}).get("frame", "LVLH").upper(),
        config.get("guidance", {}).get("rpo", {}).get("deputy_desired_relative_state"),
        r_chief,
        v_chief,
        state.get("sim_time", 0.0)
    )

    # Get Kp and Kd from config (scalar or list), default to 1s
    Kp = config.get("control", {}).get("gains", {}).get("K1", 1.0)
    Kd = config.get("control", {}).get("gains", {}).get("K2", 1.0)

    # Compute error terms
    delta_r = rho_deputy - rho_des
    delta_r_dot = rho_dot_deputy - rho_dot_des

    # Compute DCM from the LVLH (orbit) frame to the inertial frame
    C_H_N = LVLH_DCM(r_chief, v_chief)

    # Compute A1 and A2 matrices
    n = np.sqrt(MU_EARTH / np.linalg.norm(r_chief)**3)  # mean motion
    A1 = _compute_A1(n)
    A2 = _compute_A2(n)

    # Compute v_star
    v_star = C_H_N @ (A2 @ rho_deputy + A1 @ rho_dot_deputy)

    # PD control
    u = - A1 @ rho_deputy - A2 @ rho_dot_deputy - Kp * delta_r - Kd * delta_r_dot

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
