import numpy as np
from utils.orbital_element_conversions.oe_conversions import inertial_to_orbital_elements, orbital_elements_to_inertial
from utils.frame_convertions.rel_to_inertial_functions import LVLH_DCM, compute_omega

def mean_oe_step(state: dict, config: dict) -> dict:
    """
    Implements PD-type control using delta Orbital Elements (dOEs).
    Converts dOEs to LVLH relative position and velocity, then applies PD control.
    """
    # Extract chief inertial state
    chief_r = np.array(state["chief_r"])
    chief_v = np.array(state["chief_v"])
    rho = np.array(state["deputy_rho"])
    rho_dot = np.array(state["deputy_rho_dot"])

    # Desired dOEs from guidance
    rpo = config.get("guidance", {}).get("rpo", {})
    dOEs_des = np.array(rpo.get("deputy_desired_relative_state", [0,0,0,0,0,0]))
    
    # Chief orbital elements
    a_chief, e_chief, i_chief, RAAN_chief, AOP_chief, TA_chief = inertial_to_orbital_elements(chief_r, chief_v)

    # Apply delta OEs
    a_dep = a_chief + dOEs_des[0]
    e_dep = e_chief + dOEs_des[1]
    i_dep = i_chief + dOEs_des[2]
    RAAN_dep = RAAN_chief + dOEs_des[3]
    AOP_dep = AOP_chief + dOEs_des[4]
    TA_dep = TA_chief + dOEs_des[5]

    # Convert desired deputy OEs to inertial
    deputy_r_des, deputy_v_des = orbital_elements_to_inertial(a_dep, e_dep, i_dep, RAAN_dep, AOP_dep, TA_dep)

    # Convert to LVLH relative state
    C_HN = LVLH_DCM(chief_r, chief_v)
    omega = compute_omega(chief_r, chief_v)
    rho_des = C_HN @ (deputy_r_des - chief_r)
    rho_dot_des = C_HN @ (deputy_v_des - chief_v) - np.cross(omega, rho_des)

    # Gains
    Kp = np.diag(np.array(config.get("guidance", {}).get("Kp", [1,1,1])))
    Kd = np.diag(np.array(config.get("guidance", {}).get("Kd", [1,1,1])))

    # PD control
    delta_r = rho - rho_des
    delta_r_dot = rho_dot - rho_dot_des
    u = -Kp @ delta_r - Kd @ delta_r_dot

    return {
        "status": "doe",
        "chief_r": state["chief_r"].tolist(),
        "chief_v": state["chief_v"].tolist(),
        "deputy_r": state["deputy_r"].tolist(),
        "deputy_v": state["deputy_v"].tolist(),
        "deputy_rho": rho.tolist(),
        "deputy_rho_dot": rho_dot.tolist(),
        "accel_cmd": u.tolist()
    }