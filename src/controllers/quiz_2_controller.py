import numpy as np
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial, inertial_to_orbital_elements, m_to_ta, ta_to_m


def grav_accel(r):
    MU_EARTH = 398600.4418  # km^3/s^2
    r_mag = np.linalg.norm(r)
    return -MU_EARTH * r / r_mag**3

def quiz_2_step(state: dict, config: dict) -> dict:
    """
    GNC step implementing u = -(f(r_d) - f(r_dd)) - K1*Δr - K2*Δr_dot
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

    # Guidance
    # Commanded relative state in differential orbital elements for quiz 2
    da_des = 0
    de_des = 0
    d_i_des = 0
    d_RAAN_des = 0
    d_AOP_des = 0
    d_M_des = np.radians(0.1)  # deg

    # Get chief orbital elements
    a_c, e_c, i_c, RAAN_c, AOP_c, TA_c = inertial_to_orbital_elements(r_chief, v_chief, units='deg')
    
    # Convert chief true anomaly to mean anomaly
    M_c = ta_to_m(np.radians(TA_c), e_c)

    # Compute desired deputy orbital elements
    a_d_des = a_c + da_des
    e_d_des = e_c + de_des
    i_d_des = i_c + d_i_des
    RAAN_d_des = RAAN_c + d_RAAN_des
    AOP_d_des = AOP_c + d_AOP_des
    M_d_des = M_c + d_M_des

    # Convert desired deputy mean anomaly to true anomaly
    TA_d_des = m_to_ta(M_d_des, e_d_des)

    # Convert desired relative deputy state to inertial
    r_deputy_des, v_deputy_des = orbital_elements_to_inertial(a_d_des, e_d_des, i_d_des, RAAN_d_des, AOP_d_des, np.degrees(TA_d_des), units='deg')

    # Get Kp and Kd from config (scalar or list), default to 1s
    Kp = config.get("control", {}).get("pd", {}).get("Kp", 1.0)
    Kd = config.get("control", {}).get("pd", {}).get("Kd", 1.0)

    # Compute error terms
    delta_r = r_deputy - r_deputy_des
    delta_r_dot = v_deputy - v_deputy_des

    # Compute gravitational feedforeward terms
    f_d = grav_accel(r_deputy)
    f_dd = grav_accel(r_deputy_des)

    # PD control
    u = -(f_d - f_dd) - Kp * delta_r - Kd * delta_r_dot

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
