import numpy as np
from utils.orbital_element_conversions.oe_conversions import lroes_to_inertial


def grav_accel(r):
    MU_EARTH = 398600.4418  # km^3/s^2
    r_mag = np.linalg.norm(r)
    return -MU_EARTH * r / r_mag**3

def quiz_3_step(state: dict, config: dict) -> dict:
    """
    GNC step implementing u = -(f(r_d) - f(r_dd)) - K1*Δr - K2*Δr_dot
    Returns command acceleration in inertial frame for dynamics propagation.
    """
    # Extract simulation time
    sim_time = state.get("sim_time", 0.0)

    # Extract chief inertial position
    r_chief = np.array(state["chief_r"])
    v_chief = np.array(state["chief_v"])

    # Extract deputy inertial position
    r_deputy = np.array(state["deputy_r"])
    v_deputy = np.array(state["deputy_v"])

    # Guidance
    # Commanded linearized relative orbital elements for quiz 3
    A_0 = 1.0  # km
    B_0 = 2.0  # km
    x_offset = 0  # km
    y_offset = 0  # km
    alpha = np.radians(0)  # radians
    beta = np.radians(0)  # radians

    # Convert desired LROEs to desired inertial deputy state
    lroes_des = [A_0, B_0, alpha, beta, x_offset, y_offset]
    r_deputy_des, v_deputy_des = lroes_to_inertial(sim_time, r_chief, v_chief, lroes_des)

    # Get Kp and Kd from config (scalar or list), default to 1s
    Kp = config.get("control", {}).get("gains", {}).get("K1", 1.0)
    Kd = config.get("control", {}).get("gains", {}).get("K2", 1.0)

    # Compute error terms
    delta_r = r_deputy - r_deputy_des
    delta_r_dot = v_deputy - v_deputy_des

    # Compute gravitational feedforeward terms
    f_d = grav_accel(r_deputy)
    f_dd = grav_accel(r_deputy_des)

    # PD control
    u = -(f_d - f_dd) - Kp * delta_r - Kd * delta_r_dot

    # Part 2: Add drag disturbance
    a_drag = -0.00001 * v_deputy/np.linalg.norm(v_deputy)  # simple drag model
    # u += a_drag

    # # Enforce saturation limit **TODO**
    # max_thrust = config.get("control", {}).get("max_thrust", None)  # in Newtons
    # if max_thrust is not None:
    #     mass = config.get("spacecraft", {}).get("deputy_mass", 1.0)  # in kg
    #     max_accel = max_thrust / mass  # in km/s^2 assuming inputs in km/s^2
    #     norm_u = np.linalg.norm(u)
    #     if norm_u > max_accel:
    #         u = (u / norm_u) * max_accel

    return {
        "accel_cmd": u.tolist()
    }
