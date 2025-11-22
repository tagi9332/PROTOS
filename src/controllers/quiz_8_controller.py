"""Quiz 8 controller implementing PD control with gravitational feedforward"""

import numpy as np
from utils.orbital_element_conversions.oe_conversions import inertial_to_oes
from utils.frame_conversions.rel_to_inertial_functions import LVLH_DCM

def quiz_8_step(state: dict, config: dict) -> dict:
    """
    GNC step implementing orbit element control law u = inv([B]) * -([A(oe_actual)] - [A(oe_desired)]) - [P]delta_oe
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
    # Convert chief and deputy current inertial to classical orbital elements
    a_c, e_c, _, _, _, ta_c = inertial_to_oes(r_chief, v_chief)
    a_d, e_d, _, _, _, ta_d = inertial_to_oes(r_deputy, v_deputy)

    # Compute deputy current orbit element differences
    d_oe_actual = np.array([a_d - a_c,
                            e_d - e_c])                       

    # Desired differential orbital elements to control to
    del_a_desired = 0
    del_e_desired = 0.0005
    del_oe_desired = np.array([del_a_desired, del_e_desired])

    # Define [B] matrix for chosen orbit
    def B_matrix(a_d, e_d, TA):
        from data.resources.constants import MU_EARTH as mu_e
          
        # Compute parameters
        p = a_d * (1 - e_d**2)
        h = np.sqrt(mu_e * p)  # Earth's mu in km^3/s^2
        r = p / (1 + e_d * np.cos(TA))

        # Only controlling a and e, inputs for radial and along-track
        B = np.array([
            [2*a_d**2*e_d*np.sin(TA)/(h), 2*a_d**2*p/(h*r), 0],
            [p*np.sin(TA)/h, ((p+r)*np.cos(TA)+r*e_d)/h, 0],
        ])

        return B
    
    B = B_matrix(a_d, e_d, ta_d)

    # Compute time-variant gains
    P_11 = 10**(-12) + 0.001*np.sin(ta_c)**(10)
    Kp = np.diag([P_11, P_11])

    # Compute error terms
    delta_oe = d_oe_actual - del_oe_desired

    # PD control
    u = np.linalg.pinv(B) @ (-Kp @ delta_oe)

    # Compute DCM from LVLH to inertial
    C_H_N = LVLH_DCM(r_deputy, v_deputy)

    # Transform command acceleration to inertial frame
    u = C_H_N.T @ u  # from LVLH to inertial

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
