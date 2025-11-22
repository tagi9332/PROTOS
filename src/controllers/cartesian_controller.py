import numpy as np
from utils.frame_conversions.rel_to_inertial_functions \
    import LVLH_DCM, rel_vector_to_inertial, inertial_to_rel_LVLH
from utils.orbital_dynamics.orbital_accel import grav_accel
from utils.orbital_element_conversions.oe_conversions import lroes_to_inertial

def _compute_lvlh_kinematics(r_c: np.ndarray, v_c: np.ndarray):
    """
    Computes LVLH (Hill frame) angular velocity and angular acceleration
    expressed in inertial coordinates.

    Parameters
    ----------
    r_c : ndarray (3,)
        Chief satellite inertial position vector.
    v_c : ndarray (3,)
        Chief satellite inertial velocity vector.

    Returns
    -------
    omega : ndarray (3,)
        LVLH frame angular velocity (rad/s) in inertial coordinates.
    omega_dot : ndarray (3,)
        Time derivative of omega (rad/s^2) in inertial coordinates.
    """

    r_norm = np.linalg.norm(r_c)

    # Specific angular momentum
    h = np.cross(r_c, v_c)

    # LVLH angular velocity: ω = h / r^2
    omega = h / (r_norm ** 2)

    # Radial velocity term ṙ = r ⋅ v / |r|
    rdot = np.dot(r_c, v_c) / r_norm

    # Time derivative: ω̇ = -2 * ṙ / r^3 * h
    omega_dot = (-2 * rdot / (r_norm ** 3)) * h

    return omega, omega_dot

def _compute_feedforward(r_c, v_c, r_d_des, rho_des):

    """
    Compute full LVLH feedforward acceleration term:
        u_d = f(r_c) - f(r_d_des) + rho_ddot_I

    Parameters
    ----------
    r_c : ndarray (3,)
        Chief inertial position.
    v_c : ndarray (3,)
        Chief inertial velocity.
    r_d_des : ndarray (3,)
        Desired deputy inertial position.
    rho_des : ndarray (3,)
        Desired LVLH relative position (in LVLH frame).
    grav_accel : callable
        Function computing gravitational acceleration at a position.
    LVLH_DCM : callable
        Function computing LVLH-to-inertial DCM.

    Returns
    -------
    u_d : ndarray (3,)
        Feedforward inertial acceleration
    """
    # LVLH kinematics: ω and ω̇
    omega, omegadot = _compute_lvlh_kinematics(r_c, v_c)

    # LVLH → inertial DCM
    C_HN = LVLH_DCM(r_c, v_c)

    # Desired relative position in inertial frame
    rho_I_des = C_HN.T @ rho_des

    # Relative acceleration in inertial
    rho_ddot_I = np.cross(omegadot, rho_I_des) + np.cross(omega, np.cross(omega, rho_I_des))

    # Feedforward term
    return grav_accel(r_c) - grav_accel(r_d_des) + rho_ddot_I

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
        raise ValueError("Desired state must be specified in LROES or LVLH frame.")

    return rho_des, rho_dot_des

def cartesian_step(state: dict, config: dict) -> dict:
    """
    Cartesian GNC step implementing:
       u = -(f(r_d) - f(r_dd)) - K1*Δr - K2*Δr_dot + u_d
    Produces deputy inertial acceleration command.
    """

    # --- Extract states ---
    r_c = np.array(state["chief_r"])
    v_c = np.array(state["chief_v"])
    r_d = np.array(state["deputy_r"])
    v_d = np.array(state["deputy_v"])

    # --- Guidance: desired LVLH relative state ---
    rho_des, rho_dot_des = _get_desired_state_LVLH(
        config.get("guidance", {}).get("rpo", {}).get("frame", "LVLH"),
        config.get("guidance", {}).get("rpo", {}).get("deputy_desired_relative_state"),
        r_c,
        v_c,
        state.get("sim_time", 0.0)
    )

    # Convert desired LVLH → inertial deputy state
    r_d_des, v_d_des = rel_vector_to_inertial(rho_des, rho_dot_des, r_c, v_c)

    # --- Control gains ---
    kp = config.get("control", {}).get("gains", {}).get("K1", 1.0)
    kd = config.get("control", {}).get("gains", {}).get("K2", 1.0)

    # --- Error terms ---
    error_r = r_d - r_d_des
    error_v = v_d - v_d_des

    # --- Final control law ---
    u = -(grav_accel(r_d) - grav_accel(r_d_des)) \
        - kp * error_r - kd * error_v \
        + _compute_feedforward(r_c, v_c, r_d_des, rho_des)

    # Return updated dictionary
    return {
        "status": "lvlh",
        "chief_r": r_c.tolist(),
        "chief_v": v_c.tolist(),
        "deputy_r": r_d.tolist(),
        "deputy_v": v_d.tolist(),
        "deputy_rho": np.array(state["deputy_rho"]).tolist(),
        "deputy_rho_dot": np.array(state["deputy_rho_dot"]).tolist(),
        "accel_cmd": u.tolist()
    }
