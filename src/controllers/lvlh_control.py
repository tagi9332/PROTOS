import numpy as np

def lvlh_step(state: dict, config: dict) -> dict:
    """
    GNC step implementing u = -(f(r_d) - f(r_dd)) - K1*Δr - K2*Δr_dot
    Returns command acceleration in inertial frame for dynamics propagation.
    """
    # Extract relative vectors
    deputy_rho = np.array(state["deputy_rho"])
    deputy_rho_dot = np.array(state["deputy_rho_dot"])

    # Guidance
    rpo = config.get("guidance", {}).get("rpo", {})
    desired = config.get("desired_relative_state", [0,0,0,0,0,0])
    deputy_rho_des = desired.get("deputy_rho_des", np.zeros(3))
    deputy_rho_dot_des = desired.get("deputy_rho_dot_des", np.zeros(3))

    # Get Kp and Kd from config (scalar or list), default to 1s
    Kp_val = config.get("control", {}).get("pd", {}).get("Kp", 1.0)
    Kd_val = config.get("control", {}).get("pd", {}).get("Kd", 1.0)

    # Ensure 1D arrays of length 3
    Kp_array = np.array([Kp_val]*3) if np.isscalar(Kp_val) else np.array(Kp_val).flatten()
    Kd_array = np.array([Kd_val]*3) if np.isscalar(Kd_val) else np.array(Kd_val).flatten()

    # Create diagonal matrices
    Kp = np.diag(Kp_array)
    Kd = np.diag(Kd_array)

    # PD control
    delta_r = deputy_rho - deputy_rho_des
    delta_r_dot = deputy_rho_dot - deputy_rho_dot_des
    u = -Kp @ delta_r - Kd @ delta_r_dot

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
