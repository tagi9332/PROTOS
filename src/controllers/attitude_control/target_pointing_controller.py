import numpy as np


# --- robust conversions ----------------------------------------------------
def EP2C(q):
    """Euler params (q0,q1,q2,q3) -> DCM (body -> inertial)"""
    q0, q1, q2, q3 = q
    C = np.zeros((3,3))
    C[0,0] = q0*q0 + q1*q1 - q2*q2 - q3*q3
    C[0,1] = 2*(q1*q2 + q0*q3)
    C[0,2] = 2*(q1*q3 - q0*q2)
    C[1,0] = 2*(q1*q2 - q0*q3)
    C[1,1] = q0*q0 - q1*q1 + q2*q2 - q3*q3
    C[1,2] = 2*(q2*q3 + q0*q1)
    C[2,0] = 2*(q1*q3 + q0*q2)
    C[2,1] = 2*(q2*q3 - q0*q1)
    C[2,2] = q0*q0 - q1*q1 - q2*q2 + q3*q3
    return C

def C2EP(C):
    """Robust DCM -> quaternion (q0, q1, q2, q3). Handles small q0 cases."""
    tr = np.trace(C)
    q = np.zeros(4)
    if tr > 1e-8:
        q0 = 0.5 * np.sqrt(1.0 + tr)
        q[0] = q0
        q[1] = (C[2,1] - C[1,2]) / (4.0*q0)
        q[2] = (C[0,2] - C[2,0]) / (4.0*q0)
        q[3] = (C[1,0] - C[0,1]) / (4.0*q0)
    else:
        # If trace is small or negative, pick largest diagonal element for stable extraction
        diag = np.array([C[0,0], C[1,1], C[2,2]])
        i = np.argmax(diag)
        if i == 0:
            q1 = 0.5 * np.sqrt(1.0 + C[0,0] - C[1,1] - C[2,2])
            q[1] = q1
            q[0] = (C[2,1] - C[1,2]) / (4.0*q1)
            q[2] = (C[0,1] + C[1,0]) / (4.0*q1)
            q[3] = (C[0,2] + C[2,0]) / (4.0*q1)
        elif i == 1:
            q2 = 0.5 * np.sqrt(1.0 - C[0,0] + C[1,1] - C[2,2])
            q[2] = q2
            q[0] = (C[0,2] - C[2,0]) / (4.0*q2)
            q[1] = (C[0,1] + C[1,0]) / (4.0*q2)
            q[3] = (C[1,2] + C[2,1]) / (4.0*q2)
        else:
            q3 = 0.5 * np.sqrt(1.0 - C[0,0] - C[1,1] + C[2,2])
            q[3] = q3
            q[0] = (C[1,0] - C[0,1]) / (4.0*q3)
            q[1] = (C[0,2] + C[2,0]) / (4.0*q3)
            q[2] = (C[1,2] + C[2,1]) / (4.0*q3)

    # Normalize quaternion
    q = q / np.linalg.norm(q)
    return q

# --- error computation -----------------------------------------------------
def get_errors(q_BN, omega_BN, target_vector, omega_target=np.zeros(3)):
    """
    Compute attitude quaternion error and rate error.
    - q_BN: current body->inertial quaternion (q0,q1,q2,q3)
    - omega_BN: angular rate of body w.r.t. inertial, expressed in body frame
    - target_vector: desired pointing vector expressed in inertial frame (e.g., vector from deputy to target)
    - omega_target: desired angular rate expressed in inertial frame (optional)
    Returns:
    - q_error: quaternion that rotates current attitude into desired attitude (q_err such that C_err = C_des.T @ C_BN)
               i.e., a small q_error vector ~ [0, e/2] for small angle error.
    - omega_error: body-frame rate error (omega_BN - C_BN @ omega_target)
    """
    # normalize target_vector and guard
    tv = np.array(target_vector, dtype=float)
    norm_tv = np.linalg.norm(tv)
    if norm_tv < 1e-12:
        raise ValueError("target_vector has zero length")
    b3_des = (tv / norm_tv)  # choose body +z to point along target (change sign if needed)

    # choose a reference axis that's not collinear with b3_des
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, b3_des)) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])

    # build desired body axes (in inertial frame)
    b2_des = np.cross(b3_des, ref)
    b2_norm = np.linalg.norm(b2_des)
    if b2_norm < 1e-12:
        # fallback if still degenerate
        ref = np.array([0.0, 1.0, 0.0])
        b2_des = np.cross(b3_des, ref)
        b2_norm = np.linalg.norm(b2_des)
        if b2_norm < 1e-12:
            raise RuntimeError("Unable to construct desired frame (degenerate target).")

    b2_des /= b2_norm
    b1_des = np.cross(b2_des, b3_des)
    b1_des /= np.linalg.norm(b1_des)

    # form desired DCM: columns are body axes expressed in inertial frame
    C_des = np.column_stack((b1_des, b2_des, b3_des))  # this is C_BN_des (body -> inertial)

    # current DCM from quaternion
    C_BN = EP2C(q_BN)

    # error DCM that rotates current body frame into desired body frame:
    C_err = C_des.T @ C_BN

    # quaternion error (rotation that takes current -> desired)
    q_err = C2EP(C_err)

    # rate error: express desired rate in body frame and subtract
    omega_des_body = C_BN.T @ omega_target  # omega_target is in inertial; map to body frame
    omega_err = omega_BN - omega_des_body

    return q_err, omega_err


def target_pointing_controller(state: dict, config: dict) -> dict:
    """
    Target Pointing Attitude Controller for the deputy.
    Computes torque commands to point the deputy's body z-axis towards a target.
    """
    target_type = config['guidance']['attitude_guidance']['attitude_reference'].upper()
    deputy_r = np.array(state.get("deputy_r", [0,0,0]))
    deputy_v = np.array(state.get("deputy_v", [0,0,0]))
    chief_r = np.array(state.get("chief_r", [0,0,0]))
    sun_vector = np.array(state.get("sun_vector", [1,0,0]))  # Placeholder

    # Determine target vector
    if target_type == "CHIEF":
        target_vector = chief_r - deputy_r
    elif target_type == "SUN":
        target_vector = sun_vector - deputy_r
    elif target_type == "NADIR":
        target_vector = -deputy_r
    elif target_type == "VELOCITY":
        target_vector = deputy_v
    else:
        raise ValueError(f"Target pointing target '{target_type}' not recognized.")

    target_vector /= np.linalg.norm(target_vector)

    # Gains
    Kp = config.get("control", {}).get("attitude_control_gains", {}).get("Kp", 0.1)
    Kd = config.get("control", {}).get("attitude_control_gains", {}).get("Kd", 0.01)

    # Current attitude & angular velocity
    q_BN = np.array(state.get("deputy_q_BN", [1,0,0,0]))
    omega_BN = np.array(state.get("deputy_omega_BN", [0,0,0]))

    # Boresight vector in body frame
    z_B = np.array(config.get("control", {}).get("body_z_axis", [0,0,1]))
    # z_B /= np.linalg.norm(z_B)    

    # Compute attitude and rate errors
    error_quat, omega_BN = get_errors(q_BN, omega_BN, target_vector, np.zeros(3))

    # PD torque command
    torque_cmd_deputy = -Kp * error_quat[1:] - Kd * omega_BN
    torque_cmd_chief = np.zeros(3)  # No control for chief

    return {
        "torque_chief": torque_cmd_chief,
        "torque_deputy": torque_cmd_deputy,
        "quat_error_deputy": error_quat, 
        " rate_error_deputy": omega_BN
    }
