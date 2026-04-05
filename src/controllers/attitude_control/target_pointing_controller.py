import numpy as np
from utils.six_dof_utils.get_errors import get_errors

def _rotate_vector_by_quat(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotates a 3D vector by a scalar-first quaternion [w, x, y, z].
    Assuming q is the q_BN quaternion, this maps an Inertial vector into the Body frame.
    """
    q_vec = q[1:4]
    q_scalar = q[0]
    
    # Standard quaternion vector rotation formula
    v_rot = v + 2.0 * np.cross(q_vec, np.cross(q_vec, v) + q_scalar * v)
    return v_rot

def _get_error_quaternion(target_vec_b: np.ndarray, boresight_vec_b: np.ndarray):
    """
    Computes the shortest-arc error quaternion to align the boresight 
    with the target vector, BOTH expressed in the Body frame.
    """
    u = boresight_vec_b / np.linalg.norm(boresight_vec_b)
    v = target_vec_b / np.linalg.norm(target_vec_b)
    
    dot_prod = np.dot(u, v)
    
    # Handle the 180-degree singularity safely
    if dot_prod < -0.999999:
        orthogonal_axis = np.cross(np.array([1, 0, 0]), u)
        if np.linalg.norm(orthogonal_axis) < 0.1:
            orthogonal_axis = np.cross(np.array([0, 1, 0]), u)
        
        orthogonal_axis /= np.linalg.norm(orthogonal_axis)
        return np.array([0.0, orthogonal_axis[0], orthogonal_axis[1], orthogonal_axis[2]])

    xyz = np.cross(u, v)
    w = 1.0 + dot_prod
    
    q_err = np.array([w, xyz[0], xyz[1], xyz[2]])
    q_err /= np.linalg.norm(q_err)
    
    return q_err

def target_pointing_controller(state: dict, config: dict, sat_name: str) -> dict:
    """
    Target Pointing Attitude Controller for a deputy satellite.
    """
    guidance_dict = config.get('guidance', {})
    attitude_guidance = guidance_dict.get('attitude_guidance', {})
    target_type = attitude_guidance.get('attitude_reference', 'VELOCITY').upper()

    deputy_state = state["deputies"][sat_name]
    deputy_r = np.array(deputy_state["r"])
    deputy_v = np.array(deputy_state["v"])
    deputy_rho = np.array(deputy_state["rho"])

    q_BN = np.array(deputy_state["q_BN"])
    omega_BN = np.array(deputy_state["omega_BN"])

    # 1. Determine Inertial Target Vector
    if target_type == "CHIEF":
        target_vector_i = -deputy_rho
    elif target_type == "NADIR":
        target_vector_i = -deputy_r
    elif target_type == "VELOCITY":
        target_vector_i = deputy_v
    else:
        raise ValueError(f"Target pointing target '{target_type}' not recognized.")

    if np.linalg.norm(target_vector_i) > 1e-6:
        target_vector_i = target_vector_i / np.linalg.norm(target_vector_i)
    else:
        target_vector_i = np.array([1, 0, 0])

    # 2. Map the Inertial target vector into the CURRENT Body Frame
    target_vector_b = _rotate_vector_by_quat(q_BN, target_vector_i)

    # 3. Compute the Error Quaternion entirely in the Body Frame
    z_B = np.array(config.get("control", {}).get("body_z_axis", [0, 0, 1]))
    q_err = _get_error_quaternion(target_vector_b, z_B)

    # 4. ENFORCE SHORTEST PATH (This stops the chattering!)
    if q_err[0] < 0:
        q_err = -q_err

    # For small errors, the vector component of the quaternion is approximately the axis-angle error
    att_err_vec = q_err[1:4] 
    
    # We want omega_err to drive the body rates to zero while pointing
    omega_err = omega_BN 

    # 5. Compute Torque (PD Controller)
    control_gains = config.get("control", {}).get("attitude_control_gains", {})
    Kp = control_gains.get("Kp", 1.0)
    Kd = control_gains.get("Kd", 0.01)

    torque_cmd_deputy = -Kp * att_err_vec - Kd * omega_err

    return {
            "torque_cmd": torque_cmd_deputy, 
            "att_error": att_err_vec,   
            "rate_error": omega_err  
        }