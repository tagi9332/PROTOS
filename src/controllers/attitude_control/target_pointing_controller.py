import numpy as np
from utils.six_dof_utils.get_errors import get_errors

def _get_target_quaternion(target_vec_i, boresight_vec_b=np.array([0, 0, 1])):
    """
    Computes the shortest-arc quaternion that rotates the boresight vector 
    to align with the target vector.
    
    Args:
        target_vec_i: Target unit vector in Inertial Frame.
        boresight_vec_b: Boresight unit vector in Body Frame.
        
    Returns:
        q_target: The [w, x, y, z] quaternion representing the desired orientation.
    """
    # Normalize inputs
    u = boresight_vec_b / np.linalg.norm(boresight_vec_b)
    v = target_vec_i / np.linalg.norm(target_vec_i)
    
    # Compute Dot Product
    dot_prod = np.dot(u, v)
    
    # Handle Singularity: Vectors are 180 degrees apart (Opposite)
    if dot_prod < -0.999999:
        orthogonal_axis = np.cross(np.array([1, 0, 0]), u)
        
        if np.linalg.norm(orthogonal_axis) < 0.1:
            orthogonal_axis = np.cross(np.array([0, 1, 0]), u)
            
        orthogonal_axis /= np.linalg.norm(orthogonal_axis)
        
        # 180 deg rotation (w=0, xyz=axis)
        return np.array([0.0, orthogonal_axis[0], orthogonal_axis[1], orthogonal_axis[2]])

    # Half-Way Quaternion Construction
    xyz = np.cross(u, v)
    
    # q_w = mag(u)*mag(v) + dot(u, v)
    w = 1.0 + dot_prod
    
    # Assemble and Normalize
    q_target = np.array([w, xyz[0], xyz[1], xyz[2]])
    q_target /= np.linalg.norm(q_target)
    
    return q_target

def target_pointing_controller(state: dict, config: dict, sat_name: str) -> dict:
    """
    Target Pointing Attitude Controller for a deputy satellite.
    """
    # Config
    guidance_dict = config.get('guidance', {})
    attitude_guidance = guidance_dict.get('attitude_guidance', {})
    target_type = attitude_guidance.get('attitude_reference', 'VELOCITY').upper()

    # State Extraction
    deputy_state = state["deputies"][sat_name]
    deputy_r = np.array(deputy_state["r"])
    deputy_v = np.array(deputy_state["v"])
    deputy_rho = np.array(deputy_state["rho"])

    q_BN = np.array(deputy_state["attitude"]["q_BN"])
    omega_BN = np.array(deputy_state["attitude"]["omega_BN"])

    # Determine Inertial Target Vector
    if target_type == "CHIEF":
        # Point at chief. Rho is (Deputy - Chief)
        target_vector = -deputy_rho
    elif target_type == "NADIR":
        # Point at Earth Center. Vector is -Position.
        target_vector = -deputy_r
    elif target_type == "VELOCITY":
        target_vector = deputy_v
    else:
        raise ValueError(f"Target pointing target '{target_type}' not recognized for {sat_name}.")

    # Normalize
    if np.linalg.norm(target_vector) > 1e-6:
        target_vector = target_vector / np.linalg.norm(target_vector)
    else:
        # Handle singularity
        target_vector = np.array([1, 0, 0])

    # Control Logic
    control_gains = config.get("control", {}).get("attitude_control_gains", {})
    Kp = control_gains.get("Kp", 1.0)
    Kd = control_gains.get("Kd",0.01)
    z_B = np.array(config.get("control", {}).get("body_z_axis", [0,0,1]))

    # Compute target quaternion
    q_target = _get_target_quaternion(target_vector, z_B)

    # Compute attitude and rate errors
    att_err_vec, omega_err = get_errors(q_BN, omega_BN, q_target, np.zeros(3))

    # Compute Torque (PD Controller)
    torque_cmd_deputy = -Kp * att_err_vec - Kd * omega_err
    
    # Not implemented for chief in this controller
    torque_cmd_chief = np.zeros(3) 

    return {
        "torque_chief": torque_cmd_chief,
        "torque_deputy": torque_cmd_deputy,
        "att_error_deputy": att_err_vec,   
        "rate_error_deputy": omega_err  
    }