import numpy as np


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

def C2MRP(C):
    """
    C2MRP
        Translates the 3x3 direction cosine matrix C into the 
        corresponding 3x1 MRP vector sigma.
        
        Enforces the constraint |sigma| <= 1 (Shadow Set switching)
        by ensuring the quaternion scalar is positive.
    """
    # Convert to Quaternion (Euler Parameters)
    b = C2EP(C)
    
    # Ensure it is a flat numpy array (not a matrix object)
    b = np.array(b).flatten()

    # Enforce Shortest Rotation (Shadow Set Check)
    # The MRP magnitude is > 1 if the scalar part (b[0]) is negative.
    # Since q and -q represent the same rotation, flip b if b[0] < 0.
    if b[0] < 0:
        b = -b

    # Calculate MRP
    # Formula: sigma = vector_part / (1 + scalar_part)
    denominator = 1.0 + b[0]
    
    # Safety check for numerical singularity (should be impossible if b[0] >= 0)
    if denominator < 1e-6:
        return np.zeros(3)

    sigma = np.zeros(3)
    sigma[0] = b[1] / denominator
    sigma[1] = b[2] / denominator
    sigma[2] = b[3] / denominator

    return sigma


def get_target_quaternion(target_vec_i, boresight_vec_b=np.array([0, 0, 1])):
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

def get_errors(q_BN, omega_BN, q_target, omega_target_i=np.zeros(3)):
    """
    Computes MRP attitude error and rate error.
    """
    # Compute DCMs
    C_BN = EP2C(q_BN)       # Body -> Inertial
    C_RN = EP2C(q_target)   # Reference(Target) -> Inertial
    
    # Compute Relative Error Matrix (Target -> Body)
    C_err = C_BN.T @ C_RN
    
    # 3. Compute MRPs (Target -> Body)
    sigma = C2MRP(C_err)

    # 5. Rate Error
    # Map inertial target rates into body frame
    omega_target_b = C_BN.T @ omega_target_i
    omega_error_b = omega_BN - omega_target_b

    return sigma, omega_error_b


def target_pointing_controller(state: dict, config: dict) -> dict:
    """
    Target Pointing Attitude Controller for the deputy.
    """
    # Config & State Extraction
    target_type = config.get("guidance", {}).get("attitude_guidance", {}).get("attitude_reference", "VELOCITY").upper()
    deputy_r = np.array(state.get("deputy_r", [0,0,0]))
    deputy_v = np.array(state.get("deputy_v", [0,0,0]))
    deputy_rho = np.array(state.get("deputy_rho", [0,0,0]))
    
    q_BN = np.array(state.get("deputy_q_BN", [1,0,0,0]))
    omega_BN = np.array(state.get("deputy_omega_BN", [0,0,0])) # Actual Rate

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
        raise ValueError(f"Target pointing target '{target_type}' not recognized.")

    # Normalize
    if np.linalg.norm(target_vector) > 1e-6:
        target_vector = target_vector / np.linalg.norm(target_vector)
    else:
        # Handle singularity (e.g. at origin)
        target_vector = np.array([1, 0, 0])

    # Control Logic
    Kp = config.get("control", {}).get("attitude_control_gains", {}).get("Kp", 0.1)
    Kd = config.get("control", {}).get("attitude_control_gains", {}).get("Kd", 0.01)
    z_B = np.array(config.get("control", {}).get("body_z_axis", [0,0,1]))

    # Compute target quaternion
    q_target = get_target_quaternion(target_vector, z_B)

    # Compute attitude and rate errors
    att_err_vec, omega_err = get_errors(q_BN, omega_BN, q_target, np.zeros(3))

    # Compute Torque (PD Controller)
    torque_cmd_deputy = -Kp * att_err_vec - Kd * omega_err
    
    # Not implemented for chief in this controller
    torque_cmd_chief = np.zeros(3) 

    return {
        "torque_chief": torque_cmd_chief,
        "torque_deputy": torque_cmd_deputy,
        "att_error_vec": att_err_vec,   
        "rate_error_deputy": omega_err  
    }