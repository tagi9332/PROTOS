import numpy as np
from utils.six_dof_utils.rigid_body_kinematics.rigid_body_kinematics import EP2C, C2MRP


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