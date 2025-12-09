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