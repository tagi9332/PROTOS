import numpy as np

# Compute DCM from LVLH to Inertial
def LVLH_to_inertial_dcm(r, v):
    r = np.array(r)
    v = np.array(v)
    R_hat = r / np.linalg.norm(r)
    C_hat = np.cross(r, v)
    C_hat = C_hat / np.linalg.norm(C_hat)
    I_hat = np.cross(C_hat, R_hat)
    C_NH = np.vstack((R_hat, I_hat, C_hat)).T
    return C_NH