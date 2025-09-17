import numpy as np

def inertial_to_rel_LVLH(rc, vc, rd, vd):
    """
    Map inertial chief/deputy states to LVLH frame relative position and velocity.

    Inputs:
        rc, vc : chief position/velocity (km, km/s) as np.array
        rd, vd : deputy position/velocity (km, km/s) as np.array
    
    Outputs:
        rho_H  : deputy relative position in LVLH frame (km)
        rhod_H : deputy relative velocity in LVLH frame (km/s)
    """
    # Constants
    mu = 398600.0  # km^3/s^2

    rc = np.array(rc)
    vc = np.array(vc)
    rd = np.array(rd)
    vd = np.array(vd)

    # LVLH frame unit vectors
    R_hat = rc / np.linalg.norm(rc)
    C_hat = np.cross(rc, vc)
    C_hat = C_hat / np.linalg.norm(C_hat)
    I_hat = np.cross(C_hat, R_hat)
    
    C_HN = np.vstack((R_hat, I_hat, C_hat))
    
    # Relative position and velocity in inertial
    dr = rd - rc
    dv = vd - vc

    # Angular velocity of LVLH frame
    omega = np.cross(rc, vc) / np.dot(rc, rc)

    # Transform to LVLH frame
    rho_H  = C_HN @ dr
    rhod_H = C_HN @ (dv - np.cross(omega, dr))
    
    return rho_H, rhod_H
