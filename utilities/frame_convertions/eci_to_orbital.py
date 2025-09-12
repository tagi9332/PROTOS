import numpy as np

def eci_to_orbital(r_eci, v_eci):
    """
    Convert a satellite state from ECI frame to the orbital (LVLH/RIC) frame.

    Parameters
    ----------
    r_eci : np.ndarray
        Satellite position vector in ECI frame [km], shape (3,)
    v_eci : np.ndarray
        Satellite velocity vector in ECI frame [km/s], shape (3,)

    Returns
    -------
    r_orb : np.ndarray
        Satellite position in orbital frame [km], shape (3,)
    v_orb : np.ndarray
        Satellite velocity in orbital frame [km/s], shape (3,)
    R_matrix : np.ndarray
        3x3 rotation matrix from ECI to orbital frame
    """
    r_eci = np.array(r_eci, dtype=float)
    v_eci = np.array(v_eci, dtype=float)

    # Radial unit vector (points from Earth to satellite)
    R_hat = r_eci / np.linalg.norm(r_eci)

    # Cross-track unit vector (orbital angular momentum direction)
    h_vec = np.cross(r_eci, v_eci)
    C_hat = h_vec / np.linalg.norm(h_vec)

    # In-track unit vector (completes right-hand system)
    I_hat = np.cross(C_hat, R_hat)

    # Rotation matrix from ECI to orbital frame
    R_matrix = np.vstack((R_hat, I_hat, C_hat)).T  # columns are the orbital axes

    # Position and velocity in orbital frame
    r_orb = R_matrix.T @ r_eci
    v_orb = R_matrix.T @ v_eci

    return r_orb, v_orb, R_matrix
