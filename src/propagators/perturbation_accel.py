import numpy as np
from .constants import MU_EARTH, R_EARTH, J2

def compute_perturb_accel(r: np.ndarray, perturb_config: dict):
    """
    Compute perturbation accelerations for a spacecraft at position r.
    
    Parameters
    ----------
    r : np.ndarray
        Position vector [km]
    perturb_config : dict
        Dictionary specifying which perturbations are active (e.g., {"J2": True})
    
    Returns
    -------
    a_pert : np.ndarray
        Perturbation acceleration vector [km/s^2]
    """
    a_pert = np.zeros(3)

    if perturb_config.get("J2", False):
        r_mag = np.linalg.norm(r)
        z2 = r[2] ** 2
        factor = 1.5 * J2 * MU_EARTH * R_EARTH**2 / r_mag**5
        a_pert[0] -= factor * (1 - 5 * z2 / r_mag**2) * r[0]
        a_pert[1] -= factor * (1 - 5 * z2 / r_mag**2) * r[1]
        a_pert[2] -= factor * (3 - 5 * z2 / r_mag**2) * r[2]

    # You can add more perturbations here (drag, SRP, third body, etc.)
    
    return a_pert
