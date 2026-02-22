import numpy as np
from data.resources.constants import MU_EARTH, R_EARTH, J2, J3, J4
from src.io_utils.init_sim_config import PerturbationsConfig

def compute_gravitational_harmonics(r: np.ndarray, perturb_config: PerturbationsConfig) -> np.ndarray:
    """
    Computes the gravitational perturbations due to Earth's zonal harmonics (J2, J3, J4).
    """
    a_harmonics = np.zeros(3)
    r_mag = np.linalg.norm(r)
    
    # Pre-compute common terms to save cycles
    r2 = r_mag**2
    z = r[2]
    z2 = z**2
    
    # --- J2 Perturbation (Oblateness) ---
    if getattr(perturb_config, "J2", False):
        factor2 = 1.5 * J2 * MU_EARTH * R_EARTH**2 / r_mag**5
        
        a_harmonics[0] -= factor2 * (1 - 5 * z2 / r2) * r[0]
        a_harmonics[1] -= factor2 * (1 - 5 * z2 / r2) * r[1]
        a_harmonics[2] -= factor2 * (3 - 5 * z2 / r2) * z

    # --- J3 Perturbation (Pear Shape) ---
    if getattr(perturb_config, "J3", False):
        z4 = z2**2
        factor3 = 2.5 * J3 * MU_EARTH * R_EARTH**3 / r_mag**7
        
        a_harmonics[0] += factor3 * r[0] * z * (7 * z2 / r2 - 3)
        a_harmonics[1] += factor3 * r[1] * z * (7 * z2 / r2 - 3)
        a_harmonics[2] += factor3 * (7 * z4 / r2 - 6 * z2 + 0.6 * r2)

    # --- J4 Perturbation ---
    if getattr(perturb_config, "J4", False):
        r4 = r2**2
        z4 = z2**2
        factor4 = 0.625 * J4 * MU_EARTH * R_EARTH**4 / r_mag**7
        
        a_harmonics[0] += factor4 * r[0] * (63 * z4 / r4 - 42 * z2 / r2 + 3)
        a_harmonics[1] += factor4 * r[1] * (63 * z4 / r4 - 42 * z2 / r2 + 3)
        a_harmonics[2] += factor4 * z * (63 * z4 / r4 - 70 * z2 / r2 + 15)

    return a_harmonics
