import numpy as np
from .constants import MU_EARTH, R_EARTH, J2
from pyatmos import expo

# Create a global Atmosphere instance (fast to reuse)
atm_model = expo()

def get_density(r: np.ndarray, epoch=None) -> float:
    """
    Get atmospheric density at a given ECI position using pyatmos.
    Simple approximation: only depends on altitude.
    
    Parameters
    ----------
    r : np.ndarray
        ECI position vector [km]
    epoch : datetime.datetime, optional
        Not used for pyatmos but kept for compatibility
    
    Returns
    -------
    rho : float
        Atmospheric density [kg/km^3]
    """
    r_mag = np.linalg.norm(r)
    alt_km = r_mag - R_EARTH
    # pyatmos returns kg/m^3, convert to kg/km^3
    rho = atm_model.density(alt_km) * 1e9
    return rho


def compute_perturb_accel(r: np.ndarray, v: np.ndarray, perturb_config: dict,
                          drag_properties: dict, mass: float, epoch=None):
    """
    Compute perturbation accelerations for a spacecraft at position r.

    Parameters
    ----------
    r : np.ndarray
        ECI position vector [km]
    v : np.ndarray
        Velocity vector [km/s]
    perturb_config : dict
        Dictionary specifying active perturbations
    drag_properties : dict
        {"area": float, "cd": float}
    mass : float
        Spacecraft mass [kg]
    epoch : datetime.datetime, optional
        UTC epoch (not needed for pyatmos, kept for compatibility)

    Returns
    -------
    a_pert : np.ndarray
        Perturbation acceleration vector [km/s^2]
    """
    a_pert = np.zeros(3)

    # --- J2 perturbation ---
    if perturb_config.get("J2", False):
        r_mag = np.linalg.norm(r)
        z2 = r[2] ** 2
        factor = 1.5 * J2 * MU_EARTH * R_EARTH**2 / r_mag**5
        a_pert[0] -= factor * (1 - 5 * z2 / r_mag**2) * r[0]
        a_pert[1] -= factor * (1 - 5 * z2 / r_mag**2) * r[1]
        a_pert[2] -= factor * (3 - 5 * z2 / r_mag**2) * r[2]

    # --- Drag perturbation ---
    if perturb_config.get("drag", False):
        v_mag = np.linalg.norm(v)
        if v_mag > 0:
            Cd = drag_properties.get("cd", 2.2)
            A = drag_properties.get("area", 1.0)
            # Compute density via pyatmos
            rho = get_density(r, epoch)
            a_drag = -0.5 * rho * Cd * A / mass * v_mag * v
            a_pert += a_drag

    return a_pert
