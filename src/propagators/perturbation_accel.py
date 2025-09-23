import numpy as np
from data.resources.constants import MU_EARTH, R_EARTH, J2

def get_density(r: np.ndarray, v: np.ndarray):
    """
    Simple atmospheric density model using exponential model.
    """
    altitude_km = np.linalg.norm(r) - R_EARTH
    if altitude_km < 0:
        return 0

    # Exponential model parameters
    scale_height = 12  # km
    rho0 = 1.225  # kg/m^3 (sea level)

    # Compute density
    rho = rho0 * np.exp(-altitude_km / scale_height)
    return rho


def compute_perturb_accel(r: np.ndarray, v: np.ndarray, perturb_config: dict,
                          drag_properties: dict, mass: float, epoch: str):
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
            Cd = drag_properties.get("cd")
            A = drag_properties.get("area")
            # Compute density via simple exponential model (TODO: replace with NRLMSISE-00 model)
            rho = get_density(r, v)
            a_drag = -0.5 * rho * Cd * A / mass * v_mag * v
            a_pert += a_drag

    return a_pert
