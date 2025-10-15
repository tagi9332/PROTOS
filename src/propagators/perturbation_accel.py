import numpy as np
from datetime import datetime
from data.resources.constants import MU_EARTH, R_EARTH, J2
from src.propagators.atmospheric_density_exponential_model import atmos_density_expm_model

def compute_perturb_accel(r: np.ndarray, v: np.ndarray, perturb_config: dict,
                          drag_properties: dict, mass: float, epoch: datetime):
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
    epoch : datetime, optional
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
        # Relative position and velocity (in meters)
        v_mag = np.linalg.norm(v)  # km/s
        r_mag = np.linalg.norm(r)  # km
        alt = r_mag - R_EARTH
        if v_mag > 0:
            Cd = drag_properties.get("cd")
            A = drag_properties.get("area")
            # Compute density via simple exponential model (TODO: replace with NRLMSISE-00 model)
            T, p, rho = atmos_density_expm_model(alt)
            a_drag = -0.5 * rho * Cd * A / mass * v_mag * 1000 * v * 1000  # type: ignore # m/s^2
            # Convert drag acceleration to km/s^2
            a_drag /= 1000
            a_pert += a_drag

    # --- Solar Radiation Pressure ---
    if perturb_config.get("SRP", False):
        # Solar constant at 1 AU in W/m^2
        S_sun = 1367.0
        # Speed of light in km/s
        c = 299792.458
        # Reflectivity coefficient (1.0 for perfect absorption, 2.0 for perfect reflection)
        Cr = 1.0
        # Effective cross-sectional area in m^2
        A_srp = drag_properties.get("area", 1.0)  # Assuming drag area as SRP area

        # Distance from the Sun in km (assuming r is the position vector in ECI)
        r_sun = np.linalg.norm(r)
        # Unit vector pointing from spacecraft to Sun
        r_hat_sun = -r / r_sun

        # Solar radiation pressure at spacecraft's distance
        P_srp = S_sun / c * (1 / r_sun**2)
        # Force due to SRP
        F_srp = P_srp * A_srp * Cr
        # Acceleration due to SRP
        a_srp = F_srp / mass
        # Acceleration vector
        a_pert += a_srp * r_hat_sun

    return a_pert
