import numpy as np
from datetime import datetime
from data.resources.constants import MU_EARTH, R_EARTH, J2, P_SRP
from utils.perturbation_utils.atmospheric_density_exponential_model import rho_expo_model
from utils.perturbation_utils.sun_vector_ECI import get_sun_vector_eci

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
        UTC epoch

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
        # Altitude
        r_mag = np.linalg.norm(r)  # km
        alt = r_mag - R_EARTH       # km

        # Velocity in m/s
        v_m_s = v * 1000.0
        v_mag_m_s = np.linalg.norm(v_m_s)

        if v_mag_m_s > 0:
            Cd = drag_properties.get("cd")
            A_m = drag_properties.get("area")  # m^2

            # Density in kg/m^3
            rho_kgm3 = rho_expo_model(alt)

            # Drag acceleration in m/s^2
            a_drag_m_s2 = -0.5 * Cd * A_m / mass * rho_kgm3 * v_mag_m_s * v_m_s # type: ignore

            # Convert to km/s^2
            a_drag = a_drag_m_s2 / (1000.0**2)

            a_pert += a_drag

    # --- Solar Radiation Pressure ---
    if perturb_config.get("SRP", False):
        # Solar constant at 1 AU in W/km^2
        S_sun = 1.3670

        # Speed of light in km/s
        c = 299792.458

        # Reflectivity coefficient (1.0 for perfect absorption, 2.0 for perfect reflection)
        Cr = 1.8

        # Effective cross-sectional area in m^2
        A_srp = drag_properties.get("area", 1.0)  # Assuming drag area as SRP area

        # Sun vector in ECI frame (km)
        r_sun = get_sun_vector_eci(epoch)

        # Sun unit vector
        r_hat_sun = r_sun / np.linalg.norm(r_sun)

        # Force due to SRP (N)
        F_srp = P_SRP * A_srp * Cr

        # Acceleration due to SRP (m/s^2)
        a_srp = F_srp / mass

        # Convert to km/s^2
        a_srp /= 1000.0

        # Vector from sun to spacecraft
        r_sun_sc = r - r_sun
        r_sun_sc_hat = r_sun_sc / np.linalg.norm(r_sun_sc)

        a_srp = a_srp * r_sun_sc_hat  # Direction away from the Sun

        # Acceleration vector
        a_pert += a_srp

    return a_pert
