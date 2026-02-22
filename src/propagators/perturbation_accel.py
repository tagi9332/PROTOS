import numpy as np
from datetime import datetime

from data.resources.constants import MU_EARTH, R_EARTH, J2, P_SRP, OMEGA_EARTH
from src.io_utils.init_sim_config import PerturbationsConfig
from utils.perturbation_utils.atmospheric_density_exponential_model import rho_expo_model
from utils.perturbation_utils.sun_vector_ECI import get_sun_vector_eci
from utils.perturbation_utils.zonal_harmonics import compute_gravitational_harmonics

def compute_perturb_accel(r: np.ndarray, v: np.ndarray, perturb_config: PerturbationsConfig,
                          drag_properties: dict, mass: float, epoch: datetime):
    
    a_pert = np.zeros(3)

    # --- Zonal Harmonics ---
    a_pert += compute_gravitational_harmonics(r, perturb_config)

    # --- Drag Perturbation --- 
    if getattr(perturb_config, "drag", False):
        r_mag = np.linalg.norm(r) 
        alt = r_mag - R_EARTH      

        # Atmospheric rotation
        w_earth = np.array([0.0, 0.0, OMEGA_EARTH]) 
        v_atm_km_s = np.cross(w_earth, r)
        
        # Relative velocity of satellite to atmosphere
        v_rel = v - v_atm_km_s
        v_rel_m_s = v_rel * 1000.0
        v_rel_mag_m_s = np.linalg.norm(v_rel_m_s)

        if v_rel_mag_m_s > 0:
            Cd = drag_properties.get("Cd", 2.2)
            A_m = drag_properties.get("area", 1.0) 

            rho_kgm3 = rho_expo_model(alt)

            # Drag acceleration in m/s^2
            a_drag_m_s2 = -0.5 * Cd * A_m / mass * rho_kgm3 * v_rel_mag_m_s * v_rel_m_s

            # Convert to km/s^2 (FIXED: Divide by 1000, not 1000^2)
            a_drag = a_drag_m_s2 / 1000.0

            a_pert += a_drag

    # --- Solar Radiation Pressure --- 
    if getattr(perturb_config, "SRP", False):
        # 1. Setup properties
        Cr = 1.8 
        A_srp = drag_properties.get("area", 1.0) 
        r_sun = get_sun_vector_eci(epoch)
        r_sun_mag = np.linalg.norm(r_sun)

        # 2. Eclipse Model (Cylindrical Shadow)
        # Project satellite vector onto the Sun vector
        s_proj = np.dot(r, r_sun) / r_sun_mag
        
        # Distance from the center axis of the shadow
        d_perp = np.sqrt(np.linalg.norm(r)**2 - s_proj**2)
        
        # The satellite is in shadow if it's on the night side (s_proj < 0) 
        # AND it's within the Earth's radius laterally.
        in_shadow = (s_proj < 0) and (d_perp < R_EARTH)

        # 3. Apply SRP only if in sunlight
        if not in_shadow:
            # Force due to SRP (N) -> P_SRP should be in N/m^2
            F_srp = P_SRP * A_srp * Cr

            # Acceleration (m/s^2)
            a_srp_m_s2 = F_srp / mass

            # Convert to km/s^2
            a_srp_km_s2 = a_srp_m_s2 / 1000.0

            # Light travels FROM Sun TO Earth, so the push direction is -r_sun
            srp_dir = -r_sun / r_sun_mag

            a_pert += (a_srp_km_s2 * srp_dir)

    return a_pert