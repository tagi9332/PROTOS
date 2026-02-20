import numpy as np
from scipy.integrate import solve_ivp

from src.propagators.perturbation_accel import compute_perturb_accel
from utils.frame_conversions.rel_to_inertial_functions import rel_vector_to_inertial, LVLH_DCM
from utils.orbital_element_conversions.oe_conversions import inertial_to_oes, m_to_ta, ta_to_m
from data.resources.constants import MU_EARTH,J2000_EPOCH

def _th_derivs(nu, y, e, p, h, u_total_t):
    """
    Inner derivative function for TH equations.
    """
    x, y_pos, z = y[:3]
    xp, yp, zp = y[3:]
    
    # Local orbit properties
    rho_local = 1 + e * np.cos(nu) 
    r_local = p / rho_local
    nu_dot_local = h / r_local**2
    
    # Scale constant Time-Domain force to Anomaly-Domain
    u_accel_nu = u_total_t / (nu_dot_local**2)
    
    # TH Equations of Motion
    x_pp = 2 * yp + (3.0 / rho_local) * x + u_accel_nu[0]
    y_pp = -2 * xp + u_accel_nu[1]
    z_pp = -z + u_accel_nu[2]
    
    return np.array([xp, yp, zp, x_pp, y_pp, z_pp])

def step_th(state: dict, dt: float, config: dict):
    """
    Tschauner-Hempel (TH) Propagator Step with Differential Perturbations.
    """
    
    # Unpack State & Config
    chief_r = state["chief_r"]
    chief_v = state["chief_v"]
    rho = state["deputy_rho"]
    rho_dot = state["deputy_rho_dot"]
    epoch = state.get("epoch", J2000_EPOCH)
    u_ctrl_t = state.get("control_accel", np.zeros(3))

    sim_config = config.get("simulation", {}).get("perturbations", {})
    sat_props = config.get("satellite_properties", {})

    # Compute differential perturbations if enabled
    if sim_config.get("enable_perturbations", False):
        # Helper to get properties
        def get_props(key):
            props = sat_props.get(key, {})
            # Default values if not provided
            mass = config.get("satellites", {}).get(key, {}).get("mass", 
                        500.0 if key == "deputy" else 250.0)
            return mass, {"Cd": props.get("Cd", 2.2), "area": props.get("area", 1.0)}

        c_mass, c_drag = get_props("chief")
        d_mass, d_drag = get_props("deputy")

        # Reconstruct deputy inertial state for perturbation calc
        deputy_r, deputy_v = rel_vector_to_inertial(rho, rho_dot, chief_r, chief_v)
        
        a_pert_chief = compute_perturb_accel(chief_r, chief_v, sim_config, c_drag, c_mass, epoch)
        a_pert_deputy = compute_perturb_accel(deputy_r, deputy_v, sim_config, d_drag, d_mass, epoch)

        # Calculate differential force in LVLH
        a_diff_eci = a_pert_deputy - a_pert_chief
        C_HN = LVLH_DCM(chief_r, chief_v)
        u_total_t = u_ctrl_t + (C_HN @ a_diff_eci)
    else:
        u_total_t = u_ctrl_t

    # Orbit Geometry
    a_semi, e, _, _, _, nu_0 = inertial_to_oes(chief_r, chief_v, units='rad')
    
    # Derive derived constants
    p = a_semi * (1 - e**2)
    n_mean = np.sqrt(MU_EARTH / a_semi**3)
    
    # Calculate Mean Anomaly propogation
    M0 = ta_to_m(nu_0, e)
    M_target = M0 + n_mean * dt
    
    # Solve for final True Anomaly
    nu_final = m_to_ta(M_target, e)
    
    # Handle wrapping for integration span
    if nu_final < nu_0:
        nu_final += 2 * np.pi

    # Integration (Anomaly Domain)
    r_mag = np.linalg.norm(chief_r)
    h_val = np.sqrt(MU_EARTH * p)
    nu_dot_0 = h_val / r_mag**2 
    
    rho_prime_0 = rho_dot / nu_dot_0
    y_curr = np.hstack((rho, rho_prime_0))

    nu_span = (nu_0, nu_final)

    # We use args=() to pass the extra parameters into _th_derivs
    sol = solve_ivp(
        _th_derivs, 
        nu_span, 
        y_curr, 
        method='RK45', 
        args=(e, p, h_val, u_total_t),
        rtol=1e-12, 
        atol=1e-12
    )
    y_final = sol.y[:, -1]

    # Reconstruction (Anomaly -> Time)
    rho_final = y_final[:3]
    rho_prime_final = y_final[3:]
    
    r_final_mag = p / (1 + e * np.cos(nu_final))
    nu_dot_final = h_val / r_final_mag**2
    rho_dot_final = rho_prime_final * nu_dot_final
    
    # Re-deriving dE explicitly since utils return M or Nu:
    def get_E(nu, ecc):
        return 2.0 * np.arctan2(np.sqrt(1.0 - ecc) * np.sin(nu/2.0),
                                np.sqrt(1.0 + ecc) * np.cos(nu/2.0))
    
    E0 = get_E(nu_0, e)
    E_final = get_E(nu_final, e)
    # Unwrap E if nu wrapped
    if E_final < E0: E_final += 2*np.pi
    dE_total = E_final - E0

    # Gauss F & G functions
    sin_dE = np.sin(dE_total)
    cos_dE = np.cos(dE_total)
    
    f = 1.0 + (a_semi / r_mag) * (cos_dE - 1.0)
    g = r_mag * np.sqrt(a_semi / MU_EARTH) * sin_dE + (np.dot(chief_r, chief_v) / MU_EARTH) * a_semi * (1.0 - cos_dE)
    f_dot = -(np.sqrt(MU_EARTH * a_semi) / (r_mag * r_final_mag)) * sin_dE
    g_dot = 1.0 + (a_semi / r_final_mag) * (cos_dE - 1.0)
    
    chief_r_next = f * chief_r + g * chief_v
    chief_v_next = f_dot * chief_r + g_dot * chief_v

    # Reconstruct Final Deputy ECI
    deputy_r_next, deputy_v_next = rel_vector_to_inertial(rho_final, rho_dot_final, chief_r_next, chief_v_next)

    return {
        "chief_r": chief_r_next,
        "chief_v": chief_v_next,
        "deputy_r": deputy_r_next, 
        "deputy_v": deputy_v_next,
        "deputy_rho": rho_final,
        "deputy_rho_dot": rho_dot_final
    }