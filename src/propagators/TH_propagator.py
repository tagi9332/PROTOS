import numpy as np
from src.propagators.perturbation_accel import compute_perturb_accel
from utils.frame_conversions.rel_to_inertial_functions import rel_vector_to_inertial, LVLH_DCM
from data.resources.constants import MU_EARTH

def _th_derivs(nu, y, e, p, h, u_total_t):
    """
    Lightweight inner derivative function for TH equations.
    """
    x, y_pos, z = y[:3]
    xp, yp, zp = y[3:]
    
    # Local orbit properties
    rho_local = 1 + e * np.cos(nu) 
    r_local = p / rho_local
    nu_dot_local = h / r_local**2
    
    # Scale constant Time-Domain force to Anomaly-Domain
    # u_nu = u_t / nu_dot^2
    u_accel_nu = u_total_t / (nu_dot_local**2)
    
    # TH Equations of Motion
    x_pp = 2 * yp + (3.0 / rho_local) * x + u_accel_nu[0]
    y_pp = -2 * xp + u_accel_nu[1]
    z_pp = -z + u_accel_nu[2]
    
    return np.array([xp, yp, zp, x_pp, y_pp, z_pp])

def step_th(state: dict, dt: float, config: dict):
    """
    Tschauner-Hempel (TH) Propagator Step with Differential Perturbations.
    
    Optimized for speed by assuming environmental perturbations (J2/Drag) are constant 
    over the integration step (Frozen Perturbation Assumption).
    """
    
    # Unpack & Setup
    chief_r = state["chief_r"]
    chief_v = state["chief_v"]
    rho = state["deputy_rho"]         
    rho_dot = state["deputy_rho_dot"] 
    epoch = state["epoch"] 
    if epoch is None:
        raise ValueError("Simulation epoch is required for SRP perturbation modeling.")
    u_ctrl_t = state.get("control_accel", np.zeros(3)) 

    # Pre-Compute Perturbations (Frozen Assumption)
    sim_config = config.get("simulation", {}).get("perturbations", {})
    sat_props = config.get("satellite_properties", {})
    
    # Safe retrieval of mass/drag properties
    chief_mass = config.get("satellites", {}).get("chief", {}).get("mass", 250.0)
    deputy_mass = config.get("satellites", {}).get("deputy", {}).get("mass", 500.0)
    chief_drag = sat_props.get("chief", {"Cd": 2.2, "area": 1.0})
    deputy_drag = sat_props.get("deputy", {"Cd": 2.2, "area": 1.0})

    # Reconstruct initial deputy inertial state for perturbation calc
    deputy_r, deputy_v = rel_vector_to_inertial(rho, rho_dot, chief_r, chief_v)
    
    # Compute Forces in ECI
    a_pert_chief = compute_perturb_accel(chief_r, chief_v, sim_config, chief_drag, chief_mass, epoch)
    a_pert_deputy = compute_perturb_accel(deputy_r, deputy_v, sim_config, deputy_drag, deputy_mass, epoch)
    
    # Differential accel in Inertial Frame
    a_diff_eci = a_pert_deputy - a_pert_chief
    
    # Rotate to LVLH (Chief's frame at t0)
    C_HN = LVLH_DCM(chief_r, chief_v)
    a_diff_lvlh = C_HN @ a_diff_eci
    
    # Total external force in Time Domain (held constant for this step)
    u_total_t = u_ctrl_t + a_diff_lvlh

    # Orbit Geometry Characterization
    r_mag = np.linalg.norm(chief_r)
    v_mag = np.linalg.norm(chief_v)
    
    h_vec = np.cross(chief_r, chief_v)
    h = np.linalg.norm(h_vec)
    
    # Eccentricity vector
    e_vec = (1.0 / MU_EARTH) * ((v_mag**2 - MU_EARTH / r_mag) * chief_r - np.dot(chief_r, chief_v) * chief_v)
    e = np.linalg.norm(e_vec)
    
    # True Anomaly (nu_0)
    if e > 1e-6:
        cos_nu = np.clip(np.dot(e_vec, chief_r) / (e * r_mag), -1.0, 1.0)
        nu_0 = np.arccos(cos_nu)
        if np.dot(chief_r, chief_v) < 0:
            nu_0 = 2 * np.pi - nu_0
    else:
        nu_0 = 0.0

    p = h**2 / MU_EARTH
    # Prevent div/0 for parabolic cases
    factor = (1 - e**2) if e < 1.0 else 1e-9
    a_semi = p / factor
    n_mean = np.sqrt(MU_EARTH / a_semi**3)
    
    # Kepler Solver (Time -> Anomaly)
    beta = np.sqrt(1 - e**2)
    
    # E0
    sin_E0 = (beta * np.sin(nu_0)) / (1 + e * np.cos(nu_0))
    cos_E0 = (e + np.cos(nu_0)) / (1 + e * np.cos(nu_0))
    E0 = np.arctan2(sin_E0, cos_E0)
    
    # M0 -> M_target
    M0 = E0 - e * np.sin(E0)
    M_target = M0 + n_mean * dt
    
    # Newton-Raphson Solver for E_final
    E_curr = M_target if e < 0.8 else np.pi 
    for _ in range(15): 
        f_val = E_curr - e * np.sin(E_curr) - M_target
        if abs(f_val) < 1e-12: break
        df_val = 1 - e * np.cos(E_curr)
        E_curr -= f_val / df_val
    E_final = E_curr
    
    # nu_final
    sin_nu_f = (beta * np.sin(E_final)) / (1 - e * np.cos(E_final))
    cos_nu_f = (np.cos(E_final) - e) / (1 - e * np.cos(E_final))
    nu_final = np.arctan2(sin_nu_f, cos_nu_f)
    
    if nu_final < nu_0:
        nu_final += 2 * np.pi
    
    d_nu_total = nu_final - nu_0

    # Integration (Anomaly Domain)
    # Transform Initial State: [rho, rho_dot] -> [rho, rho']
    nu_dot_0 = h / r_mag**2 
    rho_prime_0 = rho_dot / nu_dot_0
    state_nu = np.hstack((rho, rho_prime_0))

    # Dynamic stepping, use 1 step for small dt, more for large arcs
    steps = max(1, int(d_nu_total / 0.1))
    h_step = d_nu_total / steps
    
    curr_nu = nu_0
    y_curr = state_nu.copy()
    
    # RK4 Loop
    for _ in range(steps):
        k1 = h_step * _th_derivs(curr_nu, y_curr, e, p, h, u_total_t)
        k2 = h_step * _th_derivs(curr_nu + 0.5 * h_step, y_curr + 0.5 * k1, e, p, h, u_total_t)
        k3 = h_step * _th_derivs(curr_nu + 0.5 * h_step, y_curr + 0.5 * k2, e, p, h, u_total_t)
        k4 = h_step * _th_derivs(curr_nu + h_step, y_curr + k3, e, p, h, u_total_t)
        
        y_curr = y_curr + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        curr_nu += h_step

    state_nu_final = y_curr

    # Reconstruction (Anomaly -> Time)
    rho_final = state_nu_final[:3]
    rho_prime_final = state_nu_final[3:]
    
    r_final_mag = p / (1 + e * np.cos(nu_final))
    nu_dot_final = h / r_final_mag**2
    rho_dot_final = rho_prime_final * nu_dot_final
    
    # Propagate Chief Analytically (Exact Keplerian geometry)
    dE_total = E_final - E0
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


