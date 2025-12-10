import numpy as np
from src.propagators.perturbation_accel import compute_perturb_accel
from utils.frame_conversions.rel_to_inertial_functions import rel_vector_to_inertial, LVLH_DCM
from data.resources.constants import MU_EARTH

def step_th(state: dict, dt: float, config: dict):
    """
    Tschauner-Hempel (TH) Propagator Step with Differential Perturbations.
    
    Integrates the relative motion equations with respect to True Anomaly (nu)
    while incorporating differential perturbations (J2, Drag, SRP).
    """
    
    #  Unpack State & Config 
    chief_r = state["chief_r"]
    chief_v = state["chief_v"]
    rho = state["deputy_rho"]         
    rho_dot = state["deputy_rho_dot"] 
    
    epoch = state.get("epoch")
    u_ctrl_t = state.get("control_accel", np.zeros(3)) 

    # Perturbation Config
    sim_config = config.get("simulation", {})
    perturb_config = sim_config.get("perturbations", {})
    
    # Satellite Properties for Drag/SRP
    sat_props = config.get("satellite_properties", {})
    chief_mass = config.get("satellites", {}).get("chief", {}).get("mass", 250.0)
    deputy_mass = config.get("satellites", {}).get("deputy", {}).get("mass", 500.0)
    chief_drag = {"cd": sat_props.get("chief", {}).get("Cd", 2.2), "area": sat_props.get("chief", {}).get("area", 1.0)}
    deputy_drag = {"cd": sat_props.get("deputy", {}).get("Cd", 2.2), "area": sat_props.get("deputy", {}).get("area", 1.0)}

    #  Characterize Chief Orbit 
    r_vec = chief_r
    v_vec = chief_v
    r_mag = np.linalg.norm(r_vec)
    v_mag = np.linalg.norm(v_vec)
    
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    
    e_vec = (1.0 / MU_EARTH) * ((v_mag**2 - MU_EARTH / r_mag) * r_vec - np.dot(r_vec, v_vec) * v_vec)
    e = np.linalg.norm(e_vec)
    
    # True Anomaly (nu_0)
    if e > 1e-6:
        cos_nu = np.clip(np.dot(e_vec, r_vec) / (e * r_mag), -1.0, 1.0)
        nu_0 = np.arccos(cos_nu)
        if np.dot(r_vec, v_vec) < 0:
            nu_0 = 2 * np.pi - nu_0
    else:
        nu_0 = 0.0

    p = h**2 / MU_EARTH
    a = p / (1 - e**2)
    n_mean = np.sqrt(MU_EARTH / a**3)
    
    #  Determine Integration Span (d_nu) 
    beta = np.sqrt(1 - e**2)
    
    # E0
    sin_E0 = (beta * np.sin(nu_0)) / (1 + e * np.cos(nu_0))
    cos_E0 = (e + np.cos(nu_0)) / (1 + e * np.cos(nu_0))
    E0 = np.arctan2(sin_E0, cos_E0)
    
    # M0 -> M_target
    M0 = E0 - e * np.sin(E0)
    M_target = M0 + n_mean * dt
    
    # Solve for E_final
    E_curr = M_target
    for _ in range(10): 
        f_val = E_curr - e * np.sin(E_curr) - M_target
        df_val = 1 - e * np.cos(E_curr)
        E_curr -= f_val / df_val
        if abs(f_val) < 1e-12: break
    E_final = E_curr
    
    # nu_final
    sin_nu_f = (beta * np.sin(E_final)) / (1 - e * np.cos(E_final))
    cos_nu_f = (np.cos(E_final) - e) / (1 - e * np.cos(E_final))
    nu_final = np.arctan2(sin_nu_f, cos_nu_f)
    
    if nu_final < nu_0:
        nu_final += 2 * np.pi
    
    d_nu_total = nu_final - nu_0

    #  Transform Initial State to Anomaly Domain 
    nu_dot_0 = h / r_mag**2 
    rho_prime_0 = rho_dot / nu_dot_0
    state_nu = np.hstack((rho, rho_prime_0))

    #  Define Dynamics with Perturbations 
    def compute_th_derivs(nu_local, y):
        # Unpack Anomaly State
        x, y_pos, z = y[:3]
        xp, yp, zp = y[3:]
        
        # A. Orbit Geometry at local step
        rho_local = 1 + e * np.cos(nu_local) 
        r_local = p / rho_local
        nu_dot_local = h / r_local**2
        
        # Reconstruct Chief ECI State at nu_local for perturbations
        # Calculate local Eccentric Anomaly E_loc from nu_local
        sin_nu_loc = np.sin(nu_local)
        cos_nu_loc = np.cos(nu_local)
        
        sin_E_loc = (rho_local * sin_nu_loc) / beta 
        E_loc = np.arctan2(sin_E_loc * beta, e + cos_nu_loc) # Re-derived for safety

        dE_local = E_loc - E0
        
        # Lagrange Coefficients (f/g) to propagate Chief from t0 to t_local
        # This is exact for the unperturbed reference orbit
        sin_dE = np.sin(dE_local)
        cos_dE = np.cos(dE_local)
        
        f = 1.0 + (a / r_mag) * (cos_dE - 1.0)
        g = r_mag * np.sqrt(a / MU_EARTH) * sin_dE + (np.dot(r_vec, v_vec) / MU_EARTH) * a * (1.0 - cos_dE)
        f_dot = -(np.sqrt(MU_EARTH * a) / (r_mag * r_local)) * sin_dE
        g_dot = 1.0 + (a / r_local) * (cos_dE - 1.0)
        
        chief_r_loc = f * chief_r + g * chief_v
        chief_v_loc = f_dot * chief_r + g_dot * chief_v

        # Reconstruct Deputy ECI State
        # Convert y (rho, rho') -> (rho, rho_dot)
        curr_rho = y[:3]
        curr_rho_dot = y[3:] * nu_dot_local
        
        # Rel -> Inertial
        deputy_r_loc, deputy_v_loc = rel_vector_to_inertial(curr_rho, curr_rho_dot, chief_r_loc, chief_v_loc)
        
        # Compute Differential Perturbations (ECI)
        a_pert_chief = compute_perturb_accel(chief_r_loc, chief_v_loc, perturb_config, chief_drag, chief_mass, epoch) # type: ignore
        a_pert_deputy = compute_perturb_accel(deputy_r_loc, deputy_v_loc, perturb_config, deputy_drag, deputy_mass, epoch) # type: ignore
        
        a_diff_eci = a_pert_deputy - a_pert_chief
        
        # Rotate Perturbations to LVLH
        C_HN = LVLH_DCM(chief_r_loc, chief_v_loc)
        a_diff_lvlh = C_HN @ a_diff_eci
        
        # Total Forcing Term (Control + Perturbations) scaled to Anomaly Domain
        # u_nu = u_t / nu_dot^2
        u_total_t = u_ctrl_t + a_diff_lvlh
        u_accel_nu = u_total_t / (nu_dot_local**2)
        
        # TH Equations
        x_pp_nat = 2 * yp + (3.0 / rho_local) * x
        y_pp_nat = -2 * xp
        z_pp_nat = -z
        
        return np.array([xp, yp, zp, 
                         x_pp_nat + u_accel_nu[0], 
                         y_pp_nat + u_accel_nu[1], 
                         z_pp_nat + u_accel_nu[2]])

    #  Integrate 
    steps = 10 
    h_step = d_nu_total / steps
    
    curr_nu = nu_0
    y_curr = state_nu.copy()
    
    for _ in range(steps):
        k1 = h_step * compute_th_derivs(curr_nu, y_curr)
        k2 = h_step * compute_th_derivs(curr_nu + 0.5 * h_step, y_curr + 0.5 * k1)
        k3 = h_step * compute_th_derivs(curr_nu + 0.5 * h_step, y_curr + 0.5 * k2)
        k4 = h_step * compute_th_derivs(curr_nu + h_step, y_curr + k3)
        
        y_curr = y_curr + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        curr_nu += h_step

    state_nu_final = y_curr

    #  Final Transform Back to Time Domain 
    rho_final = state_nu_final[:3]
    rho_prime_final = state_nu_final[3:]
    
    r_final_mag = p / (1 + e * np.cos(nu_final))
    nu_dot_final = h / r_final_mag**2
    rho_dot_final = rho_prime_final * nu_dot_final
    
    # Propagate Chief Analytically 
    dE_total = E_final - E0
    sin_dE = np.sin(dE_total)
    cos_dE = np.cos(dE_total)
    
    f = 1.0 + (a / r_mag) * (cos_dE - 1.0)
    g = r_mag * np.sqrt(a / MU_EARTH) * sin_dE + (np.dot(r_vec, v_vec) / MU_EARTH) * a * (1.0 - cos_dE)
    f_dot = -(np.sqrt(MU_EARTH * a) / (r_mag * r_final_mag)) * sin_dE
    g_dot = 1.0 + (a / r_final_mag) * (cos_dE - 1.0)
    
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