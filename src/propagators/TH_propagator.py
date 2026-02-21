import numpy as np
from scipy.integrate import solve_ivp

from src.propagators.perturbation_accel import compute_perturb_accel
from utils.frame_conversions.rel_to_inertial_functions import rel_vector_to_inertial, LVLH_DCM
from utils.orbital_element_conversions.oe_conversions import inertial_to_oes, m_to_ta, ta_to_m
from data.resources.constants import MU_EARTH, J2000_EPOCH

def _th_derivs(nu, y, e, p, h, u_total_t):
    """Inner derivative function for TH equations."""
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

def step_th(sat_state: dict, dt: float, config: dict, **kwargs):
    """
    Tschauner-Hempel (TH) Propagator Step.
    Dynamically handles both Chief (Keplerian) and Deputy (TH) propagation.
    """
    is_chief = kwargs.get("is_chief", False)
    epoch = sat_state.get("epoch", J2000_EPOCH)

    chief_state = kwargs.get("chief_state", {})
    # ==========================================
    # 1. SETUP CHIEF INITIAL STATE & ORBIT GEOMETRY
    # ==========================================
    if is_chief:
        chief_r = np.array(sat_state["r"])
        chief_v = np.array(sat_state["v"])
    else:
        chief_state = kwargs.get("chief_state", {})
        chief_r = np.array(chief_state["r"])
        chief_v = np.array(chief_state["v"])

    # Orbit Geometry
    a_semi, e, _, _, _, nu_0 = inertial_to_oes(chief_r, chief_v, units='rad')
    p = a_semi * (1 - e**2)
    n_mean = np.sqrt(MU_EARTH / a_semi**3)
    
    # Calculate Mean Anomaly propagation
    M0 = ta_to_m(nu_0, e)
    M_target = M0 + n_mean * dt
    
    # Solve for final True Anomaly
    nu_final = m_to_ta(M_target, e)
    if nu_final < nu_0:
        nu_final += 2 * np.pi

    r_mag = np.linalg.norm(chief_r)
    h_val = np.sqrt(MU_EARTH * p)
    r_final_mag = p / (1 + e * np.cos(nu_final))

    # ==========================================
    # 2. KEPLERIAN CHIEF PROPAGATION (Gauss f/g)
    # ==========================================
    def get_E(nu, ecc):
        return 2.0 * np.arctan2(np.sqrt(1.0 - ecc) * np.sin(nu/2.0),
                                np.sqrt(1.0 + ecc) * np.cos(nu/2.0))
    
    E0 = get_E(nu_0, e)
    E_final = get_E(nu_final, e)
    if E_final < E0: E_final += 2*np.pi
    dE_total = E_final - E0

    sin_dE = np.sin(dE_total)
    cos_dE = np.cos(dE_total)
    
    f = 1.0 + (a_semi / r_mag) * (cos_dE - 1.0)
    g = r_mag * np.sqrt(a_semi / MU_EARTH) * sin_dE + (np.dot(chief_r, chief_v) / MU_EARTH) * a_semi * (1.0 - cos_dE)
    f_dot = -(np.sqrt(MU_EARTH * a_semi) / (r_mag * r_final_mag)) * sin_dE
    g_dot = 1.0 + (a_semi / r_final_mag) * (cos_dE - 1.0)
    
    chief_r_next = f * chief_r + g * chief_v
    chief_v_next = f_dot * chief_r + g_dot * chief_v

    # If this is the chief, we are done! Return its new inertial state.
    if is_chief:
        return {"r": chief_r_next.tolist(), "v": chief_v_next.tolist()}

    # ==========================================
    # 3. DEPUTY PROPAGATION (TH Equations)
    # ==========================================
    rho = np.array(sat_state["rho"])
    rho_dot = np.array(sat_state["rho_dot"])
    u_ctrl_t = np.array(sat_state.get("accel_cmd", [0.0, 0.0, 0.0]))
    sim_config = getattr(config.get("simulation", {}), "perturbations", {})

    # Compute differential perturbations utilizing our perfectly clean state dictionary!
    if sim_config.get("enable_perturbations", False):
        c_mass = chief_state.get("mass", 250.0)
        c_drag = {"Cd": chief_state.get("Cd", 2.2), "area": chief_state.get("area", 1.0)}
        
        d_mass = sat_state.get("mass", 500.0)
        d_drag = {"Cd": sat_state.get("Cd", 2.2), "area": sat_state.get("area", 1.0)}

        # Reconstruct deputy inertial state for perturbation calc
        deputy_r, deputy_v = rel_vector_to_inertial(rho, rho_dot, chief_r, chief_v)
        
        a_pert_chief = compute_perturb_accel(chief_r, chief_v, sim_config, c_drag, c_mass, epoch)
        a_pert_deputy = compute_perturb_accel(deputy_r, deputy_v, sim_config, d_drag, d_mass, epoch)

        a_diff_eci = a_pert_deputy - a_pert_chief
        C_HN = LVLH_DCM(chief_r, chief_v)
        u_total_t = u_ctrl_t + (C_HN @ a_diff_eci)
    else:
        u_total_t = u_ctrl_t

    # Integration (Anomaly Domain)
    nu_dot_0 = h_val / r_mag**2 
    rho_prime_0 = rho_dot / nu_dot_0
    y_curr = np.hstack((rho, rho_prime_0))

    sol = solve_ivp(
        _th_derivs, 
        (nu_0, nu_final), 
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
    
    nu_dot_final = h_val / r_final_mag**2
    rho_dot_final = rho_prime_final * nu_dot_final

    # Reconstruct Final Deputy ECI (using the chief_next we analytically propagated)
    deputy_r_next, deputy_v_next = rel_vector_to_inertial(rho_final, rho_dot_final, chief_r_next, chief_v_next)

    return {
        "r": deputy_r_next.tolist(), 
        "v": deputy_v_next.tolist(),
        "rho": rho_final.tolist(),
        "rho_dot": rho_dot_final.tolist()
    }