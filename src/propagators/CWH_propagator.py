import numpy as np
from scipy.integrate import solve_ivp

from src.propagators.perturbation_accel import compute_perturb_accel
from utils.frame_conversions.rel_to_inertial_functions import LVLH_DCM, compute_omega
from data.resources.constants import MU_EARTH

def step_cwh(sat_state: dict, dt: float, config: dict, **kwargs):
    """
    Clohessy-Wiltshire (CWH) Propagator Step.
    Dynamically handles Chief (ECI) and Deputy (LVLH) propagation.
    """
    is_chief = kwargs.get("is_chief", False)
    chief_state = kwargs.get("chief_state", {})
    
    # Get epoch for sun vectors (SRP)
    epoch = sat_state.get("epoch", chief_state.get("epoch", 0.0)) 
    sim_config = config.get("simulation", {})
    perturb_config = getattr(sim_config, "perturbations", {})

    # ==========================================
    # 1. CHIEF PROPAGATION (ECI 2-Body + Perturbations)
    # ==========================================
    if is_chief:
        c_r = np.array(sat_state["r"])
        c_v = np.array(sat_state["v"])
        mass = sat_state.get("mass", 250.0)
        drag = {"Cd": sat_state.get("Cd", 2.2), "area": sat_state.get("area", 1.0)}

        def chief_dyn(t, y):
            r = y[0:3]
            v = y[3:6]
            a_2body = -MU_EARTH * r / np.linalg.norm(r)**3
            a_pert = compute_perturb_accel(r, v, perturb_config, drag, mass, epoch)
            return np.hstack((v, a_2body + a_pert))

        sol = solve_ivp(chief_dyn, (0, dt), np.hstack((c_r, c_v)), method='RK45', rtol=1e-12, atol=1e-12)
        
        return {
            "r": sol.y[0:3, -1].tolist(), 
            "v": sol.y[3:6, -1].tolist()
        }

    # ==========================================
    # 2. DEPUTY PROPAGATION (CWH Relative Dynamics)
    # ==========================================
    # Chief properties
    c_r = np.array(chief_state.get("r", [0, 0, 0]))
    c_v = np.array(chief_state.get("v", [0, 0, 0]))
    c_mass = chief_state.get("mass", 250.0)
    c_drag = {"Cd": chief_state.get("Cd", 2.2), "area": chief_state.get("area", 1.0)}

    # Deputy properties
    rho = np.array(sat_state["rho"])
    rho_dot = np.array(sat_state["rho_dot"])
    u_ctrl = np.array(sat_state.get("accel_cmd", [0.0, 0.0, 0.0]))
    d_mass = sat_state.get("mass", 500.0)
    d_drag = {"Cd": sat_state.get("Cd", 2.2), "area": sat_state.get("area", 1.0)}

    def combined_dynamics(t, y):
        curr_c_r = y[0:3]
        curr_c_v = y[3:6]
        curr_rho = y[6:9]
        curr_rho_dot = y[9:12]
        c_r_mag = np.linalg.norm(curr_c_r)

        # Chief Dynamics (ECI)
        a_chief_2body = -MU_EARTH * curr_c_r / c_r_mag**3
        a_pert_chief_eci = compute_perturb_accel(curr_c_r, curr_c_v, perturb_config, c_drag, c_mass, epoch)
        a_chief_total_eci = a_chief_2body + a_pert_chief_eci

        # CWH Dynamics (LVLH)
        n = np.sqrt(MU_EARTH / c_r_mag**3)
        x, y_pos, z = curr_rho
        xd, yd, _ = curr_rho_dot

        ax_cwh = 2*n*yd + 3*n**2*x
        ay_cwh = -2*n*xd
        az_cwh = -n**2*z
        a_cwh_natural = np.array([ax_cwh, ay_cwh, az_cwh])

        # Perturbation Handling
        C_HN = LVLH_DCM(curr_c_r, curr_c_v)

        # Rotate rho to ECI before cross products
        rho_eci = C_HN.T @ curr_rho
        dep_r_eci = curr_c_r + rho_eci
        
        omega_eci = compute_omega(curr_c_r, curr_c_v) 
        dep_v_eci = curr_c_v + np.cross(omega_eci, rho_eci) + (C_HN.T @ curr_rho_dot)

        a_pert_deputy_eci = compute_perturb_accel(dep_r_eci, dep_v_eci, perturb_config, d_drag, d_mass, epoch)
        a_diff_eci = a_pert_deputy_eci - a_pert_chief_eci
        a_diff_lvlh = C_HN @ a_diff_eci

        # Total Relative Acceleration
        u_ctrl_lvlh = C_HN @ u_ctrl 
        a_rel_total_lvlh = a_cwh_natural + a_diff_lvlh + u_ctrl_lvlh

        return np.hstack((curr_c_v, a_chief_total_eci, curr_rho_dot, a_rel_total_lvlh)).flatten()

    # Integrate the combined system
    full_state = np.hstack((c_r, c_v, rho, rho_dot))
    sol = solve_ivp(combined_dynamics, (0, dt), full_state, method='Radau', rtol=1e-12, atol=1e-12)
    next_full_state = sol.y[:, -1]

    # Unpack Deputy states
    next_c_r = next_full_state[0:3]
    next_c_v = next_full_state[3:6]
    next_rho = next_full_state[6:9]
    next_rho_dot = next_full_state[9:12]

    # Reconstruct Deputy ECI using the integrated Chief state
    C_HN_next = LVLH_DCM(next_c_r, next_c_v)
    
    # Rotate next_rho to ECI
    next_rho_eci = C_HN_next.T @ next_rho
    next_d_r = next_c_r + next_rho_eci
    
    omega_next = compute_omega(next_c_r, next_c_v)
    next_d_v = next_c_v + np.cross(omega_next, next_rho_eci) + (C_HN_next.T @ next_rho_dot)

    return {
        "r": next_d_r.tolist(),
        "v": next_d_v.tolist(),
        "rho": next_rho.tolist(),
        "rho_dot": next_rho_dot.tolist()
    }