import numpy as np
from utils.numerical_methods.rk4 import rk54
from src.propagators.perturbation_accel import compute_perturb_accel
from utils.frame_conversions.rel_to_inertial_functions import LVLH_DCM, compute_omega
from data.resources.constants import MU_EARTH

def step_cwh(state: dict, dt: float, config: dict):

    chief_r = state["chief_r"]
    chief_v = state["chief_v"]
    deputy_rho = state["deputy_rho"]          # relative position in LVLH
    deputy_rho_dot = state["deputy_rho_dot"]  # relative velocity in LVLH
    
    # Control inputs for CWH must be given in LVLH frame
    u_ctrl = state.get("control_accel", np.zeros(3))

    sim = config.get("simulation", {})
    perturb_config = sim.get("perturbations", {})
    epoch = state["epoch"] 
    
    sat_props = config.get("satellite_properties", {})
    chief_props = sat_props.get("chief", {})
    deputy_props = sat_props.get("deputy", {})

    chief_mass = chief_props.get("mass", 500.0)
    deputy_mass = deputy_props.get("mass", 250.0)
    
    chief_drag = {"Cd": chief_props.get("Cd", 2.2), "area": chief_props.get("area", 1.0)}
    deputy_drag = {"Cd": deputy_props.get("Cd", 2.2), "area": deputy_props.get("area", 1.0)}

    # -------------------------------
    # Dynamics functions for RK4
    # -------------------------------
    # State Vector Y = [Chief_r(3), Chief_v(3), Rho(3), Rho_dot(3)]
    
    def combined_dynamics(y):
        # Unpack current integration step states
        c_r = y[0:3]
        c_v = y[3:6]
        c_r_mag = np.linalg.norm(c_r)
        
        curr_rho = y[6:9]
        curr_rho_dot = y[9:12]

        # Chief Dynamics (ECI)
        a_chief_2body = -MU_EARTH * c_r / c_r_mag**3
        a_pert_chief_eci = compute_perturb_accel(c_r, c_v, perturb_config, chief_drag, chief_mass, epoch)
        
        a_chief_total_eci = a_chief_2body + a_pert_chief_eci

        # CWH Dynamics (LVLH)
        # Calculate instantaneous mean motion 'n' based on current radius
        n = np.sqrt(MU_EARTH / c_r_mag**3)

        x, y, z = curr_rho
        xd, yd, _ = curr_rho_dot

        # Standard CWH (unperturbed) accelerations
        ax_cwh = 2*n*yd + 3*n**2*x
        ay_cwh = -2*n*xd
        az_cwh = -n**2*z
        a_cwh_natural = np.array([ax_cwh, ay_cwh, az_cwh])

        # Perturbation Handling (Differential)
        # Reconstruct Deputy State in ECI to calculate its perturbations
        C_HN = LVLH_DCM(c_r, c_v) # Inertial -> LVLH Rotation Matrix
        
        # Deputy Position ECI: r_dep = r_chief + C_NB^T * rho
        dep_r_eci = c_r + C_HN.T @ curr_rho
        
        # Deputy Velocity ECI: v_dep = v_chief + C_NB^T * (omega x rho + rho_dot)
        omega_eci = compute_omega(c_r, c_v) # Angular velocity of LVLH frame wrt Inertial
        dep_v_eci = c_v + C_HN.T @ (np.cross(omega_eci, curr_rho) + curr_rho_dot)

        # Compute Deputy Perturbations in ECI
        a_pert_deputy_eci = compute_perturb_accel(dep_r_eci, dep_v_eci, perturb_config, deputy_drag, deputy_mass, epoch)

        # Compute Differential Acceleration in ECI
        a_diff_eci = a_pert_deputy_eci - a_pert_chief_eci

        # Rotate Differential Acceleration into LVLH
        a_diff_lvlh = C_HN @ a_diff_eci

        # Total Relative Acceleration
        a_rel_total_lvlh = a_cwh_natural + a_diff_lvlh + u_ctrl

        # Return concatenated derivatives [v_chief, a_chief, rho_dot, rho_ddot]
        return np.hstack((c_v, a_chief_total_eci, curr_rho_dot, a_rel_total_lvlh))


    # -------------------------------
    # RK4 Integration
    # -------------------------------
    full_state = np.hstack((chief_r, chief_v, deputy_rho, deputy_rho_dot))
    next_full_state = rk54(combined_dynamics, full_state, dt)

    # -------------------------------
    # Unpack and Return
    # -------------------------------
    chief_r_next = next_full_state[0:3]
    chief_v_next = next_full_state[3:6]
    deputy_rho_next = next_full_state[6:9]
    deputy_rho_dot_next = next_full_state[9:12]

    # Reconstruct Deputy ECI
    C_HN_next = LVLH_DCM(chief_r_next, chief_v_next)
    deputy_r_next = chief_r_next + C_HN_next.T @ deputy_rho_next
    
    omega_next = compute_omega(chief_r_next, chief_v_next)
    deputy_v_next = chief_v_next + C_HN_next.T @ (np.cross(omega_next, deputy_rho_next) + deputy_rho_dot_next)

    return {
        "chief_r": chief_r_next,
        "chief_v": chief_v_next,
        "deputy_r": deputy_r_next,
        "deputy_v": deputy_v_next,
        "deputy_rho": deputy_rho_next,
        "deputy_rho_dot": deputy_rho_dot_next
    }