## CWH relative motion propagator
import numpy as np
from src.propagators.perturbation_accel import compute_perturb_accel
from utils.frame_convertions.rel_to_inertial_functions import LVLH_DCM, rel_vector_to_inertial, compute_omega
from data.resources.constants import MU_EARTH

def step_cwh(state: dict, dt: float, config: dict):
    chief_r = state["chief_r"]
    chief_v = state["chief_v"]
    deputy_r = state["deputy_r"]
    deputy_v = state["deputy_v"]
    deputy_rho = state["deputy_rho"]
    deputy_rho_dot = state["deputy_rho_dot"]

    # Simulation and perturbations config
    epoch = state.get("epoch")
    sim = config.get("simulation", {})
    perturb_config = sim.get("perturbations", {})

    # Satellite configs
    sat_props = config.get("satellite_properties", {})
    chief_props = sat_props.get("chief", {})
    deputy_props = sat_props.get("deputy", {})

    # Masses (TODO: Not currently working, default to 250kg and 500kg)
    chief_mass = config.get("satellites", {}).get("chief", {}).get("mass", 250.0)
    deputy_mass = config.get("satellites", {}).get("deputy", {}).get("mass", 500.0)

    # Drag-related properties (inside 'properties')
    chief_drag_properties = {
        "cd": chief_props.get("Cd", 2.2),
        "area": chief_props.get("area", 1.0)
    }
    deputy_drag_properties = {
        "cd": deputy_props.get("Cd", 2.2),
        "area": deputy_props.get("area", 1.0)
    }


    # Mean motion (assume circular chief orbit)
    r_mag = np.linalg.norm(chief_r)
    n = np.sqrt(MU_EARTH / r_mag**3)

    x, y, z = deputy_rho
    vx, vy, vz = deputy_rho_dot

    # CWH accelerations
    ax = 3 * n**2 * x + 2 * n * vy
    ay = -2 * n * vx
    az = -n**2 * z

    # Compute perturbation accelerations if enabled
    # Chief perturbations
    a_pert_chief_inertial = compute_perturb_accel(chief_r, chief_v, perturb_config, chief_drag_properties, chief_mass, epoch)

    # Deputy perturbations
    a_pert_deputy_inertial = compute_perturb_accel(deputy_r, deputy_v, perturb_config, deputy_drag_properties, deputy_mass, epoch)

    # Compute differential perturbation acceleration
    a_diff_inertial = a_pert_deputy_inertial - a_pert_chief_inertial

    # Transform differential perturbation to LVLH frame
    # Compute DCM from inertial to LVLH
    C_HN = LVLH_DCM(chief_r, chief_v)

    # Compute omega and omega_dot
    h_chief = np.cross(chief_r, chief_v)
    f_dot = h_chief / np.linalg.norm(chief_r)**2
    f_ddot = -2 * np.dot(chief_v,[1,0,0]) / np.linalg.norm(chief_r) * f_dot

    # Compute Coriolis acceleration correction
    at1 = -2 * np.cross(f_dot, deputy_rho_dot)

    # Compute centrifugal acceleration correction
    at2 = - np.cross(f_dot, np.cross(f_dot, deputy_rho))

    # Compute Euler acceleration correction
    at3 = - np.cross(f_ddot, deputy_rho)

    a_diff = C_HN @ a_diff_inertial + at1 + at2 + at3  # transform to LVLH

    # Add differential perturbation to relative accelerations
    ax += a_diff[0]
    ay += a_diff[1]
    az += a_diff[2]

    # Euler integration of relative state for one step
    deputy_rho_next = deputy_rho + deputy_rho_dot * dt
    deputy_rho_dot_next = deputy_rho_dot + np.array([ax, ay, az]) * dt

    # Propagate chief using keplerian motion (circular orbit)
    theta = n * dt
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    STM = np.array([[cos_theta, -sin_theta, 0],
                  [sin_theta,  cos_theta, 0],
                  [0,          0,         1]])
    chief_r_next = STM @ chief_r
    chief_v_next = STM @ chief_v

    # Update deputy inertial position and velocity
    deputy_r_next, deputy_v_next = rel_vector_to_inertial(deputy_rho_next, deputy_rho_dot_next, chief_r_next, chief_v_next)

        # Return updated state as dictionary
    return {
        "chief_r": chief_r_next,
        "chief_v": chief_v_next,
        "deputy_r": deputy_r_next,
        "deputy_v": deputy_v_next,
        "deputy_rho": deputy_rho_next,
        "deputy_rho_dot": deputy_rho_dot_next
    }