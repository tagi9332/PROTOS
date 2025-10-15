## 2-Body full nonlinear equations of motion
from logging import config
import numpy as np
from src.propagators.perturbation_accel import compute_perturb_accel
from utils.frame_convertions.rel_to_inertial_functions import LVLH_DCM, rel_vector_to_inertial, compute_omega
from data.resources.constants import MU_EARTH, R_EARTH, J2

def step_2body(state: dict, dt: float, config: dict):
    """
    Propagate chief and deputy using two-body dynamics in inertial frame with RK4.
    state: dict containing chief and deputy positions/velocities
    dt: time step [s]
    config: simulation config dict
    """
    chief_r = state["chief_r"]
    chief_v = state["chief_v"]
    deputy_r = state["deputy_r"]
    deputy_v = state["deputy_v"]

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


    # Acceleration function
    def compute_accel(r: np.ndarray, v: np.ndarray, perturb_config: dict, drag_properties: dict, mass: float):
        """
        Compute total acceleration: central body + perturbations
        """
        r_mag = np.linalg.norm(r)
        a_total = -MU_EARTH * r / r_mag**3
        a_total += compute_perturb_accel(r, v, perturb_config, drag_properties, mass, epoch) # type: ignore
        return a_total

    # RK4 integration for chief
    k1_vc = compute_accel(chief_r, chief_v, perturb_config, chief_drag_properties, chief_mass) * dt
    k1_rc = chief_v * dt

    k2_vc = compute_accel(chief_r + 0.5 * k1_rc, chief_v + 0.5 * k1_vc, perturb_config, chief_drag_properties, chief_mass) * dt
    k2_rc = (chief_v + 0.5 * k1_vc) * dt

    k3_vc = compute_accel(chief_r + 0.5 * k2_rc, chief_v + 0.5 * k2_vc, perturb_config, chief_drag_properties, chief_mass) * dt
    k3_rc = (chief_v + 0.5 * k2_vc) * dt

    k4_vc = compute_accel(chief_r + k3_rc, chief_v + k3_vc, perturb_config, chief_drag_properties, chief_mass) * dt
    k4_rc = (chief_v + k3_vc) * dt

    chief_r_next = chief_r + (k1_rc + 2*k2_rc + 2*k3_rc + k4_rc) / 6
    chief_v_next = chief_v + (k1_vc + 2*k2_vc + 2*k3_vc + k4_vc) / 6

    # RK4 integration for deputy
    k1_vd = compute_accel(deputy_r, deputy_v, perturb_config, deputy_drag_properties, deputy_mass) * dt
    k1_rd = deputy_v * dt

    k2_vd = compute_accel(deputy_r + 0.5 * k1_rd, deputy_v + 0.5 * k1_vd, perturb_config, deputy_drag_properties, deputy_mass) * dt
    k2_rd = (deputy_v + 0.5 * k1_vd) * dt

    k3_vd = compute_accel(deputy_r + 0.5 * k2_rd, deputy_v + 0.5 * k2_vd, perturb_config, deputy_drag_properties, deputy_mass) * dt
    k3_rd = (deputy_v + 0.5 * k2_vd) * dt

    k4_vd = compute_accel(deputy_r + k3_rd, deputy_v + k3_vd, perturb_config, deputy_drag_properties, deputy_mass) * dt
    k4_rd = (deputy_v + k3_vd) * dt

    deputy_r_next = deputy_r + (k1_rd + 2*k2_rd + 2*k3_rd + k4_rd) / 6
    deputy_v_next = deputy_v + (k1_vd + 2*k2_vd + 2*k3_vd + k4_vd) / 6

    # Compute updated relative state in LVLH frame
    C_HN = LVLH_DCM(chief_r_next, chief_v_next) 
    deputy_rho_next = C_HN @ (deputy_r_next - chief_r_next)
    omega = compute_omega(chief_r_next, chief_v_next)
    deputy_rho_dot_next = C_HN @ (deputy_v_next - chief_v_next) - np.cross(omega, deputy_rho_next)  # assuming zero angular velocity for simplicity

    # Return updated state
    return {
        "chief_r": chief_r_next,
        "chief_v": chief_v_next,
        "deputy_r": deputy_r_next,
        "deputy_v": deputy_v_next,
        "deputy_rho": deputy_rho_next,
        "deputy_rho_dot": deputy_rho_dot_next
    }