import numpy as np
from utils.numerical_methods.rk4 import rk54
from src.propagators.perturbation_accel import compute_perturb_accel
from utils.frame_conversions.rel_to_inertial_functions import LVLH_DCM, compute_omega
from data.resources.constants import MU_EARTH

def step_2body(state: dict, dt: float, config: dict):

    chief_r = state["chief_r"]
    chief_v = state["chief_v"]
    deputy_r = state["deputy_r"]
    deputy_v = state["deputy_v"]
    u_ctrl = state.get("control_accel", np.zeros(3))

    sim = config.get("simulation", {})
    perturb_config = sim.get("perturbations", {})

    sat_props = config.get("satellite_properties", {})
    chief_props = sat_props.get("chief", {})
    deputy_props = sat_props.get("deputy", {})

    chief_mass = config.get("satellites", {}).get("chief", {}).get("mass", 250.0)
    deputy_mass = config.get("satellites", {}).get("deputy", {}).get("mass", 500.0)

    chief_drag = {"cd": chief_props.get("Cd", 2.2), "area": chief_props.get("area", 1.0)}
    deputy_drag = {"cd": deputy_props.get("Cd", 2.2), "area": deputy_props.get("area", 1.0)}

    epoch = state.get("epoch")

    # -------------------------------
    # Dynamics functions for RK4
    # -------------------------------
    def chief_dynamics(y):
        r = y[:3]
        v = y[3:]
        r_mag = np.linalg.norm(r)

        a = -MU_EARTH * r / r_mag**3
        a += compute_perturb_accel(r, v, perturb_config, chief_drag, chief_mass, epoch) # type: ignore

        return np.hstack((v, a))

    def deputy_dynamics(y):
        r = y[:3]
        v = y[3:]
        r_mag = np.linalg.norm(r)

        a = -MU_EARTH * r / r_mag**3
        a += compute_perturb_accel(r, v, perturb_config, deputy_drag, deputy_mass, epoch) # type: ignore
        a += u_ctrl   # control only acts on deputy

        return np.hstack((v, a))

    # -------------------------------
    # RK4 Integration
    # -------------------------------
    chief_state = np.hstack((chief_r, chief_v))
    deputy_state = np.hstack((deputy_r, deputy_v))

    chief_next = rk54(chief_dynamics, chief_state, dt)
    deputy_next = rk54(deputy_dynamics, deputy_state, dt)

    chief_r_next = chief_next[:3]
    chief_v_next = chief_next[3:]
    deputy_r_next = deputy_next[:3]
    deputy_v_next = deputy_next[3:]

    # -------------------------------
    # Relative state computation
    # -------------------------------
    C_HN = LVLH_DCM(chief_r_next, chief_v_next)
    deputy_rho_next = C_HN @ (deputy_r_next - chief_r_next)

    omega = compute_omega(chief_r_next, chief_v_next)
    deputy_rho_dot_next = C_HN @ (deputy_v_next - chief_v_next) - np.cross(omega, deputy_rho_next)

    return {
        "chief_r": chief_r_next,
        "chief_v": chief_v_next,
        "deputy_r": deputy_r_next,
        "deputy_v": deputy_v_next,
        "deputy_rho": deputy_rho_next,
        "deputy_rho_dot": deputy_rho_dot_next
    }
