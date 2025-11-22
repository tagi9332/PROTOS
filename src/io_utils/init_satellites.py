import numpy as np
from utils.orbital_element_conversions.oe_conversions import orbital_elements_to_inertial, lroes_to_inertial, inertial_to_oes
from utils.frame_conversions.rel_to_inertial_functions import (
    inertial_to_rel_LVLH, rel_vector_to_inertial,
    LVLH_DCM, compute_omega
)

# Initialize chief state
def _init_chief_state(chief: dict):
    chief_initial = chief["initial_state"]
    frame = chief_initial.get("frame", "").upper()
    chief_vector = np.array(chief_initial["state"])

    if frame == "ECI":
        # Chief given directly in ECI
        chief_r = chief_vector[:3]
        chief_v = chief_vector[3:]

    elif frame == "OES":
        # Chief given in orbital elements vector [a, e, i, raan, argp, ta]
        chief_r, chief_v = orbital_elements_to_inertial(*chief_vector, mu=398600.4418, units='deg')

    else:
        raise ValueError("Chief initial state must be either ECI or ORBITAL_ELEMENTS")

    return chief_r, chief_v

def _init_deputy_state(deputy: dict, chief_r: np.ndarray, chief_v: np.ndarray, chief):

    deputy_state = np.array(deputy["initial_state"]["state"])
    frame = deputy["initial_state"].get("frame", "").upper()

    # ================================
    # CASE 1 — ECI input; r,v in km and km/s
    # ================================
    if frame == "ECI":
        deputy_r = deputy_state[:3]
        deputy_v = deputy_state[3:]

        # Convert to LVLH
        C_HN = LVLH_DCM(chief_r, chief_v)
        deputy_rho = C_HN @ (deputy_r - chief_r)

        omega = compute_omega(chief_r, chief_v)
        deputy_rho_dot = C_HN @ (deputy_v - chief_v) - np.cross(omega, deputy_rho)

    # ================================
    # CASE 2 — LVLH input; [rho_x, rho_y, rho_z, rho_dot_x, rho_dot_y, rho_dot_z] in km and km/s
    # ================================
    elif frame == "LVLH":
        deputy_rho = deputy_state[:3]
        deputy_rho_dot = deputy_state[3:]

        deputy_r, deputy_v = rel_vector_to_inertial(
            deputy_rho, deputy_rho_dot, chief_r, chief_v
        )

    # ================================
    # CASE 3 — LROES input; [A0, B0, alpha, beta, x_off, y_off]
    # ================================
    elif frame == "LROES":
        deputy_r, deputy_v = lroes_to_inertial(0, chief_r, chief_v, deputy_state)

        C_HN = LVLH_DCM(chief_r, chief_v)
        deputy_rho = C_HN @ (deputy_r - chief_r)

        omega = compute_omega(chief_r, chief_v)
        deputy_rho_dot = C_HN @ (deputy_v - chief_v) - np.cross(omega, deputy_rho)

    # ================================
    # CASE 4 — DOES input (delta orbital elements); [d_a, d_e, d_i, d_raan, d_argp, d_ta]
    # ================================
    elif frame == "DOES":
        # If chief initial state is given in OEs, use it directly to avoid numerical instability with zero eccentricities
        # (Otherwise, compute from inertial state)
        if chief["initial_state"].get("frame", "").upper() == "OES":
            oe_chief = np.array(chief["initial_state"]["state"]) 
        else:
            oe_chief = inertial_to_oes(chief_r, chief_v, units='deg')

        # Apply delta OEs
        oe_dep = oe_chief + deputy_state

        # Convert deputy OEs to inertial
        deputy_r, deputy_v = orbital_elements_to_inertial(*oe_dep, units='deg')

        # Convert to LVLH relative position and velocity
        deputy_rho, deputy_rho_dot = inertial_to_rel_LVLH(deputy_r, deputy_v, chief_r, chief_v)

    else:
        raise ValueError("Deputy initial state frame not recognized. Must be one of: ECI, LVLH, LROES, DOES.")

    return deputy_r, deputy_v, deputy_rho, deputy_rho_dot


def init_satellites(raw_config: dict, sim_config: dict) -> dict:
    """
    Initialize the chief and deputy satellite's states.
    """
    # Process satellites
    satellites = raw_config.get("satellites", [])

    # Separate chief and deputy
    chief = next(sat for sat in satellites if sat["name"].lower() == "chief")
    deputy = next(sat for sat in satellites if sat["name"].lower() == "deputy")

    # Initialize chief state
    chief_r, chief_v = _init_chief_state(chief)

    # Initialize deputy state
    deputy_r, deputy_v, deputy_rho, deputy_rho_dot = _init_deputy_state(deputy, chief_r, chief_v, chief)
    
    # Dynamics input: inertial positions/velocities + simulation config
    dynamics_input = {
        "satellite_properties": {
            "chief": chief.get("properties", {}),
            "deputy": deputy.get("properties", {})
        },
        "simulation": {
            "propagator": sim_config.get("propagator", "2BODY").upper(),
            "perturbations": sim_config.get("perturbations", {})
        }  
    }

    return {
        "dynamics_input": dynamics_input,
        "init_state": {
            "sim_time": 0.0,
            "epoch": sim_config['epoch'],
            "chief_r": chief_r,
            "chief_v": chief_v,
            "deputy_r": deputy_r,
            "deputy_v": deputy_v,
            "deputy_rho": deputy_rho,
            "deputy_rho_dot": deputy_rho_dot
        }
    }