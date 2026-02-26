import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

from data.resources.constants import MU_EARTH
from utils.orbital_element_conversions.oe_conversions import oes_to_inertial, lroes_to_inertial, inertial_to_oes
from utils.frame_conversions.rel_to_inertial_functions import (
    inertial_to_rel_LVLH, rel_vector_to_inertial,
    LVLH_DCM, compute_omega
)
from src.io_utils.init_sim_config import SimulationConfig

@dataclass
class AttitudeState:
    q_BN: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    omega_BN: np.ndarray = field(default_factory=lambda: np.zeros(3))

@dataclass
class InitialState:
    frame: str
    state: np.ndarray
    attitude: AttitudeState = field(default_factory=AttitudeState)

@dataclass
class SatelliteConfig:
    name: str
    initial_state: InitialState
    properties: Dict[str, Any] = field(default_factory=dict)
    gnc: Dict[str, Any] = field(default_factory=dict)

# Initialize chief state
def _init_chief_state(chief: SatelliteConfig) -> Tuple[np.ndarray, np.ndarray]:
    chief_initial = chief.initial_state
    frame = chief_initial.frame.upper()
    chief_vector = np.array(chief_initial.state)

    if frame == "ECI":
        # Chief given directly in ECI
        chief_r = chief_vector[:3]
        chief_v = chief_vector[3:]

    elif frame == "OES":
        # Chief given in orbital elements vector [a, e, i, raan, argp, ta]
        chief_r, chief_v = oes_to_inertial(*chief_vector, mu=MU_EARTH, units='deg')

    else:
        raise ValueError("Chief initial state must be either ECI or ORBITAL_ELEMENTS")

    return chief_r, chief_v

def _init_attitude(satellite: SatelliteConfig) -> Tuple[np.ndarray, np.ndarray]:
    attitude = satellite.initial_state.attitude
    return attitude.q_BN, attitude.omega_BN

def _init_deputy_state(deputy: SatelliteConfig, chief_r: np.ndarray, chief_v: np.ndarray, chief: SatelliteConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    deputy_state = deputy.initial_state.state
    frame = deputy.initial_state.frame.upper()

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
        # If chief initial state is given in OEs, use it directly
        # (Otherwise, compute from inertial state)
        if chief.initial_state.frame.upper() == "OES":
            oe_chief = chief.initial_state.state
        else:
            oe_chief = inertial_to_oes(chief_r, chief_v, units='deg')

        # Apply delta OEs
        oe_dep = oe_chief + deputy_state

        # Convert deputy OEs to inertial
        deputy_r, deputy_v = oes_to_inertial(*oe_dep, units='deg')

        # Convert to LVLH relative position and velocity
        deputy_rho, deputy_rho_dot = inertial_to_rel_LVLH(deputy_r, deputy_v, chief_r, chief_v)

    else:
        raise ValueError(f"Frame '{frame}' not recognized for deputy '{deputy.name}'. Must be one of: ECI, LVLH, LROES, DOES.")

    return deputy_r, deputy_v, deputy_rho, deputy_rho_dot


def init_satellites(raw_config: dict, sim_config: Any) -> dict:
    """
    Initialize the chief and deputy satellite's states.
    """
    # Parse raw dicts
    satellites: List[SatelliteConfig] = []
    seen_names = set()

    for sat_dict in raw_config.get("satellites", []):
        sat_name = sat_dict["name"]
        lower_name = sat_name.lower()

        # ================================
        # NAME VALIDATION CHECK
        # ================================
        if lower_name in seen_names:
            if lower_name == "chief":
                raise ValueError("Multiple satellites named 'chief' (case-insensitive) were found. Only one Chief is allowed.")
            else:
                raise ValueError(f"Duplicate satellite name found: '{sat_name}'. All satellites must have unique names.")
        
        seen_names.add(lower_name)
        # ================================

        att_dict = sat_dict.get("initial_state", {}).get("attitude", {})
        attitude = AttitudeState(
            q_BN=np.array(att_dict.get("q_BN", [1, 0, 0, 0])),
            omega_BN=np.array(att_dict.get("omega_BN", [0, 0, 0]))
        )
        
        init_state = InitialState(
            frame=sat_dict.get("initial_state", {}).get("frame", ""),
            state=np.array(sat_dict.get("initial_state", {}).get("state", [])),
            attitude=attitude
        )
        
        satellites.append(SatelliteConfig(
            name=sat_name,
            initial_state=init_state,
            properties=sat_dict.get("properties", {}),
            gnc=sat_dict.get("gnc", {})
        ))

    # Identify Chief and Deputies
    try:

        chief = next(sat for sat in satellites if sat.name.lower() == "chief")
    except StopIteration:
        raise ValueError("A satellite named 'chief' must be present in the configuration.")
        
    deputies = [sat for sat in satellites if sat.name.lower() != "chief"]

    # Process Chief State
    chief_r, chief_v = _init_chief_state(chief)
    chief_state_dict = {"r": chief_r,
                         "v": chief_v,
                         "mass": chief.properties.get("mass", 500.0),
                         "Cd": chief.properties.get("Cd", 2.2),
                         "area": chief.properties.get("area", 1.0),
                         "inertia_matrix": chief.properties.get("inertia_matrix", np.diag([20, 20, 20]))
                        }
    
    if sim_config.simulation_mode == "6DOF":
        q_c, w_c = _init_attitude(chief)
        chief_state_dict.update({"q_BN": q_c, "omega_BN": w_c})

    # Process Deputy States
    deputies_state_dict = {}
    deputies_properties_dict = {}
    
    for dep in deputies:
        r, v, rho, rho_dot = _init_deputy_state(dep, chief_r, chief_v, chief)
        
        dep_state = {
            "r": r, 
            "v": v, 
            "rho": rho, 
            "rho_dot": rho_dot,
            "mass": dep.properties.get("mass", 500.0),
            "Cd": dep.properties.get("Cd", 2.2),
            "area": dep.properties.get("area", 1.0),
            "inertia_matrix": dep.properties.get("inertia_matrix", np.diag([20, 20, 20]))
        }
        
        if sim_config.simulation_mode == "6DOF":
            q_d, w_d = _init_attitude(dep)
            dep_state.update({"q_BN": q_d, "omega_BN": w_d})
            
        # Store under the deputy's name
        deputies_state_dict[dep.name] = dep_state
        deputies_properties_dict[dep.name] = dep.properties

    # Build Final Outputs
    dynamics_input = {
        "satellite_properties": {
            "chief": chief.properties,
            "deputies": deputies_properties_dict
        },
        "simulation": {
            "propagator": sim_config.propagator,
            "perturbations": getattr(sim_config, "perturbations", {}),
            "simulation_mode": sim_config.simulation_mode
        }  
    }

    init_state = {
        "sim_time": 0.0,
        "epoch": sim_config.epoch,
        "chief": chief_state_dict,
        "deputies": deputies_state_dict
    }

    return {
        "dynamics_input": dynamics_input,
        "init_state": init_state,
        "parsed_satellites": satellites
    }