"""
Dynamics Propagation Module
"""
import copy
import numpy as np

from utils.frame_conversions.rel_to_inertial_functions import inertial_to_rel_LVLH
from src.propagators import (
    step_cwh,
    step_th,
    step_2body,
    q_step
)

# Map config strings to propagator functions
PROPAGATOR_REGISTRY = {
    "CWH": step_cwh,
    "TH": step_th,
    "2BODY": step_2body,
}

def dyn_step(dt: float, state: dict, config: dict) -> dict:
    """
    Propagate the simulation state one time step forward for all satellites.
    """
    # Setup
    sim_config = config.get("simulation", {})
    prop_type = getattr(sim_config, "propagator", "2BODY").upper()
    is_6dof = getattr(sim_config, "simulation_mode", "3DOF").upper() == "6DOF"

    propagator_func = PROPAGATOR_REGISTRY.get(prop_type)
    if not propagator_func:
        raise ValueError(f"Unknown propagator '{prop_type}'. Available: {list(PROPAGATOR_REGISTRY.keys())}")

    next_state = copy.deepcopy(state)
    
    # Initialize sim_time if not present
    next_state["sim_time"] = next_state.get("sim_time", 0.0) + dt


    # PROPAGATE CHIEF
    chief_next = propagator_func(state["chief"], dt, config, is_chief=True)
    next_state["chief"].update(chief_next)
    
    # Propagate Chief Attitude (6DOF)
    if is_6dof:
        chief_att_next = q_step(dt, state["chief"], config)
        next_state["chief"].update(chief_att_next)

    # PROPAGATE DEPUTIES
    for sat_name, sat_data in state["deputies"].items():
            
        # Translation
        dep_next = propagator_func(sat_data, dt, config, chief_state=state["chief"])
        next_state["deputies"][sat_name].update(dep_next)

        # Attitude Propagation (6DOF)
        if is_6dof:
            att_next = q_step(dt, sat_data, config)
            next_state["deputies"][sat_name].update(att_next)

    # UPDATE RELATIVE STATES
    r_c = np.array(next_state["chief"]["r"])
    v_c = np.array(next_state["chief"]["v"])
    
    for sat_name, sat_data in next_state["deputies"].items():
        r_d = np.array(sat_data["r"])
        v_d = np.array(sat_data["v"])
        
        # Recompute LVLH relative state based on the newly propagated inertial vectors
        rho, rho_dot = inertial_to_rel_LVLH(r_d, v_d, r_c, v_c)
        
        next_state["deputies"][sat_name]["rho"] = rho.tolist()
        next_state["deputies"][sat_name]["rho_dot"] = rho_dot.tolist()

    return next_state