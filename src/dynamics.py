"""
Dynamics Propagation Module
"""
from src.propagators.CWH_propagator import step_cwh
from src.propagators.TH_propagator import step_th
from src.propagators.two_body_propagator import step_2body
from src.propagators.lin_rel_motion_propagator import step_linearized_2body
from src.propagators.quaternion_integrator import q_step

# 1. Registry: Map config strings to propagator functions
PROPAGATOR_REGISTRY = {
    "CWH": step_cwh,
    "TH": step_th,
    "2BODY": step_2body,
    "LINEARIZED_2BODY": step_linearized_2body,
}

def dyn_step(dt: float, state: dict, config: dict) -> dict:
    """
    Propagate the simulation state one time step forward.

    Handles translational dynamics via the selected propagator and 
    rotational dynamics (if 6DOF) via quaternion integration.

    Parameters
    ----------
    dt : float
        Time step [s]
    state : dict
        Current state dictionary (inertial + relative states)
    config : dict
        Dynamics configuration dictionary

    Returns
    -------
    dict
        Next state dictionary
    """
    # Translational Propagation ---
    # Default to "2BODY" if not specified
    prop_type = config.get("simulation", {}).get("propagator", "2BODY").upper()
    
    propagator_func = PROPAGATOR_REGISTRY.get(prop_type)
    
    if not propagator_func:
        raise ValueError(f"Unknown propagator '{prop_type}'. Available: {list(PROPAGATOR_REGISTRY.keys())}")

    next_state = propagator_func(state, dt, config)

    # Attitude Propagation (6DOF Only) ---
    is_6dof = config.get("simulation", {}).get("simulation_mode", "3DOF").upper() == "6DOF"
    
    if is_6dof:
        # Propagate quaternions and angular velocities
        att_next = q_step(dt, state, config)
        next_state.update(att_next)

    return next_state