from src.propagators.CWH_propagator import step_cwh
from src.propagators.TH_propagator import step_th
from src.propagators.two_body_propagator import step_2body
from src.propagators.lin_rel_motion_propagator import step_linearized_2body
from src.propagators.quaternion_integrator import q_step

# -------------------------------
# Main step function for time-stepping
# -------------------------------
def dyn_step(dt: float, state: dict, config: dict):
    """
    Propagate the deputy state one time step using the selected scheme.
    
    state: 'chief_r', 'chief_v', 'deputy_r', 'deputy_v', 'deputy_rho', 'deputy_rho_dot'
    dt: timestep [s]
    config: dict from io_utils.parse_input["dynamics"]
    """
    propagator = config.get("simulation", {}).get("propagator", "2BODY").upper() # default to full nonlinear progagation
    if propagator == "CWH":
        next_state = step_cwh(state, dt, config)
    elif propagator == "TH":
        next_state = step_th(state, dt, config)
    elif propagator == "2BODY":
        next_state = step_2body(state, dt, config)
    elif propagator == "LINEARIZED_2BODY":
        next_state = step_linearized_2body(state, dt, config)
    else:
        raise ValueError(f"Unknown propagator '{propagator}'")
    
    # Attitude propagation (6DOF only)
    if config.get("simulation", {}).get("simulation_mode", "3DOF").upper() == "6DOF":
        attitude_state = q_step(dt, state, config)

        next_state.update(attitude_state)

    return next_state
