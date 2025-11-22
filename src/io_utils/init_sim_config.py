import numpy as np
import dateutil.parser

def init_sim_config(raw_config: dict) -> dict:
    # Extract simulation parameters
    sim_config = raw_config.get("simulation", {})

    # Simulation time vector 
    dt = sim_config.get("time_step", 1)
    duration = sim_config.get("duration", 3600.0)
    steps = int(duration / dt) + 1
    t_eval = np.linspace(0, duration, steps)

    # Parse epoch if present
    epoch_str = sim_config.get("epoch", None)
    if epoch_str is not None:
        sim_config["epoch"] = dateutil.parser.isoparse(epoch_str)
    else:
        # Default to J2000: 2000-01-01 12:00:00 UTC
        sim_config["epoch"] = dateutil.parser.isoparse("2000-01-01T12:00:00Z")

    # Extract output parameters
    output_config = raw_config.get("output", {})

    # Extract propagator selection (default to 2BODY)
    propagator = sim_config.get("propagator", "2BODY").upper()
    sim_config["propagator"] = propagator

    return {
        "output": output_config,
        "simulation": sim_config,
        "t_eval": t_eval
    }