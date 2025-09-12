import commentjson as json  # instead of import json
import numpy as np

def parse_input(file_path: str) -> dict:
    """
    Parse the JSONX input file and prepare input dictionaries
    for dynamics, GNC, and postprocessing.
    """
    with open(file_path, "r") as f:
        config = json.load(f) 
    
    # Extract simulation parameters
    sim_config = config.get("simulation", {})
    
    # Extract output parameters
    output_config = config.get("output", {})

    # Extract propagator selection (default to CWH)
    propagator = sim_config.get("propagator", "CWH").upper()
    sim_config["propagator"] = propagator  # store in sim_config for downstream use

    # Process satellites
    satellites = config.get("satellites", [])

    # Separate chief and deputy
    chief = next(sat for sat in satellites if sat["name"].lower() == "chief")
    deputy = next(sat for sat in satellites if sat["name"].lower() == "deputy")

    # Convert chief to RIC if it's in orbit frame
    if "initial_state" in chief and chief["initial_state"].get("frame", "").lower() == "orbit":
        chief_r, chief_v = orbit_to_RIC(chief["initial_state"])
    elif "initial_state_RIC" in chief:
        chief_r = np.array(chief["initial_state_RIC"]["r"])
        chief_v = np.array(chief["initial_state_RIC"]["v"])
    else:
        raise ValueError("Chief initial state not properly defined")
    
    # Deputy relative state in RIC
    if "initial_state_RIC" in deputy:
        deputy_r = np.array(deputy["initial_state_RIC"]["r"])
        deputy_v = np.array(deputy["initial_state_RIC"]["v"])
    else:
        raise ValueError("Deputy initial state not properly defined")
    
    # Dynamics input: absolute positions/velocities + simulation config
    dynamics_input = {
        "chief_r": chief_r,
        "chief_v": chief_v,
        "deputy_r": deputy_r,
        "deputy_v": deputy_v,
        "satellite_properties": {
            "chief": chief.get("properties", {}),
            "deputy": deputy.get("properties", {})
        },
        "simulation": sim_config  # includes propagator
    }

    # GNC input: same as dynamics, plus any GNC-specific params
    gnc_input = {
        "trajectory": None,  # Will be filled after dynamics run
        "satellites": {
            "chief": chief,
            "deputy": deputy
        },
        "simulation": sim_config,
        "output": output_config
    }

    # Postprocess input: trajectory and GNC results
    postprocess_input = {
        "trajectory_file": output_config.get("trajectory_file", "data/results/trajectory.csv"),
        "gnc_file": output_config.get("gnc_file", "data/results/gnc_results.csv"),
        "plots": output_config.get("plots", True),
        "propagator": propagator  # pass propagator to postprocess
    }

    # Return all in a single config dict
    config = {
        "dynamics": dynamics_input,
        "gnc": gnc_input,
        "postprocess": postprocess_input,
        "raw": config
    }

    return config

# -------------------------
# Example orbit-to-RIC converter
def orbit_to_RIC(orbit_state: dict):
    """
    Convert chief's orbit-frame state to RIC frame reference for deputy.
    This is a stub; replace with actual transformation.
    """
    r = np.array(orbit_state["r"])  # [km]
    v = np.array(orbit_state["v"])  # [km/s]
    # TODO: implement actual orbit -> RIC conversion if needed
    return r, v
