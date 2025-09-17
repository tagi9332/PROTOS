import commentjson as json
import numpy as np

def parse_input(file_path: str) -> dict:
    """
    Parse the JSONX input file and prepare input dictionaries
    for dynamics, GNC, and postprocessing.
    """
    with open(file_path, "r") as f:
        raw_config = json.load(f)
    
    # Extract simulation parameters
    sim_config = raw_config.get("simulation", {})
    
    # Extract output parameters
    output_config = raw_config.get("output", {})

    # Extract propagator selection (default to CWH)
    propagator = sim_config.get("propagator", "CWH").upper()
    sim_config["propagator"] = propagator  # store in sim_config

    # Process satellites
    satellites = raw_config.get("satellites", [])
    chief = next(sat for sat in satellites if sat["name"].lower() == "chief")
    deputy = next(sat for sat in satellites if sat["name"].lower() == "deputy")

    # Chief inertial state
    if "initial_state" in chief and chief["initial_state"].get("frame", "").lower() == "eci":
        chief_r = np.array(chief["initial_state"]["r"])
        chief_v = np.array(chief["initial_state"]["v"])
    elif "initial_state" in chief and chief["initial_state"].get("frame", "").lower() == "orbit":
        chief_r, chief_v = orbit_to_RIC(chief["initial_state"])
    elif "initial_state_RIC" in chief:
        chief_r = np.array(chief["initial_state_RIC"]["r"])
        chief_v = np.array(chief["initial_state_RIC"]["v"])
    else:
        raise ValueError("Chief initial state not properly defined")

    
    # Deputy inertial state
    if "initial_state_RIC" in deputy:
        deputy_r = np.array(deputy["initial_state_RIC"]["r"])
        deputy_v = np.array(deputy["initial_state_RIC"]["v"])
    else:
        raise ValueError("Deputy initial state not properly defined")
    
    # Initialize LVLH deputy state to zero (computed during first step)
    deputy_r_LVLH = np.zeros(3)
    deputy_v_LVLH = np.zeros(3)

    # -------------------------
    # Dynamics input
    # -------------------------
    dynamics_input = {
        "chief_r": chief_r,
        "chief_v": chief_v,
        "deputy_r": deputy_r,
        "deputy_v": deputy_v,
        "deputy_r_LVLH": deputy_r_LVLH,
        "deputy_v_LVLH": deputy_v_LVLH,
        "satellite_properties": {
            "chief": chief.get("properties", {}),
            "deputy": deputy.get("properties", {})
        },
        "simulation": sim_config
    }

    # -------------------------
    # GNC input
    # -------------------------
    gnc_input = {
        "trajectory": None,  # filled after dynamics run
        "satellites": {
            "chief": chief,
            "deputy": deputy
        },
        "simulation": sim_config,
        "output": output_config
    }

    # -------------------------
    # Postprocess input
    # -------------------------
    postprocess_input = {
        "trajectory_file": output_config.get("trajectory_file", "data/results/trajectory.csv"),
        "gnc_file": output_config.get("gnc_file", "data/results/gnc_results.csv"),
        "plots": output_config.get("plots", True),
        "propagator": propagator
    }

    return {
        "dynamics": dynamics_input,
        "gnc": gnc_input,
        "postprocess": postprocess_input,
        "raw": raw_config
    }

# -------------------------
# Example orbit-to-RIC converter
# -------------------------
def orbit_to_RIC(orbit_state: dict):
    """
    Convert chief's orbit-frame state to RIC frame reference.
    Currently returns inertial ECI for chief; deputy LVLH will be computed in dynamics.
    """
    r = np.array(orbit_state["r"])
    v = np.array(orbit_state["v"])
    return r, v
