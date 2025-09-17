import commentjson as json  # instead of import json
import numpy as np

from utils.frame_convertions.rel_to_inertial_functions import rel_vector_to_inertial, LVLH_DCM
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

    # Chief in inertial ECI frame
    chief_r = np.array(chief["initial_state"]["r"])
    chief_v = np.array(chief["initial_state"]["v"])
    
    # Convert deputy to inertial frame if needed
    if "initial_state" in deputy and deputy["initial_state"].get("frame", "").upper() == "ECI":
        deputy_r = np.array(deputy["initial_state"]["r"])
        deputy_v = np.array(deputy["initial_state"]["v"])
        # Compute DCM from inertial to RIC
        C_HN = LVLH_DCM(chief_r, chief_v) 
        deputy_rho = C_HN @ (deputy_r - chief_r)
        omega = np.array([0, 0, np.linalg.norm(np.cross(chief_r, chief_v)) / np.dot(chief_r, chief_r)])
        deputy_rho_dot = C_HN @ (deputy_v - chief_v) - np.cross(omega, deputy_rho)  # assuming zero angular velocity for simplicity
    elif "initial_state" in deputy and deputy["initial_state"].get("frame", "").upper() == "LVLH":
        deputy_rho = np.array(deputy["initial_state"]["r"])
        deputy_rho_dot = np.array(deputy["initial_state"]["v"])
        deputy_r, deputy_v = rel_vector_to_inertial(deputy_rho, deputy_rho_dot, chief_r, chief_v)
    elif "initial_state" in deputy and deputy["initial_state"].get("frame", "").upper() == "RIC":
        # TODO: implement RIC to inertial conversion
        raise ValueError("RIC frame conversion not implemented yet")
    else:
        raise ValueError("Deputy initial state not properly defined")
    
    # Dynamics input: inertial positions/velocities + simulation config
    dynamics_input = {
        "chief_r": chief_r,
        "chief_v": chief_v,
        "deputy_r": deputy_r,
        "deputy_v": deputy_v,
        "deputy_rho": deputy_rho,
        "deputy_rho_dot": deputy_rho_dot,
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
