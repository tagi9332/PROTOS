import commentjson as json

from src.io_utils.init_gnc import init_gnc
from src.io_utils.init_postprocess import init_postprocess
from src.io_utils.init_satellites import init_satellites
from src.io_utils.init_sim_config import init_sim_config
from utils.print_functions.print_sim_header import print_sim_header


def init_PROTOS(file_path: str) -> dict:
    """
    Parse the JSONX input file and prepare input dictionaries
    for dynamics, GNC, and postprocessing.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f) 

    config = init_sim_config(raw_config)
    sim_config = config.simulation
    output_config = config.output

    # -----------------------
    # Dynamics Section Handling
    # -----------------------
    dyn_config = init_satellites(raw_config, sim_config) 
    init_state = dyn_config["init_state"]
    parsed_satellites = dyn_config["parsed_satellites"]

    # Print the simulation header
    print_sim_header(file_path, sim_config, init_state)

    # Attitude dynamics step check
    if sim_config.simulation_mode.upper() == "6DOF" and sim_config.time_step > 0.1:
        yellow_start = "\033[33m"
        color_reset = "\033[0m"
        
        message = (f"WARNING: Simulation is in 6DOF mode with a timestep of {sim_config.time_step}s. "
                   f"Large steps with ZOH torque can cause instability. "
                   f"Consider reducing the timestep for better accuracy.")
        
        print(f"{yellow_start}{message}{color_reset}")

    # -----------------------
    # GNC Initialization
    # -----------------------
    gnc_input = init_gnc(parsed_satellites, sim_config)

    # -----------------------
    # Postprocessing
    # -----------------------
    postprocess_input = init_postprocess(output_config)

    return {
        "simulation": sim_config,
        "t_eval": sim_config.t_eval,
        "dynamics": dyn_config.get("dynamics_input", {}),
        "init_state": init_state,
        "gnc": gnc_input,
        "postprocess": postprocess_input,
        "raw": raw_config
    }