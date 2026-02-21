import commentjson as json
from src.io_utils.init_gnc import init_gnc
from src.io_utils.init_postprocess import init_postprocess
from src.io_utils.init_satellites import init_satellites
from src.io_utils.init_sim_config import init_sim_config


def init_PROTOS(file_path: str) -> dict:
    """
    Parse the JSONX input file and prepare input dictionaries
    for dynamics, GNC, and postprocessing.
    """
    print(f"Initializing PROTOS with config file: {file_path}")

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

    # Attitude dynamics step check
    if sim_config.simulation_mode.upper() == "6DOF" and sim_config.time_step > 0.1:
        print(f"WARNING: Simulation is in 6DOF mode with a timestep of {sim_config.time_step}s. Large steps with ZOH torque can cause instability. Consider reducing the timestep for better accuracy.")    

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