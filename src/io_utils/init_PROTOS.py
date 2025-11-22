import commentjson as json
from src.io_utils.init_gnc import init_gnc
from src.io_utils.init_postprocess import init_postprocess
from src.io_utils.init_satellites import init_satellites
from src.io_utils.init_sim_config import init_sim_config


def parse_input(file_path: str) -> dict:
    """
    Parse the JSONX input file and prepare input dictionaries
    for dynamics, GNC, and postprocessing.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f) 

    config = init_sim_config(raw_config)
    sim_config = config["simulation"]
    output_config = config["output"]

    # -----------------------
    # Dynamics Section Handling
    # -----------------------
    satellites = init_satellites(raw_config, sim_config)
    dynamics_input = satellites["dynamics_input"]
    
    # -----------------------
    # GNC Initialization
    # -----------------------
    gnc_input = init_gnc(raw_config)

    # -----------------------
    # Postprocessing
    # -----------------------
    postprocess_input = init_postprocess(output_config)

    return {
        "simulation": sim_config,
        "dynamics": dynamics_input,
        "gnc": gnc_input,
        "postprocess": postprocess_input,
        "raw": raw_config
    }