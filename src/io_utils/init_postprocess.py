from dataclasses import dataclass, field
from typing import Dict, Any

from src.io_utils.init_sim_config import OutputConfig

def init_postprocess(output_config: 'OutputConfig') -> dict:
    postprocess_input = {
        "save_gnc_results": output_config.save_gnc_results,
        "save_orbit_elements": output_config.save_orbit_elements,
        "animate_trajectories": output_config.animate_trajectory,
        "plots": output_config.plots,
    }

    return postprocess_input