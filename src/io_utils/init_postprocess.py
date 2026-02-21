from dataclasses import dataclass, field
from typing import Dict, Any

from src.io_utils.init_sim_config import OutputConfig

def init_postprocess(output_config: 'OutputConfig') -> dict:
    postprocess_input = {
        "trajectory_file": output_config.trajectory_file,
        "gnc_file": output_config.gnc_file,
        "plots": output_config.plots,
    }

    return postprocess_input