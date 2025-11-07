import numpy as np
from src.controllers.lvlh_control import lvlh_step
from src.controllers.oe_control import doe_step

def step(state: dict, config: dict) -> dict:
    """
    Main GNC step. Chooses control mode and executes corresponding step.
    
    state: dictionary with chief/deputy inertial and relative states
    config: dictionary containing GNC configuration
    
    Returns updated state dictionary including command acceleration.
    """

    guidance_type = config.get("control", {}).get("control_method", "").upper()

    if guidance_type == "LVLH":      
        if guidance_type == "OES":
            return oe_step(state, config)
        else:
            # Default to LVLH
            return lvlh_step(state, config)

    else:
        # No guidance; pass-through
        return {
            "status": "pass-through",
            "chief_r": state["chief_r"].tolist(),
            "chief_v": state["chief_v"].tolist(),
            "deputy_r": state["deputy_r"].tolist(),
            "deputy_v": state["deputy_v"].tolist(),
            "deputy_rho": state["deputy_rho"].tolist(),
            "deputy_rho_dot": state["deputy_rho_dot"].tolist(),
            "accel_cmd": [0,0,0]
        }
