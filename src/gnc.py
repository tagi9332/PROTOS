import numpy as np
from src.controllers.cartesian_control import cartesian_step
from src.controllers.mean_oe_control import mean_oe_step
from src.controllers.quiz_2_controller import quiz_2_step
from src.controllers.quiz_3_controller import quiz_3_step

def step(state: dict, config: dict) -> dict:
    """
    Main GNC step. Chooses control mode and executes corresponding step.
    
    state: dictionary with chief/deputy inertial and relative states
    config: dictionary containing GNC configuration
    
    Returns updated state dictionary including command acceleration.
    """

    control_method = config.get("control", {}).get("control_method", "").upper()

    if control_method == "MEAN_OES":  
        return mean_oe_step(state, config)       
    elif control_method == "QUIZ_2":     # Specific controller to complete Quiz 2
        return quiz_2_step(state, config)
    elif control_method == "QUIZ_3":     # Specific controller to complete Quiz 3
        return quiz_3_step(state, config)
    elif control_method == "CARTESIAN":
        return cartesian_step(state, config)


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
