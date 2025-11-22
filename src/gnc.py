from src.controllers.cartesian_controller import cartesian_step
from src.controllers.mean_oe_controller import mean_oe_step
from src.controllers.quiz_2_controller import quiz_2_step
from src.controllers.quiz_3_controller import quiz_3_step
from src.controllers.quiz_8_controller import quiz_8_step
from src.controllers.quiz_10_controller import quiz_10_step

def step(state: dict, config: dict) -> dict:
    """
    Main GNC step. Chooses control mode and executes corresponding step.
    
    state: dictionary with chief/deputy inertial and relative states
    config: dictionary containing GNC configuration
    
    Returns updated state dictionary including command acceleration.
    """
    # If guidance is not RPO, pass through
    guidance_type = config.get("guidance", {}).get("type", "").upper()

    if guidance_type == "RPO":
        # Select controller based on configuration
        control_method = config.get("control", {}).get("control_method", "").upper()

        if control_method == "MEAN_OES":  
            return mean_oe_step(state, config)       
        elif control_method == "QUIZ_2":     # Specific controller to complete Quiz 2
            return quiz_2_step(state, config)
        elif control_method == "QUIZ_3":     # Specific controller to complete Quiz 3
            return quiz_3_step(state, config)
        elif control_method == "QUIZ_8":     # Specific controller to complete Quiz 8
            return quiz_8_step(state, config)
        elif control_method == "QUIZ_10":     # Specific controller to complete Quiz 10
            return quiz_10_step(state, config)
        elif control_method == "CARTESIAN":
            return cartesian_step(state, config)
        else:
            raise ValueError(f"Control method '{control_method}' not recognized for RPO guidance.")

        # Enforce saturation limits if specified in config
        # **TODO**
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
