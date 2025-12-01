from src.controllers.attitude_control.target_pointing_controller import target_pointing_controller

def attitude_step(state: dict, config: dict) -> dict:
    """
    Choose attitude controller based on input and execute control step. Returns commanded torque for chief and deputy
    """
    # Get attitude guidance mode; options: INERTIAL_POINTING, LVLH_POINTING, TARGET_POINTING
    attitude_guidance_mode = config['guidance']['attitude_guidance']['type'].upper()
    if attitude_guidance_mode == "NONE":
        return {
            "torque_chief": [0.0, 0.0, 0.0],
            "torque_deputy": [0.0, 0.0, 0.0]
        }
    elif attitude_guidance_mode == "INERTIAL_POINTING":
        # Not yet implemented
        raise NotImplementedError("Inertial pointing attitude control not yet implemented.")
    elif attitude_guidance_mode == "LVLH_POINTING":
        # Not yet implemented
        raise NotImplementedError("LVLH pointing attitude control not yet implemented.")
    elif attitude_guidance_mode == "TARGET_POINTING":
        return target_pointing_controller(state, config)
    else:
        raise ValueError(f"Attitude guidance mode '{attitude_guidance_mode}' not recognized.")
