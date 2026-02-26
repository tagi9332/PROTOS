import numpy as np

from src.controllers.attitude_control.target_pointing_controller import target_pointing_controller

def attitude_step(state: dict, config: dict, sat_name: str) -> dict:
    """
    Choose attitude controller based on input and execute control step. 
    Returns commanded torque and error states for the specific satellite.
    """
    # Safely get attitude guidance mode
    guidance_dict = config.get('guidance', {})
    attitude_guidance = guidance_dict.get('attitude_guidance', {})
    attitude_guidance_mode = attitude_guidance.get('type', 'NONE').upper()

    if attitude_guidance_mode == "NONE":
        # Return a zeroed-out command
        return {
            "torque_cmd": [0.0, 0.0, 0.0],
            "att_error": [1.0, 0.0, 0.0, 0.0], # Scalar-first quaternion for zero error
            "rate_error": [0.0, 0.0, 0.0]
        }
    elif attitude_guidance_mode == "INERTIAL_POINTING":
        raise NotImplementedError(f"Inertial pointing not yet implemented for {sat_name}.")
    elif attitude_guidance_mode == "LVLH_POINTING":
        raise NotImplementedError(f"LVLH pointing not yet implemented for {sat_name}.")
    elif attitude_guidance_mode == "TARGET_POINTING":
        res = target_pointing_controller(state, config, sat_name)
    else:
        raise ValueError(f"Attitude guidance mode '{attitude_guidance_mode}' not recognized for {sat_name}.")

    # Apply Torque Saturation
    max_torque = config.get('control', {}).get('max_torque')

    if max_torque is not None:
        res["torque_cmd"] = np.clip(res["torque_cmd"], -max_torque, max_torque).tolist()

    return res