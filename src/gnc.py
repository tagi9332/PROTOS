"""
Guidance, Navigation, and Control (GNC) Module
"""
import numpy as np
from src.controllers import (
    cwh_step,
    cartesian_step,
    mean_oe_step,
    attitude_step
)

# Map config strings to functions
TRANSLATIONAL_CONTROLLERS = {
    "CWH": cwh_step,
    "CARTESIAN": cartesian_step,
    "MEAN_OES": mean_oe_step,
}

def gnc_step(state: dict, full_gnc_config: dict) -> dict:
    """
    Main GNC step. Iterates over all satellites and dispatches to the appropriate guidance and control functions based on the configuration.

    Returns:
        dict: A dictionary keyed by satellite name containing 'accel_cmd' and optional 6DOF torque/error commands.
    """
    # Initialize master output dictionary
    master_gnc_out = {}

    # Loop through every satellite that has a GNC configuration defined
    for sat_name, sat_config in full_gnc_config.items():
        
        # Initialize the output for THIS specific satellite
        sat_cmds = {
            "accel_cmd": np.zeros(3)
        }

        # Translational Guidance ---
        guidance_type = sat_config.get("guidance", {}).get("type", "").upper()
        
        if guidance_type == "RPO":
            control_method = sat_config.get("control", {}).get("control_method", "").upper()
            
            # Look up the controller in the registry
            control_func = TRANSLATIONAL_CONTROLLERS.get(control_method)
            
            if not control_func:
                raise ValueError(f"Control method '{control_method}' not recognized for RPO guidance on {sat_name}.")
                
            # Execute control and merge results
            # NOTE: We now pass sat_name so the controller knows which state to pull!
            trans_out = control_func(state, sat_config, sat_name)
            sat_cmds["accel_cmd"] = trans_out.get("accel_cmd", np.zeros(3))

        # Attitude Control (6DOF) ---
        is_6dof = sat_config.get("simulation_mode", "3DOF").upper() == "6DOF"
        
        if is_6dof:
            # Execute attitude control (also needs to know which satellite!)
            att_cmds = attitude_step(state, sat_config, sat_name)

            # Merge attitude control results for this satellite
            sat_cmds.update({
                "torque_cmd": att_cmds.get("torque_cmd", np.zeros(3)),
                "att_error": att_cmds.get("att_error", np.zeros(4)),
                "rate_error": att_cmds.get("rate_error", np.zeros(3))
            })
            
        # Store this satellite's commands in the master dictionary
        master_gnc_out[sat_name] = sat_cmds

    return master_gnc_out