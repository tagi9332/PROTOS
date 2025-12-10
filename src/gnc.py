"""
Guidance, Navigation, and Control (GNC) Module
"""
import numpy as np
from src.controllers.cwh_controller import cwh_step
from src.controllers.cartesian_controller import cartesian_step
from src.controllers.mean_oe_controller import mean_oe_step
from src.controllers.attitude_control.attitude_controller import attitude_step

# Map config strings to functions
TRANSLATIONAL_CONTROLLERS = {
    "CWH": cwh_step,
    "CARTESIAN": cartesian_step,
    "MEAN_OES": mean_oe_step,
}

def gnc_step(state: dict, config: dict) -> dict:
    """
    Main GNC step. Dispatches to specific translational and attitude controllers.

    Returns:
        dict: A dictionary containing 'accel_cmd' and optional 6DOF torque/error commands.
    """
    # Initialize output
    gnc_out = {
        "accel_cmd": np.zeros(3)
    }

    # Translational Guidance ---
    guidance_type = config.get("guidance", {}).get("type", "").upper()
    
    if guidance_type == "RPO":
        control_method = config.get("control", {}).get("control_method", "").upper()
        
        # Look up the controller in the registry
        control_func = TRANSLATIONAL_CONTROLLERS.get(control_method)
        
        if not control_func:
            raise ValueError(f"Control method '{control_method}' not recognized for RPO guidance.")
            
        # Execute control and merge results
        trans_out = control_func(state, config)
        gnc_out["accel_cmd"] = trans_out.get("accel_cmd", np.zeros(3))

    # Attitude Control (6DOF) ---
    is_6dof = config.get("simulation_mode", "3DOF").upper() == "6DOF"
    
    if is_6dof:
        # Execute attitude control
        att_cmds = attitude_step(state, config)

        # Merge attitude control results
        gnc_out.update({
            "torque_cmd_chief": att_cmds.get("torque_chief", np.zeros(3)),
            "torque_cmd_deputy": att_cmds.get("torque_deputy", np.zeros(3)),
            "att_error_deputy": att_cmds.get("att_error_deputy", np.zeros(4)),
            "rate_error_deputy": att_cmds.get("rate_error_deputy", np.zeros(3))
        })

    return gnc_out