from src.controllers.cwh_controller import cwh_step
from src.controllers.cartesian_controller import cartesian_step
from src.controllers.mean_oe_controller import mean_oe_step
from src.controllers.quiz_2_controller import quiz_2_step
from src.controllers.quiz_3_controller import quiz_3_step
from src.controllers.quiz_8_controller import quiz_8_step
from src.controllers.quiz_10_controller import quiz_10_step
# Import your attitude controller here
from src.controllers.attitude_control.attitude_controller import attitude_step   

def gnc_step(state: dict, config: dict) -> dict:
    """
    Main GNC step. Chooses control mode and executes corresponding step.
    
    Handles both translational control (acceleration) and, if in 6DOF mode,
    attitude control (torque commands).

    Parameters
    ----------
    state : dict
        Dictionary with chief/deputy inertial and relative states
    config : dict
        Dictionary containing GNC configuration
    
    Returns
    -------
    dict
        Updated state dictionary including:
            - accel_cmd: commanded acceleration in inertial frame
            - torque_cmd_chief: commanded torque for chief (if 6DOF)
            - torque_cmd_deputy: commanded torque for deputy (if 6DOF)
    """
    guidance_type = config.get("guidance", {}).get("type", "").upper()
    is_6dof = config.get("simulation_mode", "3DOF").upper() == "6DOF"

    # -------------------------------
    # Translational Guidance
    # -------------------------------
    accel_cmd = [0, 0, 0]  # default if no guidance
    if guidance_type == "RPO":
        control_method = config.get("control", {}).get("control_method", "").upper()

        if control_method == "MEAN_OES":  
            state_out = mean_oe_step(state, config)
        elif control_method == "QUIZ_2":
            state_out = quiz_2_step(state, config)
        elif control_method == "QUIZ_3":
            state_out = quiz_3_step(state, config)
        elif control_method == "QUIZ_8":
            state_out = quiz_8_step(state, config)
        elif control_method == "QUIZ_10":
            state_out = quiz_10_step(state, config)
        elif control_method == "CARTESIAN":
            state_out = cartesian_step(state, config)
        elif control_method == "CWH":
            state_out = cwh_step(state, config)
        else:
            raise ValueError(f"Control method '{control_method}' not recognized for RPO guidance.")

        accel_cmd = state_out.get("accel_cmd", [0,0,0])
    else:
        # No guidance, default pass-through
        state_out = {}

    # -------------------------------
    # Attitude Control (6DOF)
    # -------------------------------
    torque_cmd_chief = [0, 0, 0]
    torque_cmd_deputy = [0, 0, 0]
    error_att = [0, 0, 0, 0]
    omega_BN = [0, 0, 0]

    if is_6dof:
        # Call attitude controller step for chief & deputy
        # attitude_step should return a dict with keys: torque_chief, torque_deputy
        att_cmds = attitude_step(state, config)
        torque_cmd_chief = att_cmds.get("torque_chief", [0,0,0])
        torque_cmd_deputy = att_cmds.get("torque_deputy", [0,0,0])
        error_att = att_cmds.get("att_error_deputy", [0,0,0,0])
        omega_BN = att_cmds.get("rate_error_deputy", [0,0,0])

    # -------------------------------
    # Return combined command dictionary
    # -------------------------------
    gnc_out_dict = {
        "accel_cmd": accel_cmd
    }

    if is_6dof:
        gnc_out_dict.update({
            "torque_cmd_chief": torque_cmd_chief,
            "torque_cmd_deputy": torque_cmd_deputy,
            "att_error_deputy": error_att, 
            "rate_error_deputy": omega_BN
        })

    return gnc_out_dict
