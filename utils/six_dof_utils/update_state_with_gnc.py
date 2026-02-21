import numpy as np

def update_state_with_gnc(state: dict, gnc_out: dict, is_6dof: bool):
    """
    Updates the master state dictionary with control commands for all satellites.
    """
    # Handle Chief (if present)
    if "chief" in gnc_out:
        state["chief"]["accel_cmd"] = np.array(gnc_out["chief"].get("accel_cmd", [0, 0, 0]))
        if is_6dof:
            state["chief"]["torque_cmd"] = np.array(gnc_out["chief"].get("torque_cmd", [0, 0, 0]))
    
    # Handle all Deputies
    if "deputies" in gnc_out:
            for sat_name, cmds in gnc_out["deputies"].items():
                
                # Safety check: Ensure the commanded deputy actually exists in the physics state
                if sat_name in state["deputies"]:
                    
                    # Translation
                    state["deputies"][sat_name]["accel_cmd"] = np.array(cmds.get("accel_cmd", [0.0, 0.0, 0.0]))
                    
                    # Attitude (6DOF)
                    if is_6dof:
                        state["deputies"][sat_name]["torque_cmd"] = np.array(cmds.get("torque_cmd", [0.0, 0.0, 0.0]))
                        
                        # Store tracking errors for telemetry/plotting
                        state["deputies"][sat_name]["att_error"]  = np.array(cmds.get("att_error", [0.0, 0.0, 0.0, 1.0]))
                        state["deputies"][sat_name]["rate_error"] = np.array(cmds.get("rate_error", [0.0, 0.0, 0.0]))