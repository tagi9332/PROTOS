import numpy as np

def package_simulation_results(trajectory, gnc_results, t_eval, is_6dof):
    """
    Compiles trajectory and GNC data into a structured time-history dictionary 
    compatible with N-satellite post-processing.
    """
    # 1. Initialize output dictionary
    post_dict = {
        "time": t_eval.tolist(),
        "is_6dof": is_6dof,
        "chief": {
            "r": [], "v": [], "accel_cmd": [], 
            "q": [], "omega": [], "torque_cmd": []
        },
        "deputies": {}
    }
    
    # Pre-initialize deputy dictionaries based on the first state
    first_state = trajectory[0]
    for sat_name in first_state["deputies"]:
        post_dict["deputies"][sat_name] = {
            "r": [], "v": [], "rho": [], "rho_dot": [], "accel_cmd": [],
            "q": [], "omega": [], "torque_cmd": [], "att_error": [], "rate_error": []
        }

    # 2. Extract Data over Time
    for i, state in enumerate(trajectory):
        # Safely grab the corresponding GNC command for this timestep
        gnc_out = gnc_results[i] if i < len(gnc_results) else {}
        
        # Grab Chief directly
        gnc_chief = gnc_out.get("chief", {})

        # --- Chief Data ---
        post_dict["chief"]["r"].append(state["chief"].get("r", [0.0, 0.0, 0.0]))
        post_dict["chief"]["v"].append(state["chief"].get("v", [0.0, 0.0, 0.0]))
        post_dict["chief"]["accel_cmd"].append(gnc_chief.get("accel_cmd", [0.0, 0.0, 0.0]))
        
        if is_6dof:
            post_dict["chief"]["q"].append(state["chief"].get("q_BN", [0.0, 0.0, 0.0, 1.0]))
            post_dict["chief"]["omega"].append(state["chief"].get("omega_BN", [0.0, 0.0, 0.0]))
            post_dict["chief"]["torque_cmd"].append(gnc_chief.get("torque_cmd", [0.0, 0.0, 0.0]))

        # --- Deputies Data ---
        for sat_name, sat_data in state["deputies"].items():
            dep_out = post_dict["deputies"][sat_name]
            
            # FIXED: Grab this specific deputy directly from the root gnc_out dict
            dep_gnc = gnc_out.get(sat_name, {})

            dep_out["r"].append(sat_data.get("r", [0.0, 0.0, 0.0]))
            dep_out["v"].append(sat_data.get("v", [0.0, 0.0, 0.0]))
            dep_out["rho"].append(sat_data.get("rho", [0.0, 0.0, 0.0]))
            dep_out["rho_dot"].append(sat_data.get("rho_dot", [0.0, 0.0, 0.0]))
            dep_out["accel_cmd"].append(dep_gnc.get("accel_cmd", [0.0, 0.0, 0.0]))

            if is_6dof:
                dep_out["q"].append(sat_data.get("q_BN", [0.0, 0.0, 0.0, 1.0]))
                dep_out["omega"].append(sat_data.get("omega_BN", [0.0, 0.0, 0.0]))
                dep_out["torque_cmd"].append(dep_gnc.get("torque_cmd", [0.0, 0.0, 0.0]))
                dep_out["att_error"].append(dep_gnc.get("att_error", [0.0, 0.0, 0.0, 1.0]))
                dep_out["rate_error"].append(dep_gnc.get("rate_error", [0.0, 0.0, 0.0]))

    # 3. Clean up empty 6DOF lists if we ran in 3DOF to save memory/confusion
    if not is_6dof:
        del post_dict["chief"]["q"], post_dict["chief"]["omega"], post_dict["chief"]["torque_cmd"]
        for sat_name in post_dict["deputies"]:
            del (post_dict["deputies"][sat_name]["q"], 
                 post_dict["deputies"][sat_name]["omega"], 
                 post_dict["deputies"][sat_name]["torque_cmd"], 
                 post_dict["deputies"][sat_name]["att_error"], 
                 post_dict["deputies"][sat_name]["rate_error"])

    # Convert everything to numpy arrays for easy math/plotting downstream
    post_dict["time"] = np.array(post_dict["time"])
    for key in post_dict["chief"]:
        post_dict["chief"][key] = np.array(post_dict["chief"][key])
    for sat_name in post_dict["deputies"]:
        for key in post_dict["deputies"][sat_name]:
            post_dict["deputies"][sat_name][key] = np.array(post_dict["deputies"][sat_name][key])

    return post_dict