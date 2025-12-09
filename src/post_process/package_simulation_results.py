import numpy as np

def package_simulation_results(trajectory, gnc_results, t_eval, is_6dof):
    """
    Compiles trajectory and GNC data into a dictionary format compatible 
    with the post_process module.
    """
    # 1. Initialize output dictionary
    post_dict = {
        "time": t_eval.tolist(),
        "full_state": [],
        "control_accel": gnc_results
    }

    # 2. Build the "full_state" vector list
    for res in trajectory:
        # Base 3DOF states (Position, Velocity, Relative State)
        state_parts = [
            res["chief_r"], res["chief_v"],
            res["deputy_r"], res["deputy_v"],
            res["deputy_rho"], res["deputy_rho_dot"],
        ]

        # Append Attitude states if 6DOF
        if is_6dof:
            state_parts.extend([
                res.get("chief_q_BN", np.zeros(4)),
                res.get("chief_omega_BN", np.zeros(3)),
                res.get("deputy_q_BN", np.zeros(4)),
                res.get("deputy_omega_BN", np.zeros(3))
            ])

        # Flatten and store
        post_dict["full_state"].append(np.hstack(state_parts).tolist())

    # 3. Extract 6DOF specific control/error metrics if necessary
    if is_6dof:
        # Use list comprehensions for cleaner extraction
        post_dict["torque_cmd_chief"] = [g.get("torque_cmd_chief", np.zeros(3)) for g in gnc_results]
        post_dict["torque_cmd_deputy"] = [g.get("torque_cmd_deputy", np.zeros(3)) for g in gnc_results]
        post_dict["att_error_deputy"] = [g.get("att_error_deputy", np.zeros(4)) for g in gnc_results]
        post_dict["rate_error_deputy"] = [g.get("rate_error_deputy", np.zeros(3)) for g in gnc_results]

    return post_dict