import numpy as np

def update_state_with_gnc(state, gnc_out, is_6dof):
    """Updates the state dictionary with control commands."""
    state["control_accel"] = gnc_out.get("accel_cmd", np.zeros(3))

    if is_6dof:
        # You can list them explicitly or use a loop here
        state["torque_cmd_chief"] = gnc_out.get("torque_cmd_chief", np.zeros(3))
        state["torque_cmd_deputy"] = gnc_out.get("torque_cmd_deputy", np.zeros(3))
        state["att_error_deputy"] = gnc_out.get("att_error_deputy", np.zeros(4))
        state["rate_error_deputy"] = gnc_out.get("rate_error_deputy", np.zeros(3))