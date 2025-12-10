"""
Main entry point for PROTOS simulation framework.
"""
from datetime import timedelta
import numpy as np

from src import dynamics, gnc
from src.io_utils import init_PROTOS
from src.post_process import post_process, package_simulation_results
from utils.six_dof_utils import update_state_with_gnc

def main():
    # --- Initialization ---
    config = init_PROTOS.parse_input("data/input_files/project_task_1.jsonx")
    sim_config = config["simulation"]
    dyn_config = config["dynamics"]
    gnc_config = config["gnc"]

    dt = sim_config.get("time_step", 1)
    t_eval = np.array(config.get("t_eval"))
    initial_epoch = sim_config.get("epoch")
    is_6dof = sim_config.get("simulation_mode", "3DOF").upper() == "6DOF"

    state = config.get("init_state", {})
    trajectory = [state.copy()]
    gnc_results = []

    # --- Propagation Loop ---
    for _ in t_eval[:-1]:
        # GNC Step
        gnc_out = gnc.gnc_step(state, gnc_config)
        gnc_results.append(gnc_out)

        # Update State with GNC Outputs
        update_state_with_gnc(state, gnc_out, is_6dof)

        # Dynamics Step
        next_state = dynamics.dyn_step(dt, state, dyn_config)

        # Update Time & Epoch
        next_state["sim_time"] = np.array(state.get("sim_time", 0.0) + dt, dtype=np.float64)
        next_state["epoch"] = state.get("epoch", initial_epoch) + timedelta(seconds=dt) # type: ignore

        # Store & Advance
        trajectory.append(next_state)
        state = next_state

    # --- Finalize ---
    final_gnc = gnc.gnc_step(state, gnc_config)
    final_gnc["control_accel"] = final_gnc.get("accel_cmd", np.zeros(3))
    gnc_results.append(final_gnc)

    # --- Post Processing ---
    post_dict = package_simulation_results(trajectory, gnc_results, t_eval, is_6dof)
    post_process.post_process(post_dict, output_dir="data/results")

if __name__ == "__main__":
    main()