"""
Main entry point for PROTOS simulation framework.
"""
from datetime import timedelta
import copy

from src import dynamics, gnc
from src.io_utils import init_PROTOS
from src.post_process import post_process, package_simulation_results
from utils.six_dof_utils import update_state_with_gnc
from utils.print_functions.print_sim_progress import print_sim_progress

def main():
    # --- Initialization ---
    input_file_path = "data/input_files/th_nmc_j2_example_config.jsonx"

    # Load config and extract sim settings
    config = init_PROTOS.init_PROTOS(input_file_path)
    sim_config = config["simulation"]
    gnc_config = config["gnc"]

    dt = sim_config.time_step
    t_eval = sim_config.t_eval
    initial_epoch = sim_config.epoch
    is_6dof = sim_config.simulation_mode.upper() == "6DOF"
    num_steps = len(t_eval) - 1

    # Initialize states
    state = config.get("init_state", {})
    
    state["epoch"] = initial_epoch
    state["chief"]["epoch"] = initial_epoch
    for sat in state["deputies"].values():
        sat["epoch"] = initial_epoch

    trajectory = [copy.deepcopy(state)]
    gnc_results = []

    # --- Propagation Loop ---
    for i, _ in enumerate(t_eval[:-1]):
        # Progress callback
        print_sim_progress(i + 1, num_steps)

        # GNC Step
        gnc_out = gnc.gnc_step(state, gnc_config)
        gnc_results.append(copy.deepcopy(gnc_out))

        # Update State with GNC Outputs
        update_state_with_gnc(state, gnc_out, is_6dof)

        # Dynamics Step
        next_state = dynamics.dyn_step(dt, state, config)

        # Update Epoch
        next_epoch = state.get("epoch", initial_epoch) + timedelta(seconds=dt)
        next_state["epoch"] = next_epoch
        
        # Update epochs for chief and deputies
        next_state["chief"]["epoch"] = next_epoch
        for sat in next_state["deputies"].values():
            sat["epoch"] = next_epoch

        # Store & Advance
        trajectory.append(copy.deepcopy(next_state))
        state = next_state

    # --- Finalize ---
    final_gnc = gnc.gnc_step(state, gnc_config)
    gnc_results.append(copy.deepcopy(final_gnc))

    # --- Post Processing ---
    post_dict = package_simulation_results(trajectory, gnc_results, t_eval, is_6dof)
    post_dict["input_file"] = input_file_path 
    post_process.post_process(post_dict)

if __name__ == "__main__":
    main()