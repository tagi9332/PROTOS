"""
Main entry point for PROTOS simulation framework.
"""
from datetime import timedelta
import numpy as np
from src.post_process import postprocess
from src.io_utils import init_PROTOS
from src import dynamics, gnc

def main():
    # Parse input
    config = init_PROTOS.parse_input("data/input_files/config_quiz_3.jsonx")
    sim_config = config["simulation"]
    dyn_config = config["dynamics"]
    gnc_config = config["gnc"]

    # Extract simulation parameters
    dt = sim_config.get("time_step", 1)
    duration = sim_config.get("duration", 3600.0)
    steps = int(duration / dt) + 1
    t_eval = np.linspace(0, duration, steps)
    initial_epoch = sim_config['epoch']

    # Initialize state
    state = {
        "sim_time": 0.0,
        "epoch": initial_epoch,
        "chief_r": np.array(dyn_config["chief_r"]),
        "chief_v": np.array(dyn_config["chief_v"]),
        "deputy_r": np.array(dyn_config["deputy_r"]),
        "deputy_v": np.array(dyn_config["deputy_v"]),
        "deputy_rho": np.array(dyn_config["deputy_rho"]),
        "deputy_rho_dot": np.array(dyn_config["deputy_rho_dot"]),
    }

    # Storage for trajectory and GNC outputs
    trajectory = [state.copy()]  # store the initial state at t=0
    gnc_results = []

    # Time-stepped propagation loop 
    for _ in enumerate(t_eval[:-1]):  # iterate over all but the last time

        # Run GNC to compute commanded control input
        gnc_out = gnc.step(state, gnc_config)

        # Store GNC output
        gnc_results.append(gnc_out)

        # Add control acceleration to dynamics configuration or state
        control_accel = gnc_out.get("accel_cmd", np.zeros(3))
        state["control_accel"] = control_accel
        gnc_out["control_accel"] = control_accel

        # Propagate dynamics using control input
        next_state = dynamics.step(state, dt, dyn_config)

        # Increment time and update state fields
        prev_sim_time = state.get("sim_time", 0.0)
        prev_epoch = state.get("epoch", initial_epoch)

        next_state["sim_time"] = np.float64(prev_sim_time + dt) # type: ignore
        next_state["epoch"] = prev_epoch + timedelta(seconds=dt) # type: ignore

        # Store propagated state and set it for next iteration
        trajectory.append(next_state)
        state = next_state

    # Execute and store final GNC step (not commanded)
    final_gnc = gnc.step(state, gnc_config)
    final_gnc["control_accel"] = final_gnc.get("accel_cmd", np.zeros(3))
    gnc_results.append(final_gnc)

    # Prepare postprocess-compatible dictionaries
    post_dict = {}

    # Store the time array
    post_dict = {"time": t_eval.tolist(), "full_state": []}

    # Build the "full_state" list
    control_accel_list = []
    for res in gnc_results:
        # Build full state vector
        state_parts = [
            res["chief_r"],
            res["chief_v"],
            res["deputy_r"],
            res["deputy_v"],
            res["deputy_rho"],
            res["deputy_rho_dot"],
        ]
        state_vector = np.hstack(state_parts)
        post_dict["full_state"].append(state_vector.tolist())

        # Collect control accelerations
        control_accel = res["control_accel"]
        control_accel_list.append(control_accel \
        if isinstance(control_accel, list) else control_accel.tolist())

    # Add control accelerations to post_dict for plotting or export
    post_dict["control_accel"] = control_accel_list

    # Postprocess results
    postprocess.postprocess(post_dict, output_dir="data/results")

if __name__ == "__main__":
    main()
