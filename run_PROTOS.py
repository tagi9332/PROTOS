"""
Main entry point for PROTOS simulation framework.
"""
from datetime import timedelta
import numpy as np
from src.post_process import post_process
from src.io_utils import init_PROTOS
from src import dynamics, gnc

def main():
    # Parse input
    config = init_PROTOS.parse_input("data/input_files/test_config_6dof_rpo.jsonx")
    sim_config = config["simulation"]
    dyn_config = config["dynamics"]
    gnc_config = config["gnc"]

    # Extract simulation parameters
    dt = sim_config.get("time_step", 1)
    t_eval = np.array(config.get("t_eval"))
    initial_epoch = sim_config.get("epoch")

    # Initialize state
    state = config.get("init_state", {})

    # Storage for trajectory and GNC outputs
    trajectory = [state.copy()]  # store the initial state at t=0
    gnc_results = []

    # Time-stepped propagation loop 
    for _ in enumerate(t_eval[:-1]):  # iterate over all but the last time

        # Run GNC to compute commanded control input
        gnc_out = gnc.gnc_step(state, gnc_config)

        # Store GNC output
        gnc_results.append(gnc_out)

        # Add control acceleration to dynamics configuration or state
        control_accel = gnc_out.get("accel_cmd", np.zeros(3))
        state["control_accel"] = control_accel
        gnc_out["control_accel"] = control_accel

        # Propagate dynamics using control input
        next_state = dynamics.dyn_step(dt, state, dyn_config)

        # Increment time and update state fields
        prev_sim_time = state.get("sim_time", 0.0)
        prev_epoch = state.get("epoch", initial_epoch)

        next_state["sim_time"] = np.float64(prev_sim_time + dt) # type: ignore
        next_state["epoch"] = prev_epoch + timedelta(seconds=dt) # type: ignore

        # Store propagated state and set it for next iteration
        trajectory.append(next_state)
        state = next_state

    # Execute and store final GNC step (not commanded)
    final_gnc = gnc.gnc_step(state, gnc_config)
    final_gnc["control_accel"] = final_gnc.get("accel_cmd", np.zeros(3))
    gnc_results.append(final_gnc)

    # Prepare postprocess-compatible dictionaries
    post_dict = {"time": t_eval.tolist(), "full_state": []}

    # Build the "full_state" list
    for res in trajectory:
        # Build full state vector
        state_parts = [
            res["chief_r"],
            res["chief_v"],
            res["deputy_r"],
            res["deputy_v"],
            res["deputy_rho"],
            res["deputy_rho_dot"],
        ]

        # Add attitude states if 6DOF
        if sim_config.get("simulation_mode", "3DOF").upper() == "6DOF":
            state_parts.extend([
                np.array(res.get("chief_q_BN", np.zeros(4))),
                np.array(res.get("chief_omega_BN", np.zeros(3))),
                np.array(res.get("deputy_q_BN", np.zeros(4))),
                np.array(res.get("deputy_omega_BN", np.zeros(3)))
            ])

        # Flatten and append to post_dict
        state_vector = np.hstack(state_parts)
        post_dict["full_state"].append(state_vector.tolist())

    # Add control accelerations to post_dict for plotting or export
    post_dict["control_accel"] = gnc_results

    # Postprocess results
    post_process.post_process(post_dict, output_dir="data/results")

if __name__ == "__main__":
    main()
