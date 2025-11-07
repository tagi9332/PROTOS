#!/usr/bin/env python3
"""
Main entry point for PROTOS simulation framework.
"""

from datetime import timedelta
from src import io_utils, dynamics, gnc, postprocess
import numpy as np

def main():
    # 1. Parse input
    config = io_utils.parse_input("data/input_files/test_config_gnc_rpo.jsonx")
    sim_config = config["simulation"]
    dyn_config = config["dynamics"]
    gnc_config = config["gnc"]

    # Extract simulation parameters
    dt = sim_config.get("time_step", 1) 
    duration = sim_config.get("duration", 3600.0)
    steps = int(duration / dt) + 1
    t_eval = np.linspace(0, duration, steps)
    epoch = sim_config["epoch"]


    # 2. Initialize state
    state = {
        "epoch": dyn_config.get("epoch"),
        "chief_r": np.array(dyn_config["chief_r"]),
        "chief_v": np.array(dyn_config["chief_v"]),
        "deputy_r": np.array(dyn_config["deputy_r"]),
        "deputy_v": np.array(dyn_config["deputy_v"]),
        "deputy_rho": np.array(dyn_config["deputy_rho"]),
        "deputy_rho_dot": np.array(dyn_config["deputy_rho_dot"]),
    }

    # Storage for trajectory and GNC outputs
    trajectory = [state.copy()]
    gnc_results = [gnc.step(state, gnc_config)]

    # 3. Time-stepped propagation loop
    for t in t_eval:
        # (a) Run GNC first to compute commanded control input
        gnc_out = gnc.step(state, gnc_config)

        # Store GNC output
        gnc_results.append(gnc_out)

        # Add control acceleration to dynamics configuration or state
        control_accel = gnc_out.get("accel_cmd", np.zeros(3))
        state["control_accel"] = control_accel

        # (b) Propagate dynamics using this control input
        next_state = dynamics.step(state, dt, dyn_config)

        # (c) Increment time and update state
        epoch += timedelta(seconds=dt)
        next_state["epoch"] = epoch

        # (d) Store propagated state
        trajectory.append(next_state)
        state = next_state  # update for next iteration

    # 4. Prepare postprocess-compatible dictionaries
    post_dict = {}

    # Store the time array
    post_dict["time"] = t_eval.tolist()

    # Build the "full_state" list
    full_state_list = []
    for res in gnc_results:
        # Collect all parts of the state vector into one array
        state_parts = [
            res["chief_r"],        # Chief position
            res["chief_v"],        # Chief velocity
            res["deputy_r"],       # Deputy position
            res["deputy_v"],       # Deputy velocity
            res["deputy_rho"],     # Deputy relative position (maybe in LVLH/Hill frame)
            res["deputy_rho_dot"], # Deputy relative velocity
        ]
        
        # Stack them into a single 1D vector
        state_vector = np.hstack(state_parts)

        # Convert to a regular Python list (again for JSON serialization)
        full_state_list.append(state_vector.tolist())

    # Add to dictionary
    post_dict["full_state"] = full_state_list


    # 5. Postprocess results (save JSON + plots)
    postprocess.postprocess(post_dict, output_dir="data/results")


if __name__ == "__main__":
    main()
