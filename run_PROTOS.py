#!/usr/bin/env python3
"""
Main entry point for PROTOS simulation framework (time-stepped execution).
"""

from src.protos import io_utils, dynamics, gnc, postprocess
import numpy as np

def main():
    # 1. Parse input
    config = io_utils.parse_input("data/input_files/test_config_orbit_frame.jsonx")
    dyn_config = config["dynamics"]
    gnc_config = config["gnc"]

    # Extract simulation parameters
    sim = dyn_config.get("simulation", {})
    dt = sim.get("time_step", 10.0)
    duration = sim.get("duration", 3600.0)
    steps = int(duration / dt) + 1
    t_eval = np.linspace(0, duration, steps)

    # 2. Initialize state (inertial)
    state = {
        "chief_r": np.array(dyn_config["chief_r"]),
        "chief_v": np.array(dyn_config["chief_v"]),
        "deputy_r": np.array(dyn_config["deputy_r"]),
        "deputy_v": np.array(dyn_config["deputy_v"]),
        # Initialize LVLH relative state
        "deputy_r_LVLH": 
        "deputy_v_LVLH": 
    }


    # Storage for trajectory and GNC outputs
    trajectory = []
    gnc_results = []

    # 3. Time-stepped propagation loop
    for t in t_eval:
        # Propagate one step
        next_state = dynamics.step(state, dt, dyn_config)
        trajectory.append(next_state)

        # Run GNC for this step
        gnc_out = gnc.step(next_state, gnc_config)
        gnc_results.append(gnc_out)

        # Update state for next step
        state = next_state

    # 4. Prepare postprocess-compatible dictionaries
    # Stack both inertial and LVLH deputy states
    post_dict = {
        "time": t_eval.tolist(),
        "state_inertial": [np.hstack((res["deputy_r"], res["deputy_v"])).tolist() for res in gnc_results],
        "state_LVLH": [np.hstack((res["deputy_r_LVLH"], res["deputy_v_LVLH"])).tolist() for res in gnc_results],
        "chief_state": [np.hstack((res["chief_r"], res["chief_v"])).tolist() for res in gnc_results]
    }

    # 5. Postprocess results (save JSON + plots)
    postprocess.postprocess(post_dict, output_dir="data/results")


if __name__ == "__main__":
    main()
