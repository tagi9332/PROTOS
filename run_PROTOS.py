#!/usr/bin/env python3
"""
Main entry point for PROTOS simulation framework.
"""

from src.protos import io_utils, dynamics, gnc, postprocess

def main():
    # 1. Parse input (updated to current test config file)
    config = io_utils.parse_input("data/input_files/test_config_orbit_frame.jsonx")

    # 2. Run dynamics propagator
    trajectory = dynamics.propagate(config["dynamics"])
    
    # 3. Run navigation, guidance, and control
    # Assume gnc.run takes trajectory and gnc config dictionary
    gnc_results = gnc.run(trajectory, config["gnc"])
    
    # 4. Postprocess results (save JSON + generate plots)
    postprocess.postprocess(gnc_results, output_dir="data/results")

if __name__ == "__main__":
    main()
