#!/usr/bin/env python3
"""
Main entry point for PROTOS simulation framework.
"""

from src.protos import io_utils, dynamics, gnc, postprocess

def main():
    # 1. Parse input
    config = io_utils.parse_input("data/sample.jsonx")

    # 2. Run dynamics propagator
    trajectory = dynamics.propagate(config)

    # 3. Run navigation, guidance, and control
    gnc_results = gnc.run(trajectory, config)

    # 4. Postprocess results
    postprocess.generate_reports(gnc_results, output_dir="data/results")

if __name__ == "__main__":
    main()
