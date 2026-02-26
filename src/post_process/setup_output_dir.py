import os
import json
import numpy as np
from datetime import datetime

def setup_output_directories(results):
    """
    Creates a timestamped main output directory, vehicle-specific subdirectories,
    and generates a simulation summary text file including the run configuration.
    """
    # Create Timestamped Main Output Directory
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    main_output_dir = os.path.join("data", "results", timestamp_str)
    os.makedirs(main_output_dir, exist_ok=True)
    
    vehicle_dirs = {}

    # Set up Chief Directory
    chief_dir = os.path.join(main_output_dir, "chief")
    os.makedirs(chief_dir, exist_ok=True)
    vehicle_dirs["chief"] = chief_dir

    # Set up Deputy Directories
    deputies_dict = results.get("deputies", {})
    for dep_name in deputies_dict.keys():
        dep_dir = os.path.join(main_output_dir, dep_name)
        os.makedirs(dep_dir, exist_ok=True)
        vehicle_dirs[dep_name] = dep_dir

    # Extract basic info and write Simulation Summary Text File
    is_6dof = results.get("is_6dof", False)
    time_data = np.array(results.get("time", []), dtype=float)
    input_file = results.get("input_file", "")

    summary_path = os.path.join(main_output_dir, "sim_summary.txt")
    with open(summary_path, "w") as f:
        f.write("PROTOS Simulation Run Summary\n")
        f.write("=============================\n")
        f.write(f"Run Timestamp: {timestamp_str}\n")
        f.write(f"Simulation Mode: {'6DOF' if is_6dof else '3DOF'}\n")
        f.write(f"Number of Deputies: {len(deputies_dict)}\n")
        f.write(f"Input Configuration File: {input_file}\n")
        
        if len(time_data) > 0:
            dt = time_data[1] - time_data[0] if len(time_data) > 1 else 0
            f.write(f"Time Step (dt): {dt} seconds\n")
            f.write(f"Total Simulation Duration: {time_data[-1]} seconds\n")
            f.write(f"Total Time Steps: {len(time_data)}\n")
            
        f.write("\nNotes:\n-----------------------------\n\n")

        # Dump JSON config file
        f.write("Original Run Configuration:\n")
        f.write("=============================\n")
        if input_file:
            try:
                with open(input_file, "r") as f_in:
                    f.write(f_in.read())
            except Exception as e:
                f.write(f"Error reading input file: {e}")
        else:
            f.write("No configuration data provided to post-processor.")

    return main_output_dir, vehicle_dirs