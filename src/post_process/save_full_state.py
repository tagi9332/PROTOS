import os
import csv
import numpy as np
from typing import Dict, Any

def _save_single_state_csv(time, sat_data, sat_name, specific_dir):
    """
    Dynamically builds and saves a state CSV for a single spacecraft by checking
    which state keys (ECI, Hill, Attitude) actually exist in its dictionary.
    """
    # Define standard mappings: (dictionary_key, [CSV Headers])
    state_map = [
        ("r", ["r_x_km", "r_y_km", "r_z_km"]),
        ("v", ["v_x_kms", "v_y_kms", "v_z_kms"]),
        ("rho", ["rho_x_km", "rho_y_km", "rho_z_km"]),
        ("rho_dot", ["rho_dot_x_kms", "rho_dot_y_kms", "rho_dot_z_kms"]),
        ("q", ["q_w", "q_x", "q_y", "q_z"]),
        ("omega", ["omega_x_rads", "omega_y_rads", "omega_z_rads"])
    ]

    header = ["time_s"]
    columns = [time]

    # Dynamically build columns based on available data
    for key, cols in state_map:
        val = sat_data.get(key)
        # Check if the data exists and matches the time history length
        if val is not None and len(val) == len(time):
            header.extend(cols)
            columns.append(np.array(val, dtype=float))
    if len(columns) == 1:
        print(f"[{sat_name}] No valid state data found. Skipping CSV.")
        return

    # Stack columns horizontally 
    stacked_data = np.column_stack(columns)

    # Save to CSV in the specific vehicle's directory
    safe_name = sat_name.replace(" ", "_").lower()
    out_csv = os.path.join(specific_dir, f"state_results_{safe_name}.csv")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(stacked_data)


def save_state_csv(results_serializable: Dict[str, Any], vehicle_dirs: Dict[str, str]):
    """
    Saves individual state CSV files for the Chief and all Deputies.
    Routes each file to the specific vehicle's directory.
    """
    time = np.array(results_serializable.get("time", []), dtype=float)
    if len(time) == 0:
        print("No time data available. Skipping state CSV generation.")
        return

    # Process Chief
    chief_data = results_serializable.get("chief", {})
    if chief_data:
        # Pull the chief's specific folder path
        chief_dir = vehicle_dirs.get("chief", "")
        _save_single_state_csv(time, chief_data, "chief", chief_dir)

    # Process all Deputies
    deputies = results_serializable.get("deputies", {})
    for sat_name, sat_data in deputies.items():
        # Pull this specific deputy's folder path
        dep_dir = vehicle_dirs.get(sat_name, "")
        _save_single_state_csv(time, sat_data, sat_name, dep_dir)