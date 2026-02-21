import os
import csv
import numpy as np

def _save_sat_accel_csv(time, accel_data, sat_name, output_dir):
    """
    Helper function to calculate delta-v and save the maneuver data 
    to a CSV file for a single satellite.
    """
    accel = np.array(accel_data, dtype=float)
    if len(accel) == 0 or accel.ndim != 2 or accel.shape[1] != 3:
        return

    # ---------------------------------------------------------
    # Compute delta-v (Vectorized: km/s^2 -> m/s)
    # ---------------------------------------------------------
    delta_v = np.zeros_like(accel)
    dt = np.diff(time)
    
    # dt[:, np.newaxis] broadcasts the 1D time diff array to match the 3D accel array
    delta_v[1:] = accel[1:] * dt[:, np.newaxis] * 1e3 

    # Cumulative absolute delta-v (True fuel expenditure per axis)
    cumulative_dv = np.cumsum(np.abs(delta_v), axis=0)

    # ---------------------------------------------------------
    # Save to CSV
    # ---------------------------------------------------------
    safe_name = sat_name.replace(" ", "_").lower()
    out_csv = os.path.join(output_dir, f"control_accel_{safe_name}.csv")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Explicit headers with units!
        writer.writerow([
            "time_s", "ax_km_s2", "ay_km_s2", "az_km_s2",
            "dvx_m_s", "dvy_m_s", "dvz_m_s",
            "cum_abs_dvx_m_s", "cum_abs_dvy_m_s", "cum_abs_dvz_m_s"
        ])

        # Write rows
        for t, a, dv, s in zip(time, accel, delta_v, cumulative_dv):
            writer.writerow([t] + list(a) + list(dv) + list(s))
            

def save_control_accel(results_serializable, output_dir):
    """
    Computes delta-v and saves the control acceleration history to CSV 
    files for the Chief and all Deputies.
    """
    time = np.array(results_serializable.get("time", []), dtype=float)
    if len(time) < 2:
        print("Not enough time data to save control accelerations.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Process Chief
    chief_accel = results_serializable.get("chief", {}).get("accel_cmd", [])
    _save_sat_accel_csv(time, chief_accel, "Chief", output_dir)

    # 2. Process Deputies
    for sat_name, sat_data in results_serializable.get("deputies", {}).items():
        _save_sat_accel_csv(time, sat_data.get("accel_cmd", []), sat_name, output_dir)