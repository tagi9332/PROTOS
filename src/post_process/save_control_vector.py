import os
import csv
import numpy as np

def save_control_accel(results_serializable, output_dir):
    if "control_accel" not in results_serializable:
        print("No control accelerations found.")
        return

    time = results_serializable["time"]
    gnc_results = results_serializable["control_accel"]

    # ---------------------------------------------------------
    # Extract accel_cmd/control_accel as a clean Nx3 array
    # ---------------------------------------------------------
    accel_list = []

    for entry in gnc_results:
        if isinstance(entry, dict):
            # Prefer accel_cmd; fall back to control_accel
            accel = entry.get("accel_cmd", entry.get("control_accel"))
            if accel is None:
                raise ValueError("GNC result entry missing accel_cmd/control_accel")
            accel_list.append(np.array(accel, dtype=float))
        else:
            # Support legacy Nx3 list format
            accel_list.append(np.array(entry, dtype=float))

    accel_array = np.vstack(accel_list)  # (N,3)

    # ---------------------------------------------------------
    # Compute delta-v and cumulative delta-v
    # ---------------------------------------------------------
    out_csv = os.path.join(output_dir, "control_accel.csv")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time", "ax", "ay", "az",
            "dvx", "dvy", "dvz",
            "sum_dvx", "sum_dvy", "sum_dvz"
        ])

        delta_v = np.zeros_like(accel_array)

        for i in range(1, len(time)):
            dt = time[i] - time[i - 1]
            delta_v[i] = accel_array[i] * dt

        cumulative = np.cumsum(delta_v, axis=0)

        # Write rows
        for t, a, dv, s in zip(time, accel_array, delta_v, cumulative):
            writer.writerow([t] + list(a) + list(dv) + list(s))

    print(f"Control accelerations saved to: {out_csv}")
