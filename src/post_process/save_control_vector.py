import os
import csv
import numpy as np

def save_control_accel(results_serializable, output_dir):
    if "control_accel" not in results_serializable:
        print("No control accelerations found.")
        return

    time = results_serializable["time"]
    accel_array = np.array(results_serializable["control_accel"], dtype=float)

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
            dt = time[i] - time[i-1]
            delta_v[i] = accel_array[i] * dt

        cumulative = np.cumsum(delta_v, axis=0)

        for t, a, dv, s in zip(time, accel_array, delta_v, cumulative):
            writer.writerow([t] + list(a) + list(dv) + list(s))

    print(f"Control accelerations saved to: {out_csv}")