import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def plot_control_accel(results_serializable: dict, output_dir: str, filename: str = "control_accel_plot.png"):
    if "control_accel" not in results_serializable or "time" not in results_serializable:
        print("Control acceleration or time data missing. Skipping plot.")
        return

    time = np.array(results_serializable["time"], dtype=float)
    gnc_results = results_serializable["control_accel"]

    # ---------------------------------------------------------
    # Extract 3x1 accel vectors from dict or legacy arrays
    # ---------------------------------------------------------
    accel_list = []

    for entry in gnc_results:
        if isinstance(entry, dict):
            # Prefer accel_cmd first
            accel = entry.get("accel_cmd", entry.get("control_accel"))
            if accel is None:
                print("Skipping GNC entry with no accel_cmd/control_accel.")
                continue
            accel_list.append(np.array(accel, dtype=float))

        else:
            # Legacy Nx3 array format support
            accel_list.append(np.array(entry, dtype=float))

    if len(accel_list) == 0:
        print("No valid control acceleration entries found. Skipping plot.")
        return

    control_accel = np.vstack(accel_list)

    # ---------------------------------------------------------
    # Validate shape
    # ---------------------------------------------------------
    if control_accel.ndim != 2 or control_accel.shape[1] != 3:
        print(f"Control accel has wrong shape {control_accel.shape}, expected (N,3). Skipping plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = [r'$a_x$ (km/s$^2$)', r'$a_y$ (km/s$^2$)', r'$a_z$ (km/s$^2$)']
    colors = ['r', 'g', 'b']

    for i in range(3):
        axes[i].plot(time, control_accel[:, i], color=colors[i], linewidth=1.8)
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True)
        axes[i].set_xlim([time[0], time[-1]])

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Control Accelerations vs Time', fontsize=14)

    fig.tight_layout()
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)

    print(f"Control acceleration plot saved to: {filepath}")
