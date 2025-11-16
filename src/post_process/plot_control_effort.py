import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Ensure non-interactive backend for script saving

def plot_control_accel(results_serializable: dict, output_dir: str, filename: str = "control_accel_plot.png"):
    if "control_accel" not in results_serializable or "time" not in results_serializable:
        print("Control acceleration or time data missing. Skipping plot.")
        return

    time = np.array(results_serializable["time"], dtype=float)
    control_accel = np.array(results_serializable["control_accel"], dtype=float)

    if control_accel.ndim != 2 or control_accel.shape[1] != 3:
        print("Control acceleration data not in expected shape (N×3). Skipping plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

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

    # Use tight_layout without rect first, then save
    fig.tight_layout()
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150)  # Specify dpi to avoid blank files
    plt.close(fig)
