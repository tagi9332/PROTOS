import os
import numpy as np
import matplotlib.pyplot as plt

def plot_delta_v(results_serializable, output_dir):
    if "control_accel" not in results_serializable or "time" not in results_serializable:
        print("No accel or time data for delta-v plot.")
        return

    time = np.array(results_serializable["time"], dtype=float)
    gnc_results = results_serializable["control_accel"]

    # ---------------------------------------------------------
    # Extract Nx3 accel array from dict or legacy form
    # ---------------------------------------------------------
    accel_list = []

    for entry in gnc_results:
        if isinstance(entry, dict):
            accel = entry.get("accel_cmd", entry.get("control_accel"))
            if accel is None:
                print("Skipping entry with no accel vector.")
                continue
            accel_list.append(np.array(accel, dtype=float))
        else:
            # Legacy flat format
            accel_list.append(np.array(entry, dtype=float))

    if len(accel_list) == 0:
        print("No valid acceleration entries found. Skipping Δv plot.")
        return

    accel = np.vstack(accel_list)

    if accel.ndim != 2 or accel.shape[1] != 3:
        print(f"Acceleration shape incorrect ({accel.shape}). Expected (N,3). Skipping Δv plot.")
        return

    # ---------------------------------------------------------
    # Compute delta-v (convert km/s² → m/s² via ×1000)
    # ---------------------------------------------------------
    delta_v = np.zeros_like(accel)

    for i in range(1, len(time)):
        dt = time[i] - time[i - 1]  # seconds
        delta_v[i] = accel[i] * dt * 1e3  # convert km/s² → m/s

    # cumulative |Δv|
    cumulative = np.cumsum(np.abs(delta_v), axis=0)

    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Plot Δv per step
    # ---------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['$Δv_x$', '$Δv_y$', '$Δv_z$']

    for i in range(3):
        axes[i].plot(time, delta_v[:, i])
        axes[i].set_ylabel(labels[i] + " (m/s)")
        axes[i].grid(True)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Delta-v per Time Step")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "delta_v_plot.png"))
    plt.close(fig)

    # ---------------------------------------------------------
    # Plot cumulative Δv
    # ---------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['Cum Δv_x', 'Cum Δv_y', 'Cum Δv_z']

    for i in range(3):
        axes[i].plot(time, cumulative[:, i])
        axes[i].set_ylabel(labels[i] + " (m/s)")
        axes[i].grid(True)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Cumulative Delta-v")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "cumulative_delta_v_plot.png"))
    plt.close(fig)
