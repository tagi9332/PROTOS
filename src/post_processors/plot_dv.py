import os
import numpy as np
import matplotlib.pyplot as plt

def plot_delta_v(results_serializable, output_dir):
    if "control_accel" not in results_serializable:
        print("No accel for delta-v plot.")
        return

    time = np.array(results_serializable["time"], dtype=float)
    accel = np.array(results_serializable["control_accel"], dtype=float)

    delta_v = np.zeros_like(accel)
    for i in range(1, len(time)):
        dt = time[i] - time[i-1]
        delta_v[i] = accel[i] * dt * 1e3

    cumulative = np.cumsum(np.abs(delta_v), axis=0)

    # Delta-v vs time (per step)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['$Δv_x$', '$Δv_y$', '$Δv_z$']
    for i in range(3):
        axes[i].plot(time, delta_v[:, i])
        axes[i].set_ylabel(labels[i] + " (m/s)")
        axes[i].grid(True)

    axes[-1].set_xlabel("Time (s)")
    plt.savefig(os.path.join(output_dir, "delta_v_plot.png"))
    plt.close()

    # Cumulative Δv
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['Cum Δv_x', 'Cum Δv_y', 'Cum Δv_z']
    for i in range(3):
        axes[i].plot(time, cumulative[:, i])
        axes[i].set_ylabel(labels[i] + " (m/s)")
        axes[i].grid(True)

    axes[-1].set_xlabel("Time (s)")
    plt.savefig(os.path.join(output_dir, "cumulative_delta_v_plot.png"))
    plt.close()