import os
import numpy as np
import matplotlib.pyplot as plt

def _plot_sat_delta_v(time, accel_data, sat_name, output_dir):
    """
    Helper function to calculate and plot delta-v and cumulative delta-v 
    for a single satellite.
    """
    accel = np.array(accel_data, dtype=float)
    if len(accel) == 0 or accel.ndim != 2 or accel.shape[1] != 3:
        return

    # ---------------------------------------------------------
    # Compute delta-v (Vectorized: km/s^2 -> m/s)
    # ---------------------------------------------------------
    delta_v = np.zeros_like(accel)
    
    # np.diff(time) gets dt for all steps instantly.
    # We slice [1:] to align the dimensions properly.
    dt = np.diff(time) 
    delta_v[1:] = accel[1:] * dt[:, np.newaxis] * 1e3 

    # Cumulative |Δv|
    cumulative = np.cumsum(np.abs(delta_v), axis=0)

    # Clean filename string
    safe_name = sat_name.replace(" ", "_").lower()
    
    # ---------------------------------------------------------
    # Plot 1: Δv per step
    # ---------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = [r'$\Delta v_x$', r'$\Delta v_y$', r'$\Delta v_z$']

    for i in range(3):
        axes[i].plot(time, delta_v[:, i], color='b')
        axes[i].set_ylabel(labels[i] + " (m/s)")
        axes[i].grid(True)
        axes[i].set_xlim([time[0], time[-1]])

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{sat_name} Delta-v per Time Step")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"delta_v_{safe_name}.png"), dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------
    # Plot 2: Cumulative Δv
    # ---------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = [r'Cum $\Delta v_x$', r'Cum $\Delta v_y$', r'Cum $\Delta v_z$']

    # Adding a print statement to the console so you know the total fuel cost!
    total_dv = cumulative[-1].sum()

    for i in range(3):
        axes[i].plot(time, cumulative[:, i], color='r')
        axes[i].set_ylabel(labels[i] + " (m/s)")
        axes[i].grid(True)
        axes[i].set_xlim([time[0], time[-1]])

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{sat_name} Cumulative Delta-v")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cumulative_delta_v_{safe_name}.png"), dpi=150)
    plt.close(fig)


def plot_delta_v(results_serializable, output_dir):
    """
    Plots the delta-v and cumulative delta-v for the Chief and all Deputies.
    Saves individual PNG files per spacecraft.
    """
    time = np.array(results_serializable.get("time", []), dtype=float)
    if len(time) < 2:
        print("Not enough time data for delta-v plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Plot Chief
    chief_accel = results_serializable.get("chief", {}).get("accel_cmd", [])
    _plot_sat_delta_v(time, chief_accel, "Chief", output_dir)

    # 2. Plot Deputies
    for sat_name, sat_data in results_serializable.get("deputies", {}).items():
        _plot_sat_delta_v(time, sat_data.get("accel_cmd", []), sat_name, output_dir)