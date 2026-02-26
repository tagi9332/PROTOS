import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

def _plot_sat_attitude(time, q, w, sat_name, specific_dir):
    """
    Helper function to plot and save quaternions and angular rates for a single satellite.
    """
    # Safety check in case of missing data
    if len(q) == 0 or len(w) == 0:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    q_labels = [r"$q_0$", r"$q_1$", r"$q_2$", r"$q_3$"]
    rate_labels = [r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]

    # -------------------------------------------------------------
    # 1) Quaternions Plot
    # -------------------------------------------------------------
    for i in range(4):
        axes[0].plot(time, q[:, i], label=q_labels[i])
    axes[0].set_ylabel("Quaternions")
    axes[0].set_title(f"{sat_name.capitalize()} Attitude Profile")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    # -------------------------------------------------------------
    # 2) Angular Rates Plot
    # -------------------------------------------------------------
    for i in range(3):
        axes[1].plot(time, w[:, i], label=rate_labels[i])
    axes[1].set_ylabel("Angular Rates (rad/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    
    # Clean up the name for file saving
    safe_name = sat_name.replace(" ", "_").lower()
    
    # Use specific directory
    filepath = os.path.join(specific_dir, f"attitude_{safe_name}.png")
    
    fig.savefig(filepath, dpi=200)
    plt.close(fig)


def plot_attitude(results_serializable: Dict[str, Any], vehicle_dirs: Dict[str, str]):
    """
    Plot quaternions and angular rates for the Chief and all Deputies.
    Routes individual attitude_[sat_name].png files to their specific vehicle folders.
    """
    if not results_serializable.get("is_6dof", False):
        return

    time = np.array(results_serializable.get("time", []), dtype=float)
    if len(time) == 0:
        return


    # Plot Chief Attitude
    chief_data = results_serializable.get("chief", {})
    qC = np.array(chief_data.get("q", []))
    wC = np.array(chief_data.get("omega", []))
    
    # Extract Chief directory
    chief_dir = vehicle_dirs.get("chief", "")
    _plot_sat_attitude(time, qC, wC, "chief", chief_dir)

    # Plot All Deputies' Attitudes
    deputies = results_serializable.get("deputies", {})
    for sat_name, sat_data in deputies.items():
        qD = np.array(sat_data.get("q", []))
        wD = np.array(sat_data.get("omega", []))
        
        # Extract Deputy directory
        dep_dir = vehicle_dirs.get(sat_name, "")
        _plot_sat_attitude(time, qD, wD, sat_name, dep_dir)