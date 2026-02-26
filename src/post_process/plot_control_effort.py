import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, Any

matplotlib.use('Agg')  # Non-interactive backend

# ==============================================================================
# Translational Control Plotter
# ==============================================================================
def plot_control_accel(results_serializable: Dict[str, Any], vehicle_dirs: Dict[str, str]):
    """
    Plots the commanded control accelerations for the Chief and all Deputies.
    Routes individual control_accel_[sat_name].png files to specific folders.
    """
    time = np.array(results_serializable.get("time", []), dtype=float)
    if len(time) == 0:
        print("Time data missing. Skipping control accel plots.")
        return

    # Helper function to generate the plot
    def _plot_single_accel(accel_data, sat_name, specific_dir):
        accel = np.array(accel_data, dtype=float)
        if len(accel) == 0 or accel.ndim != 2 or accel.shape[1] != 3:
            return

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        labels = [r'$a_x$ (km/s$^2$)', r'$a_y$ (km/s$^2$)', r'$a_z$ (km/s$^2$)']
        colors = ['r', 'g', 'b']

        for i in range(3):
            axes[i].plot(time, accel[:, i], color=colors[i], linewidth=1.8)
            axes[i].set_ylabel(labels[i])
            axes[i].grid(True)
            axes[i].set_xlim([time[0], time[-1]])

        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(f'{sat_name.capitalize()} Control Accelerations vs Time', fontsize=14)
        fig.tight_layout()
        
        safe_name = sat_name.replace(" ", "_").lower()
        fig.savefig(os.path.join(specific_dir, f"control_accel_{safe_name}.png"), dpi=150)
        plt.close(fig)

    # 1. Plot Chief
    chief_accel = results_serializable.get("chief", {}).get("accel_cmd", [])
    chief_dir = vehicle_dirs.get("chief", "")
    _plot_single_accel(chief_accel, "chief", chief_dir)

    # 2. Plot Deputies
    for sat_name, sat_data in results_serializable.get("deputies", {}).items():
        dep_dir = vehicle_dirs.get(sat_name, "")
        _plot_single_accel(sat_data.get("accel_cmd", []), sat_name, dep_dir)


# ==============================================================================
# Attitude Control Plotter
# ==============================================================================
def plot_attitude_control(results_serializable: Dict[str, Any], vehicle_dirs: Dict[str, str]):
    """
    Plots the commanded torques and tracking errors for all satellites.
    Routes individual attitude_control_[sat_name].png files to specific folders.
    """
    if not results_serializable.get("is_6dof", False):
        return

    time = np.array(results_serializable.get("time", []), dtype=float)
    if len(time) == 0:
        return

    # Helper function to generate stacked dynamic subplots
    def _plot_single_attitude_control(sat_data, sat_name, specific_dir):
        torque = np.array(sat_data.get("torque_cmd", []), dtype=float)
        att_err = np.array(sat_data.get("att_error", []), dtype=float)
        rate_err = np.array(sat_data.get("rate_error", []), dtype=float)

        data_to_plot = []
        titles = []

        if len(torque) > 0 and torque.shape[1] == 3:
            data_to_plot.append((torque, f'{sat_name.capitalize()} Control Torques'))
            
        if len(att_err) > 0:
            # If quaternion (4 elements), slice to just the vector part (q1, q2, q3)
            if att_err.shape[1] == 4:
                att_err = att_err[:, 1:4]
            if att_err.shape[1] == 3:
                data_to_plot.append((att_err, f'{sat_name.capitalize()} Attitude Error Vector'))
                
        if len(rate_err) > 0 and rate_err.shape[1] == 3:
            data_to_plot.append((rate_err, f'{sat_name.capitalize()} Angular Rate Error'))

        num_plots = len(data_to_plot)
        if num_plots == 0:
            return

        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)
        if num_plots == 1:
            axes = [axes]

        colors = ['r', 'g', 'b']
        component_labels = [r'Body frame x', r'Body frame y', r'Body frame z']

        for k, (data, title) in enumerate(data_to_plot):
            ax = axes[k]
            for i in range(3):
                ax.plot(time, data[:, i], color=colors[i], label=component_labels[i], linewidth=1.5)
            ax.set_title(title)
            ax.set_ylabel('Magnitude')
            ax.grid(True)
            ax.set_xlim([time[0], time[-1]])
            ax.legend(loc='best', fontsize='small')

        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(f'{sat_name.capitalize()} Attitude Control Performance', fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        safe_name = sat_name.replace(" ", "_").lower()
        fig.savefig(os.path.join(specific_dir, f"attitude_control_{safe_name}.png"), dpi=150)
        plt.close(fig)

    # Plot Chief (Often only has torques, no errors)
    chief_dir = vehicle_dirs.get("chief", "")
    _plot_single_attitude_control(results_serializable.get("chief", {}), "chief", chief_dir)

    # Plot Deputies
    for sat_name, sat_data in results_serializable.get("deputies", {}).items():
        dep_dir = vehicle_dirs.get(sat_name, "")
        _plot_single_attitude_control(sat_data, sat_name, dep_dir)