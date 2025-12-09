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


def plot_attitude_control(results_serializable: dict, output_dir: str, filename: str = "attitude_control_summary.png"):
    
    # Define keys
    torque_key = "torque_cmd_deputy"
    att_err_key = "att_error_deputy" 
    rate_err_key = "rate_error_deputy"

    # --- 1. Data Validation and Extraction ---
    if "time" not in results_serializable:
        print("Time data missing. Skipping plots.")
        return
    
    time = np.array(results_serializable["time"], dtype=float)
    
    # Extract and validate torque data
    if torque_key in results_serializable:
        torque_data = np.array(results_serializable[torque_key], dtype=float)
        if torque_data.ndim != 2 or torque_data.shape[1] != 3:
            torque_data = None
            print(f"Torque data has wrong shape {torque_data.shape}. Skipping torque plot.")
    else:
        torque_data = None
    
    # Extract and validate attitude error data (and filter to 3 components)
    if att_err_key in results_serializable:
        att_err_data = np.array(results_serializable[att_err_key], dtype=float)
        if att_err_data.ndim == 2 and att_err_data.shape[1] == 4:
            # If 4 components (Quaternion), take vector part (q_x, q_y, q_z)
            att_error_vector = att_err_data[:, 1:4]
        elif att_err_data.ndim == 2 and att_err_data.shape[1] == 3:
            # Already a 3-component vector (MRP or vector part)
            att_error_vector = att_err_data
        else:
            att_error_vector = None
            print(f"Attitude error data has wrong shape {att_err_data.shape}. Skipping attitude error plot.")
    else:
        att_error_vector = None

    # Extract and validate rate error data
    if rate_err_key in results_serializable:
        rate_err_data = np.array(results_serializable[rate_err_key], dtype=float)
        if rate_err_data.ndim != 2 or rate_err_data.shape[1] != 3:
            rate_err_data = None
            print(f"Rate error data has wrong shape {rate_err_data.shape}. Skipping rate error plot.")
    else:
        rate_err_data = None
    
    # Check if any data is available to plot
    if torque_data is None and att_error_vector is None and rate_err_data is None:
        print("No valid attitude control or error data found for plotting.")
        return

    # --- 2. Single Figure Setup ---
    # Determine which plots to show (max 3)
    data_to_plot = []
    titles = []
    y_labels = []
    
    if torque_data is not None:
        data_to_plot.append(torque_data)
        titles.append('Deputy Control Torques')
        y_labels.append([r'$\tau_x$ (Nm)', r'$\tau_y$ (Nm)', r'$\tau_z$ (Nm)'])
    
    if att_error_vector is not None:
        data_to_plot.append(att_error_vector)
        titles.append('Deputy Attitude Error Vector')
        y_labels.append([r'$e_1$', r'$e_2$', r'$e_3$'])
        
    if rate_err_data is not None:
        data_to_plot.append(rate_err_data)
        titles.append('Deputy Angular Rate Error')
        y_labels.append([r'$\delta \omega_x$ (rad/s)', r'$\delta \omega_y$ (rad/s)', r'$\delta \omega_z$ (rad/s)'])

    num_plots = len(data_to_plot)
    if num_plots == 0:
        return
        
    # Create the single figure with stacked subplots
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)
    if num_plots == 1:
        # If only one plot, axes is not an array, so make it iterable
        axes = [axes]

    colors = ['r', 'g', 'b'] # x, y, z components
    component_labels = [r'Body frame x', 
                        r'Body frame y', 
                        r'Body frame z']

    # --- 3. Plotting Loop ---
    for k in range(num_plots):
        current_data = data_to_plot[k]
        current_axes = axes[k]
        
        # Plot all three components on the single subplot axis
        for i in range(3):
            # Plot the i-th component (x, y, or z)
            current_axes.plot(time, current_data[:, i], color=colors[i], 
                              label=component_labels[i], linewidth=1.5)
            
        current_axes.set_title(titles[k])
        current_axes.set_ylabel('Magnitude') # Generic Y-label for the combined plot
        current_axes.grid(True)
        current_axes.set_xlim([time[0], time[-1]])
        current_axes.legend(loc='best', fontsize='small')

    # Set the final x-label on the bottom plot
    axes[-1].set_xlabel('Time (s)')
    
    # Add an overall title
    fig.suptitle('Deputy Attitude Control Performance', fontsize=16)

    # --- 4. Save and Close ---
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout for suptitle
    fig.savefig(filepath, dpi=150)
    plt.close(fig)

    print(f"Control summary plot saved to: {filepath}")