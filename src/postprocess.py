import os
import numpy as np
from src.post_processors.save_full_state import save_state_csv
from src.post_processors.save_element_vectors import save_orbital_elements
from src.post_processors.plot_ric_traj import plot_RIC_trajectory
from src.post_processors.plot_eci_traj import plot_ECI_trajectories
from src.post_processors.plot_3d_ric_traj import plot_3d_RIC_trajectory
from src.post_processors.save_control_vector import save_control_accel
from src.post_processors.plot_dv import plot_delta_v
from src.post_processors.plot_oes import plot_orbital_elements
from src.post_processors.plot_control_effort import plot_control_accel




def _convert_ndarray(obj):
    if isinstance(obj, dict):
        return {k: _convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_ndarray(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def postprocess(results, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    results_serializable = _convert_ndarray(results)

    # Save raw state vectors
    save_state_csv(results_serializable, output_dir)

    # Save classical & differential COEs
    coes = save_orbital_elements(results_serializable, output_dir)

    # Trajectory plots
    plot_ECI_trajectories(results_serializable, output_dir)

    # Control profiles
    save_control_accel(results_serializable, output_dir)
    plot_delta_v(results_serializable, output_dir)
    plot_control_accel(results_serializable, output_dir) # type: ignore



    # COE plots
    if coes is not None:
        chief, deputy, delta = coes
        time = np.array(results_serializable["time"], dtype=float) # type: ignore
        plot_orbital_elements(time, chief, deputy, delta, output_dir)

    print(f"Postprocess completed. Output saved in {output_dir}")

    # Open RIC-frame plot in interactive window
    plot_3d_RIC_trajectory(results_serializable, output_dir, show_plot=True)




















#---------------------------------------------------------------------------------------------------------




# import os
# import csv
# import numpy as np
# import matplotlib.pyplot as plt
# from data.resources.constants import MU_EARTH
# from utils.frame_convertions.rel_to_inertial_functions import LVLH_DCM
# from utils.orbital_element_conversions.oe_conversions import inertial_to_orbital_elements


# def _convert_ndarray(obj):
#     """
#     Recursively convert any np.ndarray in obj to a list.
#     """
#     if isinstance(obj, dict):
#         return {k: _convert_ndarray(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [_convert_ndarray(v) for v in obj]
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     else:
#         return obj

# def postprocess(results: dict, output_dir: str):
#     """
#     Save simulation results and generate trajectory plots in the RIC frame.
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     # Recursively convert any ndarray to list (if needed)
#     results_serializable = _convert_ndarray(results)

#     # ---------------------------------
#     # Save main simulation state results
#     # ---------------------------------
#     output_csv = os.path.join(output_dir, "state_results.csv")
#     with open(output_csv, "w", newline="") as f:
#         writer = csv.writer(f)

#         # Write header row
#         writer.writerow([
#             "time",
#             "chief_r_x", "chief_r_y", "chief_r_z",
#             "chief_v_x", "chief_v_y", "chief_v_z",
#             "deputy_r_x", "deputy_r_y", "deputy_r_z",
#             "deputy_v_x", "deputy_v_y", "deputy_v_z",
#             "deputy_rho_x", "deputy_rho_y", "deputy_rho_z",
#             "deputy_rho_dot_x", "deputy_rho_dot_y", "deputy_rho_dot_z"
#         ])

#         # Write data rows
#         for t, state_vector in zip(results["time"], results["full_state"]):
#             writer.writerow([t] + state_vector)

#     print(f"Results saved to {output_csv}")

#     # ----------------------------------
#     # Save classical orbital elements and differential elements
#     # ---------------------------------
#     coe_output_csv = os.path.join(output_dir, "orbital_elements.csv")

#     states = np.array(results_serializable.get("full_state", []), dtype=float) # type: ignore
#     if len(states) == 0:
#         print("No trajectory data found for COE export. Skipping orbital_elements.csv.")
#     else:
#         mu = MU_EARTH  # km^3/s^2, Earth gravitational parameter

#         chief_r = states[:, 0:3]
#         chief_v = states[:, 3:6]
#         deputy_r = states[:, 6:9]
#         deputy_v = states[:, 9:12]

#         time = np.array(results_serializable["time"], dtype=float) # type: ignore

#         # Allocate arrays
#         chief_coes = np.zeros((len(time), 6))
#         deputy_coes = np.zeros((len(time), 6))
#         delta_coes = np.zeros((len(time), 6))

#         for k in range(len(time)):
#             # Chief COEs: [a, e, i, RAAN, ω, ν]
#             a_c, e_c, i_c, RAAN_c, ARGP_c, TA_c = inertial_to_orbital_elements(chief_r[k], chief_v[k], MU_EARTH,'deg')
#             a_d, e_d, i_d, RAAN_d, ARGP_d, TA_d = inertial_to_orbital_elements(deputy_r[k], deputy_v[k], MU_EARTH, 'deg')

#             chief_coes[k] = [a_c, e_c, i_c, RAAN_c, ARGP_c, TA_c]
#             deputy_coes[k] = [a_d, e_d, i_d, RAAN_d, ARGP_d, TA_d]
#             # Differential orbital elements δOEs
#             delta_coes[k] = (chief_coes[k] - deputy_coes[k])

#         # Stack everything for export
#         coe_data = np.column_stack([
#             time,
#             chief_coes[:, :],  # a, e, i, RAAN, ARGP, TA
#             deputy_coes[:, :],  # a, e, i, RAAN, ARGP, TA
#             delta_coes[:, :]   # δa, δe, δi, δRAAN, δARGP, δTA
#         ])

#         header = [
#             "time [s]",
#             "a_c [km]", "e_c [-]", "i_c [deg]", "RAAN_c [deg]", "ARGP_c [deg]", "TA_c [deg]",
#             "a_d [km]", "e_d [-]", "i_d [deg]", "RAAN_d [deg]", "ARGP_d [deg]", "TA_d [deg]",
#             "delta_a [km]", "delta_e [-]", "delta_i [deg]", "delta_RAAN [deg]", "delta_ARGP [deg]", "delta_TA [deg]"
#         ]

#         with open(coe_output_csv, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(header)
#             writer.writerows(coe_data)

#         print(f"Classical orbital elements and differential elements saved to {coe_output_csv}")

#     # ---------------------------------
#     # Save control accelerations (if available)
#     # ---------------------------------
#     if "control_accel" in results_serializable:
#         output_ctrl_csv = os.path.join(output_dir, "control_accel.csv")
#         time_array = results_serializable["time"] # type: ignore
#         accel_array = np.array(results_serializable["control_accel"])  # type: ignore # shape (N,3)

#         # Compute delta-v using simple forward difference
#         delta_v_array = np.zeros_like(accel_array)
#         for i in range(1, len(time_array)):
#             dt = time_array[i] - time_array[i-1]  # type: ignore # seconds
#             delta_v_array[i] = accel_array[i] * dt

#         # Optionally, compute cumulative delta-v
#         cumulative_delta_v = np.cumsum(delta_v_array, axis=0)

#         with open(output_ctrl_csv, "w", newline="") as f:
#             writer = csv.writer(f)
#             # header: acceleration + delta-v columns
#             writer.writerow([
#                 "time [s]", 
#                 "ax [km/s^2]", "ay [km/s^2]", "az [km/s^2]",
#                 "dvx [km/s]", "dvy [km/s]", "dvz [km/s]",
#                 "sum_dvx [km/s]", "sum_dvy [km/s]", "sum_dvz [km/s]"
#             ])

#             for t, accel, dv, cum_dv in zip(time_array, accel_array, delta_v_array, cumulative_delta_v):
#                 writer.writerow([t] + list(accel) + list(dv) + list(cum_dv))

#         print(f"Control accelerations and delta-v saved to {output_ctrl_csv}")
#     else:
#         print("No control acceleration data found in results. Skipping control_accel.csv export.")

#     # ---------------------------------
#     # Plot Trajectories
#     # ---------------------------------
#     def plot_trajectories(results_serializable, output_dir):
#         # Extract time and states
#         time = results_serializable.get("time", [])
#         states = results_serializable.get("full_state", [])

#         if not time or not states:
#             print("No trajectory data found in results. Skipping plots.")
#             return

#         states = np.array(states, dtype=float)  # shape (N, 18)

#         # -------------------------
#         # Relative RIC frame
#         # -------------------------
#         rho = states[:, 12:15]       # deputy_rho
#         rho_dot = states[:, 15:18]   # deputy_rho_dot

#         x_r, y_r, z_r = rho.T

#         x0_r, y0_r, z0_r = x_r[0], y_r[0], z_r[0]
#         xf_r, yf_r, zf_r = x_r[-1], y_r[-1], z_r[-1]

#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection='3d')
#         ax.plot(x_r, y_r, z_r, label='Deputy Trajectory', color='blue')
#         ax.scatter([0], [0], [0], color='red', s=60, label='Chief (origin)') # type: ignore
#         ax.scatter([x0_r], [y0_r], [z0_r], color='green', s=60, label='Deputy Start') # type: ignore
#         ax.scatter([xf_r], [yf_r], [zf_r], color='black', s=60, label='Deputy End') # type: ignore
#         ax.set_xlabel('Radial (km)')
#         ax.set_ylabel('In-track (km)')
#         ax.set_zlabel('Cross-track (km)') # type: ignore
#         ax.set_title('Deputy Relative Trajectory in RIC Frame')
#         ax.legend()
#         ax.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, "trajectory_RIC.png"))
#         plt.close()

#         # 2D projections
#         projections = [
#             ('Radial vs In-track', x_r, y_r, 'Radial (km)', 'In-track (km)', 'RIC_xy.png'),
#             ('Radial vs Cross-track', x_r, z_r, 'Radial (km)', 'Cross-track (km)', 'RIC_xz.png'),
#             ('In-track vs Cross-track', y_r, z_r, 'In-track (km)', 'Cross-track (km)', 'RIC_yz.png')
#         ]

#         for title, X, Y, xlabel, ylabel, filename in projections:
#             plt.figure(figsize=(7, 6))
#             plt.plot(X, Y, color='blue')
#             plt.scatter([0], [0], color='red', s=60, label='Chief')
#             plt.scatter([X[0]], [Y[0]], color='green', s=60, label='Start')
#             plt.scatter([X[-1]], [Y[-1]], color='black', s=60, label='End')
#             plt.xlabel(xlabel)
#             plt.ylabel(ylabel)
#             plt.title(title)
#             plt.legend()
#             plt.grid(True)
#             plt.axis('equal')
#             plt.tight_layout()
#             plt.savefig(os.path.join(output_dir, filename))
#             plt.close()

#         # -------------------------
#         # Inertial ECI frame
#         # -------------------------
#         chief_r = states[:, 0:3]
#         deputy_r = states[:, 6:9]
#         x_c, y_c, z_c = chief_r.T
#         x_d, y_d, z_d = deputy_r.T

#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection='3d')
#         ax.plot(x_c, y_c, z_c, label='Chief Trajectory', color='red')
#         ax.plot(x_d, y_d, z_d, label='Deputy Trajectory', color='blue')
#         ax.scatter([x_c[0]], [y_c[0]], [z_c[0]], color='red', s=60, label='Chief Start') # type: ignore
#         ax.scatter([x_d[0]], [y_d[0]], [z_d[0]], color='green', s=60, label='Deputy Start') # type: ignore
#         ax.set_xlabel('ECI X (km)')
#         ax.set_ylabel('ECI Y (km)')
#         ax.set_zlabel('ECI Z (km)') # type: ignore
#         ax.set_title('Inertial Trajectories (ECI Frame)')
#         ax.legend()
#         ax.grid(True)

#         max_range = np.array([x_c.max()-x_c.min(), y_c.max()-y_c.min(), z_c.max()-z_c.min()]).max() / 2.0
#         mid_x = (x_c.max()+x_c.min()) * 0.5
#         ax.set_xlim(mid_x - max_range, mid_x + max_range)
#         ax.set_ylim(mid_x - max_range, mid_x + max_range)
#         ax.set_zlim(mid_x - max_range, mid_x + max_range) # type: ignore

#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, "trajectory_ECI.png"))
#         plt.close()

#         print(f"RIC and ECI trajectory plots saved in {output_dir}")

#     plot_trajectories(results_serializable, output_dir)

#     # ---------------------------------
#     # Plot Control Accelerations
#     # ---------------------------------
#     if "control_accel" in results_serializable:
#         control_accel = np.array(results_serializable["control_accel"], dtype=float) # type: ignore
#         time = np.array(results_serializable["time"], dtype=float) # type: ignore

#         if control_accel.ndim == 2 and control_accel.shape[1] == 3:
#             fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

#             labels = [r'$a_x$ (km/s$^2$)', r'$a_y$ (km/s$^2$)', r'$a_z$ (km/s$^2$)']
#             colors = ['r', 'g', 'b']  # RGB for each axis

#             for i in range(3):
#                 axes[i].plot(time, control_accel[:, i], color=colors[i], linewidth=1.8)
#                 axes[i].set_ylabel(labels[i])
#                 axes[i].grid(True)
#                 axes[i].set_xlim([time[0], time[-1]])

#             axes[-1].set_xlabel('Time (s)')
#             fig.suptitle('Control Accelerations vs Time', fontsize=14)
#             fig.tight_layout(rect=[0, 0, 1, 0.96]) # type: ignore
#             plt.savefig(os.path.join(output_dir, "control_accel_plot.png"))
#             plt.close()
#             print(f"Control acceleration plots saved to {output_dir}")
#         else:
#             print("Control acceleration data not in expected shape (N×3). Skipping plot.")

#     # ---------------------------------
#     # Plot Commanded Delta-Vs
#     # ---------------------------------
#     if "control_accel" in results_serializable:
#         control_accel = np.array(results_serializable["control_accel"], dtype=float)  # type: ignore # shape (N,3)
#         time = np.array(results_serializable["time"], dtype=float) # type: ignore

#         if control_accel.ndim == 2 and control_accel.shape[1] == 3:
#             # Compute delta-v in m/s per timestep (forward difference)
#             delta_v = np.zeros_like(control_accel) 
#             for i in range(1, len(time)):
#                 dt = time[i] - time[i-1]
#                 delta_v[i] = control_accel[i] * dt * 1e3  # km/s^2 * s = m/s

#             # Compute cumulative delta-v
#             cumulative_delta_v = np.cumsum(np.abs(delta_v), axis=0)

#             # Plot delta-v per timestep
#             fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
#             labels = [r'$\Delta v_x$ (m/s)', r'$\Delta v_y$ (m/s)', r'$\Delta v_z$ (m/s)']
#             colors = ['r', 'g', 'b']

#             for i in range(3):
#                 axes[i].plot(time, delta_v[:, i], color=colors[i], linewidth=1.8)
#                 axes[i].set_ylabel(labels[i])
#                 axes[i].grid(True)
#                 axes[i].set_xlim([time[0], time[-1]])

#             axes[-1].set_xlabel('Time (s)')
#             fig.suptitle('Commanded Delta-V vs Time', fontsize=14)
#             fig.tight_layout(rect=[0, 0, 1, 0.96]) # type: ignore
#             plt.savefig(os.path.join(output_dir, "delta_v_plot.png"))
#             plt.close()
#             print(f"Commanded delta-v plots saved to {output_dir}")

#             # Plot cumulative delta-v
#             fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
#             labels_cum = [r'Cumulative $\Delta v_x$ (m/s)',
#                         r'Cumulative $\Delta v_y$ (m/s)',
#                         r'Cumulative $\Delta v_z$ (m/s)']

#             for i in range(3):
#                 axes[i].plot(time, cumulative_delta_v[:, i], color=colors[i], linewidth=1.8)
#                 axes[i].set_ylabel(labels_cum[i])
#                 axes[i].grid(True)
#                 axes[i].set_xlim([time[0], time[-1]])

#             axes[-1].set_xlabel('Time (s)')
#             fig.suptitle('Cumulative Commanded Delta-V vs Time', fontsize=14)
#             fig.tight_layout(rect=[0, 0, 1, 0.96]) # type: ignore
#             plt.savefig(os.path.join(output_dir, "cumulative_delta_v_plot.png"))
#             plt.close()
#             print(f"Cumulative delta-v plots saved to {output_dir}")

#         else:
#             print("Control acceleration data not in expected shape (Nx3). Skipping delta-v plots.")


#     # ---------------------------------
#     # Plot Classical Orbital Elements
#     # ---------------------------------
#     coe_file = os.path.join(output_dir, "orbital_elements.csv")
#     if os.path.exists(coe_file):
#         # Load the saved COE data
#         coe_data = np.loadtxt(coe_file, delimiter=",", skiprows=1)
#         time = coe_data[:, 0]
#         coe_labels = [r"$a$ (km)", r"$e$", r"$i$ (deg)", r"$\Omega$ (deg)", r"$\omega$ (deg)", r"$\nu$ (deg)"]

#         def plot_coe_set(time, coe_data, labels, title, filename):
#             fig, axes = plt.subplots(len(labels), 1, figsize=(10, 2 * len(labels)), sharex=True)
#             for i in range(len(labels)):
#                 axes[i].plot(time, coe_data[:, i], linewidth=1.5)
#                 axes[i].set_ylabel(labels[i])
#                 axes[i].grid(True)
#             axes[-1].set_xlabel("Time (s)")
#             fig.suptitle(title, fontsize=14)
#             fig.tight_layout(rect=[0, 0, 1, 0.96]) # type: ignore
#             plt.savefig(os.path.join(output_dir, filename))
#             plt.close()

#         # Plot Chief COEs
#         plot_coe_set(time, chief_coes, coe_labels, "Chief Orbital Elements vs Time", "chief_orbital_elements.png") # type: ignore

#         # Plot Deputy COEs
#         plot_coe_set(time, deputy_coes, coe_labels, "Deputy Orbital Elements vs Time", "deputy_orbital_elements.png") # type: ignore

#         # Plot Differential COEs
#         plot_coe_set(time, delta_coes, coe_labels, "Differential Orbital Elements vs Time", "diff_orbital_elements.png") # type: ignore

#         print(f"Orbital element plots saved to {output_dir}")

#     else:
#         print("No orbital_elements.csv found. Skipping orbital element plots.")


#     def plot_hill_frame_trajectory(results_serializable, output_dir, show_plot=True):
#         """
#         Generate a 3D plot of the deputy's relative position in the chief's Hill frame over time.
#         Saves a PNG to output_dir.
#         """

#         os.makedirs(output_dir, exist_ok=True)

#         states = np.array(results_serializable.get("full_state", []), dtype=float)
#         if len(states) == 0:
#             print("No trajectory data available for Hill-frame plot. Skipping.")
#             return

#         relative_positions = states[:, 12:15]  # deputy_rho
#         chief_positions = states[:, 0:3]       # chief inertial

#         # Initialize array for Hill-frame relative positions
#         relative_positions_H = np.zeros_like(relative_positions)

#         for k in range(len(relative_positions)):
#             r_c = chief_positions[k]
#             v_c = states[k, 3:6]
#             R_hill = LVLH_DCM(r_c, v_c)  # inertial -> Hill
#             relative_positions_H[k] = R_hill @ relative_positions[k]

#         # 3D plot
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection='3d')

#         ax.plot(
#             relative_positions_H[:, 0],  # radial
#             relative_positions_H[:, 1],  # along-track
#             relative_positions_H[:, 2],  # cross-track
#             label='Deputy Trajectory', color='blue', linewidth=1.5
#         )
#         # Start and end markers
#         ax.scatter([relative_positions_H[0, 0]], [relative_positions_H[0, 1]], [relative_positions_H[0, 2]], # type: ignore
#                 color='green', s=60, label='Start') # type: ignore
#         ax.scatter([relative_positions_H[-1, 0]], [relative_positions_H[-1, 1]], [relative_positions_H[-1, 2]], # type: ignore
#                 color='black', s=60, label='End') # type: ignore
#         # Chief origin
#         ax.scatter([0], [0], [0], color='red', s=60, label='Chief (origin)') # type: ignore

#         ax.set_xlabel("Radial (x) [km]")
#         ax.set_ylabel("Along-track (y) [km]")
#         ax.set_zlabel("Cross-track (z) [km]") # type: ignore
#         ax.set_title("Deputy Relative Motion in Chief's Hill Frame")
#         ax.legend()
#         ax.grid(True)
#         ax.view_init(elev=20, azim=-60) # type: ignore
#         ax.set_box_aspect([1, 1, 1])  # type: ignore # equal aspect

#         # Show interactive window if requested
#         if show_plot:
#             plt.show()

#         plt.close()

#     # Call the function inside postprocess
#     plot_hill_frame_trajectory(results_serializable, output_dir)


