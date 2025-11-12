import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from utils.frame_convertions.rel_to_inertial_functions import LVLH_DCM


def _convert_ndarray(obj):
    """
    Recursively convert any np.ndarray in obj to a list.
    """
    if isinstance(obj, dict):
        return {k: _convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_ndarray(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def postprocess(results: dict, output_dir: str):
    """
    Save simulation results and generate trajectory plots in the RIC frame.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Recursively convert any ndarray to list (if needed)
    results_serializable = _convert_ndarray(results)

    # ---------------------------------
    # Save main simulation state results
    # ---------------------------------
    output_csv = os.path.join(output_dir, "results.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header row
        writer.writerow([
            "time",
            "chief_r_x", "chief_r_y", "chief_r_z",
            "chief_v_x", "chief_v_y", "chief_v_z",
            "deputy_r_x", "deputy_r_y", "deputy_r_z",
            "deputy_v_x", "deputy_v_y", "deputy_v_z",
            "deputy_rho_x", "deputy_rho_y", "deputy_rho_z",
            "deputy_rho_dot_x", "deputy_rho_dot_y", "deputy_rho_dot_z"
        ])

        # Write data rows
        for t, state_vector in zip(results["time"], results["full_state"]):
            writer.writerow([t] + state_vector)

    print(f"Results saved to {output_csv}")

    # ---------------------------------
    # Save control accelerations (if available)
    # ---------------------------------
    if "control_accel" in results_serializable:
        output_ctrl_csv = os.path.join(output_dir, "control_accel.csv")
        with open(output_ctrl_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "ax", "ay", "az"])  # header

            for t, accel in zip(results_serializable["time"], results_serializable["control_accel"]):
                writer.writerow([t] + list(accel))

        print(f"Control accelerations saved to {output_ctrl_csv}")
    else:
        print("No control acceleration data found in results. Skipping control_accel.csv export.")

    # ---------------------------------
    # Plot Trajectories (existing section)
    # ---------------------------------
    def plot_trajectories(results_serializable, output_dir):
        # Extract time and states
        time = results_serializable.get("time", [])
        states = results_serializable.get("full_state", [])

        if not time or not states:
            print("No trajectory data found in results. Skipping plots.")
            return

        states = np.array(states, dtype=float)  # shape (N, 18)

        # -------------------------
        # Relative RIC frame
        # -------------------------
        rho = states[:, 12:15]       # deputy_rho
        rho_dot = states[:, 15:18]   # deputy_rho_dot

        x_r, y_r, z_r = rho.T
        vx_r, vy_r, vz_r = rho_dot.T

        x0_r, y0_r, z0_r = x_r[0], y_r[0], z_r[0]
        xf_r, yf_r, zf_r = x_r[-1], y_r[-1], z_r[-1]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_r, y_r, z_r, label='Deputy Trajectory', color='blue')
        ax.scatter([0], [0], [0], color='red', s=60, label='Chief (origin)')
        ax.scatter([x0_r], [y0_r], [z0_r], color='green', s=60, label='Deputy Start')
        ax.scatter([xf_r], [yf_r], [zf_r], color='black', s=60, label='Deputy End')
        ax.set_xlabel('Radial (km)')
        ax.set_ylabel('In-track (km)')
        ax.set_zlabel('Cross-track (km)')
        ax.set_title('Deputy Relative Trajectory in RIC Frame')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "trajectory_RIC.png"))
        plt.close()

        # 2D projections
        projections = [
            ('Radial vs In-track', x_r, y_r, 'Radial (km)', 'In-track (km)', 'RIC_xy.png'),
            ('Radial vs Cross-track', x_r, z_r, 'Radial (km)', 'Cross-track (km)', 'RIC_xz.png'),
            ('In-track vs Cross-track', y_r, z_r, 'In-track (km)', 'Cross-track (km)', 'RIC_yz.png')
        ]

        for title, X, Y, xlabel, ylabel, filename in projections:
            plt.figure(figsize=(7, 6))
            plt.plot(X, Y, color='blue')
            plt.scatter([0], [0], color='red', s=60, label='Chief')
            plt.scatter([X[0]], [Y[0]], color='green', s=60, label='Start')
            plt.scatter([X[-1]], [Y[-1]], color='black', s=60, label='End')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()

        # -------------------------
        # Inertial ECI frame
        # -------------------------
        chief_r = states[:, 0:3]
        deputy_r = states[:, 6:9]
        x_c, y_c, z_c = chief_r.T
        x_d, y_d, z_d = deputy_r.T

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_c, y_c, z_c, label='Chief Trajectory', color='red')
        ax.plot(x_d, y_d, z_d, label='Deputy Trajectory', color='blue')
        ax.scatter([x_c[0]], [y_c[0]], [z_c[0]], color='red', s=60, label='Chief Start')
        ax.scatter([x_d[0]], [y_d[0]], [z_d[0]], color='green', s=60, label='Deputy Start')
        ax.set_xlabel('ECI X (km)')
        ax.set_ylabel('ECI Y (km)')
        ax.set_zlabel('ECI Z (km)')
        ax.set_title('Inertial Trajectories (ECI Frame)')
        ax.legend()
        ax.grid(True)

        max_range = np.array([x_c.max()-x_c.min(), y_c.max()-y_c.min(), z_c.max()-z_c.min()]).max() / 2.0
        mid_x = (x_c.max()+x_c.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_x - max_range, mid_x + max_range)
        ax.set_zlim(mid_x - max_range, mid_x + max_range)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "trajectory_ECI.png"))
        plt.close()

        print(f"RIC and ECI trajectory plots saved in {output_dir}")

    plot_trajectories(results_serializable, output_dir)


    def plot_hill_frame_trajectory(results_serializable, output_dir, show_plot=True):
        """
        Generate a 3D plot of the deputy's relative position in the chief's Hill frame over time.
        Saves a PNG to output_dir.
        """

        os.makedirs(output_dir, exist_ok=True)

        states = np.array(results_serializable.get("full_state", []), dtype=float)
        if len(states) == 0:
            print("No trajectory data available for Hill-frame plot. Skipping.")
            return

        relative_positions = states[:, 12:15]  # deputy_rho
        chief_positions = states[:, 0:3]       # chief inertial

        # Initialize array for Hill-frame relative positions
        relative_positions_H = np.zeros_like(relative_positions)

        for k in range(len(relative_positions)):
            r_c = chief_positions[k]
            v_c = states[k, 3:6]
            R_hill = LVLH_DCM(r_c, v_c)  # inertial -> Hill
            relative_positions_H[k] = R_hill @ relative_positions[k]

        # 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(
            relative_positions_H[:, 0],  # radial
            relative_positions_H[:, 1],  # along-track
            relative_positions_H[:, 2],  # cross-track
            label='Deputy Trajectory', color='blue', linewidth=1.5
        )
        # Start and end markers
        ax.scatter([relative_positions_H[0, 0]], [relative_positions_H[0, 1]], [relative_positions_H[0, 2]],
                color='green', s=60, label='Start')
        ax.scatter([relative_positions_H[-1, 0]], [relative_positions_H[-1, 1]], [relative_positions_H[-1, 2]],
                color='black', s=60, label='End')
        # Chief origin
        ax.scatter([0], [0], [0], color='red', s=60, label='Chief (origin)')

        ax.set_xlabel("Radial (x) [km]")
        ax.set_ylabel("Along-track (y) [km]")
        ax.set_zlabel("Cross-track (z) [km]")
        ax.set_title("Deputy Relative Motion in Chief's Hill Frame")
        ax.legend()
        ax.grid(True)
        ax.view_init(elev=20, azim=-60)
        ax.set_box_aspect([1, 1, 1])  # equal aspect

        # Show interactive window if requested
        if show_plot:
            plt.show()

        plt.close()

    # Call the function inside postprocess
    plot_hill_frame_trajectory(results_serializable, output_dir)



    # ---------------------------------
    # Plot Control Accelerations (RGB version)
    # ---------------------------------
    if "control_accel" in results_serializable:
        control_accel = np.array(results_serializable["control_accel"], dtype=float)
        time = np.array(results_serializable["time"], dtype=float)

        if control_accel.ndim == 2 and control_accel.shape[1] == 3:
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

            labels = [r'$a_x$ (km/s$^2$)', r'$a_y$ (km/s$^2$)', r'$a_z$ (km/s$^2$)']
            colors = ['r', 'g', 'b']  # RGB for each axis

            for i in range(3):
                axes[i].plot(time, control_accel[:, i], color=colors[i], linewidth=1.8)
                axes[i].set_ylabel(labels[i])
                axes[i].grid(True)
                axes[i].set_xlim([time[0], time[-1]])

            axes[-1].set_xlabel('Time (s)')
            fig.suptitle('Control Accelerations vs Time', fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(output_dir, "control_accel_plot.png"))
            plt.close()
            print(f"Control acceleration plots saved to {output_dir}")
        else:
            print("Control acceleration data not in expected shape (N×3). Skipping plot.")

