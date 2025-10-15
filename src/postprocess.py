import os
import csv
import numpy as np
import matplotlib.pyplot as plt

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

def postprocess(gnc_results: dict, output_dir: str):
    """
    Save GNC results and generate trajectory plots in the RIC frame.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Recursively convert any ndarray to list (if needed)
    gnc_serializable = _convert_ndarray(gnc_results)

    # Save GNC results to CSV
    output_csv = os.path.join(output_dir, "gnc_results.csv")
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

        # Write data rows (zip to align time with each state vector)
        for t, state_vector in zip(gnc_results["time"], gnc_results["full_state"]):
            writer.writerow([t] + state_vector)

    print(f"GNC results saved to {output_csv}")

    def plot_trajectories(gnc_serializable, output_dir):
        # Extract time and states
        time = gnc_serializable.get("time", [])  # type: ignore
        states = gnc_serializable.get("full_state", [])  # type: ignore

        if not time or not states:
            print("No trajectory data found in gnc_results. Skipping plots.")
            return

        # Convert states to numpy array
        states = np.array(states, dtype=float)  # shape (N, 18)

        # -------------------------
        # Relative RIC frame
        # -------------------------
        rho = states[:, 12:15]       # deputy_rho
        rho_dot = states[:, 15:18]   # deputy_rho_dot

        x_r, y_r, z_r = rho.T  # RIC frame
        vx_r, vy_r, vz_r = rho_dot.T

        # Initial and final relative positions
        x0_r, y0_r, z0_r = x_r[0], y_r[0], z_r[0]
        xf_r, yf_r, zf_r = x_r[-1], y_r[-1], z_r[-1]

        # 3D Relative Trajectory
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_r, y_r, z_r, label='Deputy Trajectory', color='blue')
        ax.scatter([0], [0], [0], color='red', s=60, label='Chief (origin)') # type: ignore
        ax.scatter([x0_r], [y0_r], [z0_r], color='green', s=60, label='Deputy Start') # type: ignore
        ax.scatter([xf_r], [yf_r], [zf_r], color='black', s=60, label='Deputy End') # type: ignore
        ax.set_xlabel('Radial (km)')
        ax.set_ylabel('In-track (km)')
        ax.set_zlabel('Cross-track (km)') # type: ignore
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
        ax.scatter([x_c[0]], [y_c[0]], [z_c[0]], color='red', s=60, label='Chief Start') # type: ignore
        ax.scatter([x_d[0]], [y_d[0]], [z_d[0]], color='green', s=60, label='Deputy Start') # type: ignore
        ax.set_xlabel('ECI X (km)')
        ax.set_ylabel('ECI Y (km)')
        ax.set_zlabel('ECI Z (km)') # type: ignore
        ax.set_title('Inertial Trajectories (ECI Frame)')
        ax.legend()
        ax.grid(True)
        # Set axis to equal for proper aspect ratio
        max_range = np.array([x_c.max()-x_c.min(), y_c.max()-y_c.min(), z_c.max()-z_c.min()]).max() / 2.0
        mid_x = (x_c.max()+x_c.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_x - max_range, mid_x + max_range)
        ax.set_zlim(mid_x - max_range, mid_x + max_range) # type: ignore

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "trajectory_ECI.png"))
        plt.close()

        print(f"RIC and ECI trajectory plots saved in {output_dir}")

    plot_trajectories(gnc_serializable, output_dir)
