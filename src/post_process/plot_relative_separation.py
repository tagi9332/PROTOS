import os
import numpy as np
import matplotlib.pyplot as plt

def plot_relative_separation(results_serializable, output_dir):
    """
    Plots stacked subplots of relative separation magnitude and relative velocity magnitude
    for every deputy relative to the chief. Adds an annotation at the closest separation point.
    Saves individual relative_separation_[sat_name].png files.
    """
    time = np.array(results_serializable.get("time", []), dtype=float)
    deputies = results_serializable.get("deputies", {})

    if len(time) == 0 or not deputies:
        print("Time or deputy data missing. Skipping relative separation plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for sat_name, sat_data in deputies.items():
        # Extract relative position and velocity (Hill frame)
        rel_pos = np.array(sat_data.get("rho", []), dtype=float)
        rel_vel = np.array(sat_data.get("rho_dot", []), dtype=float)

        # Ensure data exists and has the correct shape
        if len(rel_pos) == 0 or len(rel_vel) == 0 or rel_pos.shape[1] != 3 or rel_vel.shape[1] != 3:
            print(f"[{sat_name}] Missing or invalid 'rho'/'rho_dot' arrays. Skipping plot.")
            continue

        # Magnitudes
        sep_mag = np.linalg.norm(rel_pos, axis=1)
        vel_mag = np.linalg.norm(rel_vel, axis=1) 

        # Closest approach
        idx_min = np.argmin(sep_mag)
        t_min = time[idx_min]
        sep_min = sep_mag[idx_min]
        vel_min = vel_mag[idx_min]

        # -------------------------
        # Plot
        # -------------------------
        fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

        # Separation magnitude
        axs[0].plot(time, sep_mag, linewidth=2, color='b')
        axs[0].scatter(t_min, sep_min, color='red', zorder=5)

        axs[0].set_ylabel("Separation [km]")
        axs[0].set_title(f"{sat_name} Relative Motion (Chief-Centered)")
        axs[0].grid(True)
        axs[0].set_xlim([time[0], time[-1]])

        # Annotation (data pointer)
        annotation_text = (
            f"Closest Approach\n"
            f"t = {t_min:.1f} s\n"
            f"d = {sep_min:.3f} km"
        )

        axs[0].annotate(
            annotation_text,
            xy=(t_min, sep_min),
            xytext=(0.05, 0.85),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color='black'),
            bbox=dict(boxstyle="round", fc="white", ec="black")
        )

        # Velocity magnitude
        axs[1].plot(time, vel_mag, linewidth=2, color='orange')
        axs[1].scatter(t_min, vel_min, color='red', zorder=5)

        axs[1].set_xlabel("Time [s]")
        axs[1].set_ylabel("Relative Speed [km/s]")
        axs[1].grid(True)
        axs[1].set_xlim([time[0], time[-1]])

        # -------------------------
        # Save
        # -------------------------
        plt.tight_layout()
        safe_name = sat_name.replace(" ", "_").lower()
        filename = os.path.join(output_dir, f"relative_separation_{safe_name}.png")
        plt.savefig(filename, dpi=300)
        plt.close(fig)

