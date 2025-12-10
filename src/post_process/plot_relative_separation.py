import os
import numpy as np
import matplotlib.pyplot as plt


def plot_relative_separation(results_serializable, output_dir):
    """
    Plot stacked subplots of:
      1) Relative separation magnitude
      2) Relative velocity magnitude

    Adds an annotation at the closest separation point.

    Assumes:
        full_state[:, 12:15] = relative position [km]
        full_state[:, 15:18] = relative velocity [km/s]
        time in results_serializable["time"] [s]
    """

    states = np.array(results_serializable.get("full_state", []), dtype=float)
    time = np.array(results_serializable.get("time", []), dtype=float)

    if len(states) == 0 or len(time) == 0:
        return

    # Relative position & velocity in Hill frame
    rel_pos = states[:, 12:15]     # km
    rel_vel = states[:, 15:18]     # km/s

    # Magnitudes
    sep_mag = np.linalg.norm(rel_pos, axis=1)   # km
    vel_mag = np.linalg.norm(rel_vel, axis=1)   # km/s

    # Closest approach
    idx_min = np.argmin(sep_mag)
    t_min = time[idx_min]
    sep_min = sep_mag[idx_min]
    vel_min = vel_mag[idx_min]

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Plot
    # -------------------------
    fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # ---- Separation magnitude ----
    axs[0].plot(time, sep_mag, linewidth=2)
    axs[0].scatter(t_min, sep_min, color='red', zorder=5)

    axs[0].set_ylabel("Separation [km]")
    axs[0].set_title("Chief–Deputy Relative Motion")
    axs[0].grid(True)

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
        arrowprops=dict(arrowstyle="->"),
        bbox=dict(boxstyle="round", fc="white", ec="black")
    )

    # ---- Velocity magnitude ----
    axs[1].plot(time, vel_mag, linewidth=2)
    axs[1].scatter(t_min, vel_min, color='red', zorder=5)

    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Relative Speed [km/s]")
    axs[1].grid(True)

    # -------------------------
    # Save
    # -------------------------
    plt.tight_layout()
    filename = os.path.join(output_dir, "relative_separation_velocity.png")
    plt.savefig(filename, dpi=300)
    plt.close(fig)

    print("Saved relative separation & velocity plot:", filename)
