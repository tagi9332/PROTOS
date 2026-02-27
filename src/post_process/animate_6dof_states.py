import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, Any

# Maximum number of animation frames to keep GIF file size manageable
_MAX_FRAMES = 200


def _build_satellites(results: Dict[str, Any]):
    """Return a list of (name, data_dict) tuples for chief + all deputies."""
    sats = []
    chief_data = results.get("chief", {})
    if chief_data:
        sats.append(("chief", chief_data))
    for name, data in results.get("deputies", {}).items():
        sats.append((name, data))
    return sats


def animate_6dof_states(results: Dict[str, Any], output_dir: str) -> None:
    """
    Create an animated GIF of the 6DOF states (ECI position, velocity, and,
    when in 6DOF mode, quaternion and angular rate) for all chief and deputy
    trajectories in the simulation.

    The animation is saved as ``animation_6dof_states.gif`` inside *output_dir*.

    Parameters
    ----------
    results : dict
        Post-processed results dictionary as produced by
        ``package_simulation_results``.  Must contain at minimum the keys
        ``"time"``, ``"chief"``, ``"deputies"``, and ``"is_6dof"``.
    output_dir : str
        Directory in which the GIF will be saved.
    """
    time = np.array(results.get("time", []), dtype=float)
    if len(time) == 0:
        print("No time data available. Skipping animation.")
        return

    is_6dof = results.get("is_6dof", False)
    sats = _build_satellites(results)
    if not sats:
        print("No satellite data available. Skipping animation.")
        return

    # ------------------------------------------------------------------ #
    # Frame decimation – cap animation length to _MAX_FRAMES              #
    # ------------------------------------------------------------------ #
    n_total = len(time)
    step = max(1, n_total // _MAX_FRAMES)
    frame_indices = list(range(0, n_total, step))
    n_frames = len(frame_indices)

    # Colour palette (one colour per satellite)
    palette = plt.cm.tab10.colors
    sat_colors = {name: palette[i % len(palette)] for i, (name, _) in enumerate(sats)}

    # ------------------------------------------------------------------ #
    # Figure layout                                                        #
    # ------------------------------------------------------------------ #
    # Rows:
    #   Row 0 : 3-D ECI trajectory  (spans all columns)
    #   Row 1 : ECI position components  r_x, r_y, r_z
    #   Row 2 : ECI velocity components  v_x, v_y, v_z
    #   Row 3 : Quaternions q0..q3          (6DOF only)
    #   Row 4 : Angular rates  ω_x, ω_y, ω_z (6DOF only)
    #
    # Columns 0-2 carry the time-series subplots.

    n_ts_rows = 2 + (2 if is_6dof else 0)  # 2 (3DOF) or 4 (6DOF) time-series rows
    n_rows = 1 + n_ts_rows

    fig = plt.figure(figsize=(14, 3 * n_rows))
    gs = fig.add_gridspec(n_rows, 3, hspace=0.55, wspace=0.35)

    # --- 3-D ECI subplot (row 0, spans all 3 columns) ---
    ax3d = fig.add_subplot(gs[0, :], projection="3d")

    # --- Time-series subplots ---
    # Row 1: r components
    ax_rx = fig.add_subplot(gs[1, 0])
    ax_ry = fig.add_subplot(gs[1, 1])
    ax_rz = fig.add_subplot(gs[1, 2])

    # Row 2: v components
    ax_vx = fig.add_subplot(gs[2, 0])
    ax_vy = fig.add_subplot(gs[2, 1])
    ax_vz = fig.add_subplot(gs[2, 2])

    ax_qw = ax_qx = ax_qy = None          # q0, q1, q2 (q3 overlaid on ax_qy)
    ax_wx = ax_wy = ax_wz = None
    if is_6dof:
        # Row 3: three quaternion component axes (q0, q1, q2); q3 is overlaid on ax_qy
        ax_qw = fig.add_subplot(gs[3, 0])
        ax_qx = fig.add_subplot(gs[3, 1])
        ax_qy = fig.add_subplot(gs[3, 2])

        # Row 4: angular rate components (ω_x, ω_y, ω_z)
        ax_wx = fig.add_subplot(gs[4, 0])
        ax_wy = fig.add_subplot(gs[4, 1])
        ax_wz = fig.add_subplot(gs[4, 2])

    # ------------------------------------------------------------------ #
    # Pre-compute ECI limits for 3-D plot                                 #
    # ------------------------------------------------------------------ #
    R_earth_km = 6378.137
    all_r = []
    for _, sat_data in sats:
        r = np.array(sat_data.get("r", []), dtype=float)
        if r.ndim == 2 and r.shape[1] == 3:
            all_r.append(r)

    if all_r:
        combined = np.vstack(all_r)
        max_val = max(np.max(np.abs(combined)), R_earth_km) * 1.1
    else:
        max_val = R_earth_km * 1.5

    # ------------------------------------------------------------------ #
    # Draw static elements on 3-D axes                                    #
    # ------------------------------------------------------------------ #
    u_e = np.linspace(0, 2 * np.pi, 40)
    v_e = np.linspace(0, np.pi, 40)
    x_e = R_earth_km * np.outer(np.cos(u_e), np.sin(v_e))
    y_e = R_earth_km * np.outer(np.sin(u_e), np.sin(v_e))
    z_e = R_earth_km * np.outer(np.ones(np.size(u_e)), np.cos(v_e))

    ax3d.plot_surface(x_e, y_e, z_e, color="b", alpha=0.08, linewidth=0, shade=True)
    ax3d.plot_wireframe(x_e, y_e, z_e, color="b", alpha=0.4,
                        rstride=5, cstride=5, linewidth=0.4)

    # Full trajectory lines (faded) and current-position markers
    traj_lines_3d = {}
    pos_markers_3d = {}
    for name, sat_data in sats:
        r = np.array(sat_data.get("r", []), dtype=float)
        color = sat_colors[name]
        if r.ndim == 2 and r.shape[1] == 3 and len(r) > 0:
            ax3d.plot(r[:, 0], r[:, 1], r[:, 2],
                      color=color, linewidth=1.0, alpha=0.25, zorder=1)
            line, = ax3d.plot([], [], [], color=color,
                              linewidth=1.5, zorder=2, label=name)
            marker, = ax3d.plot([], [], [], "o", color=color,
                                markersize=6, zorder=3)
            traj_lines_3d[name] = (line, r)
            pos_markers_3d[name] = (marker, r)

    ax3d.set_xlim(-max_val, max_val)
    ax3d.set_ylim(-max_val, max_val)
    ax3d.set_zlim(-max_val, max_val)
    ax3d.set_xlabel("X (km)", fontsize=8)
    ax3d.set_ylabel("Y (km)", fontsize=8)
    ax3d.set_zlabel("Z (km)", fontsize=8)
    ax3d.set_title("ECI Trajectories", fontsize=9)
    ax3d.legend(loc="upper right", fontsize=7)

    # ------------------------------------------------------------------ #
    # Time-series: draw full curves (faded) and animated vertical lines  #
    # ------------------------------------------------------------------ #
    ts_axes_config = [
        (ax_rx, "r", 0, "r$_x$ (km)"),
        (ax_ry, "r", 1, "r$_y$ (km)"),
        (ax_rz, "r", 2, "r$_z$ (km)"),
        (ax_vx, "v", 0, "v$_x$ (km/s)"),
        (ax_vy, "v", 1, "v$_y$ (km/s)"),
        (ax_vz, "v", 2, "v$_z$ (km/s)"),
    ]
    if is_6dof:
        ts_axes_config += [
            (ax_qw, "q", 0, "q$_0$"),
            (ax_qx, "q", 1, "q$_1$"),
            (ax_qy, "q", 2, "q$_2$"),
            (ax_wx, "omega", 0, "$\\omega_x$ (rad/s)"),
            (ax_wy, "omega", 1, "$\\omega_y$ (rad/s)"),
            (ax_wz, "omega", 2, "$\\omega_z$ (rad/s)"),
        ]
        # q has 4 components; add q3 separately paired with ax_qy (reuse slot)
        # We skip ax_qz since layout is 3-column; q3 overlaid on ax_qy below.

    vlines = []  # (axes, vline_artist) pairs for animated vertical lines

    for ax_ts, key, comp, ylabel in ts_axes_config:
        if ax_ts is None:
            continue
        ax_ts.set_xlabel("Time (s)", fontsize=7)
        ax_ts.set_ylabel(ylabel, fontsize=7)
        ax_ts.tick_params(labelsize=6)
        ax_ts.grid(True, linewidth=0.4)

        for name, sat_data in sats:
            arr = np.array(sat_data.get(key, []), dtype=float)
            color = sat_colors[name]
            if arr.ndim == 2 and arr.shape[1] > comp and len(arr) == n_total:
                ax_ts.plot(time, arr[:, comp], color=color,
                           linewidth=0.8, alpha=0.55, label=name)

        ax_ts.legend(fontsize=6, loc="upper right")
        if len(time) > 0:
            vl = ax_ts.axvline(time[0], color="k", linewidth=0.8, linestyle="--")
            vlines.append((ax_ts, vl))

    # If 6DOF: overlay q3 on the ax_qy panel so all 4 quaternion components visible
    if is_6dof and ax_qy is not None:
        for name, sat_data in sats:
            q = np.array(sat_data.get("q", []), dtype=float)
            color = sat_colors[name]
            if q.ndim == 2 and q.shape[1] == 4 and len(q) == n_total:
                ax_qy.plot(time, q[:, 3], color=color,
                           linewidth=0.8, alpha=0.55, linestyle=":")
        ax_qy.set_ylabel("q$_2$ / q$_3$", fontsize=7)

    # Time label
    time_text = fig.text(0.5, 0.98, "", ha="center", va="top",
                         fontsize=9, fontweight="bold")

    # ------------------------------------------------------------------ #
    # Animation update function                                           #
    # ------------------------------------------------------------------ #
    def _update(frame_num):
        idx = frame_indices[frame_num]
        t_now = time[idx]

        # 3-D current-position markers and growing trajectory lines
        for name, (line, r) in traj_lines_3d.items():
            line.set_data(r[:idx + 1, 0], r[:idx + 1, 1])
            line.set_3d_properties(r[:idx + 1, 2])
        for name, (marker, r) in pos_markers_3d.items():
            marker.set_data([r[idx, 0]], [r[idx, 1]])
            marker.set_3d_properties([r[idx, 2]])

        # Vertical time-indicator lines
        for _, vl in vlines:
            vl.set_xdata([t_now, t_now])

        time_text.set_text(f"t = {t_now:.1f} s")
        return []

    # ------------------------------------------------------------------ #
    # Build and save animation                                            #
    # ------------------------------------------------------------------ #
    fps = max(5, min(20, n_frames // 10))
    anim = animation.FuncAnimation(
        fig, _update, frames=n_frames, interval=1000 // fps, blit=False
    )

    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, "animation_6dof_states.gif")

    writer = animation.PillowWriter(fps=fps)
    anim.save(gif_path, writer=writer)
    plt.close(fig)

    print(f"Animation saved to {gif_path}")
