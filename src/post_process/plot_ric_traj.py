import os
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def save_plane_views(results_serializable, output_dir):
    """
    Saves static 2D plane views (XY, XZ, YZ) of the relative trajectory.
    
    Args:
        results_serializable (dict): Simulation results.
        output_dir (str): Directory to save the figure.
    """
    states = np.array(results_serializable.get("full_state", []), dtype=float)
    if len(states) == 0:
        return

    # Extract LVLH positions: x (Radial), y (Along-Track), z (Cross-Track)
    r_x = states[:, 12]
    r_y = states[:, 13]
    r_z = states[:, 14]

    os.makedirs(output_dir, exist_ok=True)

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Relative Motion Plane Views (Hill Frame)', fontsize=16)

    # 1. In-Plane Motion (Radial vs Along-Track) - "Top Down"
    axs[0].plot(r_y, r_x, label='Trajectory', color='b')
    axs[0].scatter(0, 0, color='r', marker='o', label='Chief')
    axs[0].scatter(r_y[0], r_x[0], color='g', marker='o', label='Start')
    axs[0].scatter(r_y[-1], r_x[-1], color='k', marker='o', label='End')
    axs[0].set_xlabel('Along-Track (y) [km]')
    axs[0].set_ylabel('Radial (x) [km]')
    axs[0].set_title('In-Plane Motion (y vs x)')
    axs[0].grid(True)
    axs[0].axis('equal')  # Crucial for orbital geometry
    axs[0].invert_xaxis() # Standard convention: +y is velocity direction (left)

    # 2. Out-of-Plane Motion (Along-Track vs Cross-Track) - "Side View"
    axs[1].plot(r_y, r_z, color='b')
    axs[1].scatter(0, 0, color='r', marker='o', label='Chief')
    axs[1].scatter(r_y[0], r_z[0], color='g', marker='o', label='Start')
    axs[1].scatter(r_y[-1], r_z[-1], color='k', marker='o', label='End')
    axs[1].set_xlabel('Along-Track (y) [km]')
    axs[1].set_ylabel('Cross-Track (z) [km]')
    axs[1].set_title('Side View (y vs z)')
    axs[1].grid(True)
    axs[1].axis('equal')
    axs[1].invert_xaxis()

    # 3. Approach View (Cross-Track vs Radial) - "Barrel View"
    axs[2].plot(r_z, r_x, color='b')
    axs[2].scatter(0, 0, color='r', marker='o', label='Chief')
    axs[2].scatter(r_z[0], r_x[0], color='g', marker='o', label='Start')
    axs[2].scatter(r_z[-1], r_x[-1], color='k', marker='o', label='End')
    axs[2].set_xlabel('Cross-Track (z) [km]')
    axs[2].set_ylabel('Radial (x) [km]')
    axs[2].set_title('Approach View (z vs x)')
    axs[2].grid(True)
    axs[2].axis('equal')

    # Set legends
    axs[0].legend()
    # Set legend location to upper right for clarity
    axs[0].legend(loc='upper right')

    # Save
    filename = os.path.join(output_dir, "RIC_plane_views.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)


def save_iso_view(results_serializable, output_dir):
    """
    Saves a static 3D isometric view of the relative trajectory.
    """
    states = np.array(results_serializable.get("full_state", []), dtype=float)
    if len(states) == 0:
        return

    rel_H = states[:, 12:15]
    
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot lines and markers
    ax.plot(rel_H[:, 0], rel_H[:, 1], rel_H[:, 2], label='Deputy Path', color='blue')
    ax.scatter(0, 0, 0, color='red', marker='o', label='Chief')
    ax.scatter(rel_H[0, 0], rel_H[0, 1], rel_H[0, 2], color='green', label='Start')
    ax.scatter(rel_H[-1, 0], rel_H[-1, 1], rel_H[-1, 2], color='black', label='End')

    ax.set_xlabel('Radial (x) [km]')
    ax.set_ylabel('Along-Track (y) [km]')
    ax.set_zlabel('Cross-Track (z) [km]')
    ax.set_title('3D Isometric View (Hill Frame)')
    
    # Force Aspect Ratio to be Equal (Critical for 3D Orbits)
    set_axes_equal(ax)

    # Set view angle (Elevation, Azimuth) for "Isometric" feel
    ax.view_init(elev=30, azim=135)

    # Set legends
    ax.legend()

    filename = os.path.join(output_dir, "RIC_iso_view.png")
    plt.savefig(filename, dpi=300)
    plt.close(fig)



def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
