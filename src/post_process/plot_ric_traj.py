import os
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 2D Plane Views (Individual plots per deputy)
# ==============================================================================
def save_plane_views(results_serializable, output_dir):
    """
    Saves individual static 2D plane views (XY, XZ, YZ) of the relative 
    trajectory for each deputy in the Hill (LVLH) frame.
    """
    deputies = results_serializable.get("deputies", {})
    if not deputies:
        return

    os.makedirs(output_dir, exist_ok=True)

    for sat_name, sat_data in deputies.items():
        rho = np.array(sat_data.get("rho", []), dtype=float)
        
        # Validate data
        if len(rho) == 0 or rho.shape[1] != 3:
            continue

        r_x, r_y, r_z = rho[:, 0], rho[:, 1], rho[:, 2]

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{sat_name} Relative Motion Plane Views (Hill Frame)', fontsize=16)

        # 1. In-Plane Motion (Radial vs Along-Track) - "Top Down"
        axs[0].plot(r_y, r_x, label='Trajectory', color='b')
        axs[0].scatter(0, 0, color='r', marker='o', label='Chief')
        axs[0].scatter(r_y[0], r_x[0], color='g', marker='o', label='Start')
        axs[0].scatter(r_y[-1], r_x[-1], color='k', marker='o', label='End')
        axs[0].set_xlabel('Along-Track (y) [km]')
        axs[0].set_ylabel('Radial (x) [km]')
        axs[0].set_title('In-Plane Motion (y vs x)')
        axs[0].grid(True)
        axs[0].axis('equal')  
        axs[0].invert_xaxis() # +y is velocity direction (left)

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

        axs[0].legend(loc='upper right')

        safe_name = sat_name.replace(" ", "_").lower()
        filename = os.path.join(output_dir, f"RIC_plane_views_{safe_name}.png")
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close(fig)


# ==============================================================================
# 3D Isometric View (Combined Swarm Plot)
# ==============================================================================
def save_iso_view(results_serializable, output_dir):
    """
    Saves a single 3D isometric view of the relative trajectory, 
    displaying all deputies around the Chief.
    """
    deputies = results_serializable.get("deputies", {})
    if not deputies:
        return

    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the Chief at the origin
    ax.scatter(0, 0, 0, color='red', marker='o', s=50, label='Chief')

    colors = plt.cm.tab10.colors
    all_rho = []

    # Plot each Deputy
    for i, (sat_name, sat_data) in enumerate(deputies.items()):
        rho = np.array(sat_data.get("rho", []), dtype=float)
        if len(rho) == 0 or rho.shape[1] != 3:
            continue
            
        color = colors[i % len(colors)]
        ax.plot(rho[:, 0], rho[:, 1], rho[:, 2], label=f'{sat_name} Path', color=color)
        ax.scatter(rho[0, 0], rho[0, 1], rho[0, 2], color='green', marker='^') # Start
        ax.scatter(rho[-1, 0], rho[-1, 1], rho[-1, 2], color='black', marker='x') # End
        
        all_rho.append(rho)

    if not all_rho:
        plt.close(fig)
        return

    ax.set_xlabel('Radial (x) [km]')
    ax.set_ylabel('Along-Track (y) [km]')
    ax.set_zlabel('Cross-Track (z) [km]')
    ax.set_title('Swarm 3D Isometric View (Hill Frame)')
    
    # Calculate Equal Bounding Box for true geometry
    combined_rho = np.vstack(all_rho)
    _set_axes_equal(ax, combined_rho)

    # Set view angle (Elevation, Azimuth) for "Isometric" feel
    ax.view_init(elev=30, azim=135)
    ax.legend()

    filename = os.path.join(output_dir, "RIC_iso_view_swarm.png")
    plt.savefig(filename, dpi=300)
    plt.close(fig)


def _set_axes_equal(ax, data):
    """
    Helper function to force equal aspect ratio for 3D plots.
    Uses the provided data to calculate the maximum bounds.
    """
    max_val = np.max(np.abs(data))
    
    # Adding a 10% buffer so the data doesn't touch the edge of the box
    plot_radius = max_val * 1.1 

    ax.set_xlim3d([-plot_radius, plot_radius])
    ax.set_ylim3d([-plot_radius, plot_radius])
    ax.set_zlim3d([-plot_radius, plot_radius])