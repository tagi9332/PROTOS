import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def save_plane_views(results_serializable, vehicle_dirs, main_output_dir):
    """
    Saves individual static 2D plane views (XY, XZ, YZ) of the relative 
    trajectory for each deputy into their specific results folders, AND
    saves a combined plot containing all deputies into the main output directory.
    """
    deputies = results_serializable.get("deputies", {})
    if not deputies:
        return

    # ==========================================
    # 1. Combined Plot Setup
    # ==========================================
    fig_all, axs_all = plt.subplots(1, 3, figsize=(18, 5))
    fig_all.suptitle('All Vehicles Relative Motion (Hill Frame)', fontsize=16)

    # Plot Chief on the combined plot
    axs_all[0].scatter(0, 0, color='r', marker='*', s=150, label='Chief', zorder=5)
    axs_all[1].scatter(0, 0, color='r', marker='*', s=150, zorder=5)
    axs_all[2].scatter(0, 0, color='r', marker='*', s=150, zorder=5)

    # Generate distinct colors for the combined plot
    colors = cm.get_cmap('tab10', len(deputies))

    # ==========================================
    # 2. Iterate Vehicles (Individual + Combined)
    # ==========================================
    for i, (sat_name, sat_data) in enumerate(deputies.items()):
        sat_output_dir = vehicle_dirs.get(sat_name)
        if not sat_output_dir:
            continue

        rho = np.array(sat_data.get("rho", []), dtype=float)
        
        # Validate data
        if len(rho) == 0 or rho.shape[1] != 3:
            continue

        r_x, r_y, r_z = rho[:, 0], rho[:, 1], rho[:, 2]
        c = colors(i)

        # ------------------------------------------
        # Add to Combined Plot
        # ------------------------------------------
        # In-Plane
        axs_all[0].plot(r_y, r_x, label=sat_name, color=c)
        axs_all[0].scatter(r_y[0], r_x[0], color=c, marker='o', s=20) # Start
        axs_all[0].scatter(r_y[-1], r_x[-1], color=c, marker='x', s=20) # End
        
        # Side View
        axs_all[1].plot(r_y, r_z, color=c)
        axs_all[1].scatter(r_y[0], r_z[0], color=c, marker='o', s=20)
        axs_all[1].scatter(r_y[-1], r_z[-1], color=c, marker='x', s=20)
        
        # Approach View
        axs_all[2].plot(r_z, r_x, color=c)
        axs_all[2].scatter(r_z[0], r_x[0], color=c, marker='o', s=20)
        axs_all[2].scatter(r_z[-1], r_x[-1], color=c, marker='x', s=20)

        # ------------------------------------------
        # Individual Plot Generation
        # ------------------------------------------
        fig_ind, axs_ind = plt.subplots(1, 3, figsize=(18, 5))
        fig_ind.suptitle(f'{sat_name.capitalize()} Relative Motion (Hill Frame)', fontsize=16)

        # In-Plane Motion (Along-Track vs Radial)
        axs_ind[0].plot(r_y, r_x, label='Trajectory', color='b')
        axs_ind[0].scatter(0, 0, color='r', marker='o', label='Chief')
        axs_ind[0].scatter(r_y[0], r_x[0], color='g', marker='o', label='Start')
        axs_ind[0].scatter(r_y[-1], r_x[-1], color='k', marker='o', label='End')
        axs_ind[0].set_xlabel('Along-Track (y) [km]')
        axs_ind[0].set_ylabel('Radial (x) [km]')
        axs_ind[0].set_title('In-Plane Motion (y vs x)')
        axs_ind[0].grid(True)
        axs_ind[0].axis('equal')  
        axs_ind[0].invert_xaxis() 

        # Side View (Along-Track vs Cross-Track)
        axs_ind[1].plot(r_y, r_z, color='b')
        axs_ind[1].scatter(0, 0, color='r', marker='o', label='Chief')
        axs_ind[1].scatter(r_y[0], r_z[0], color='g', marker='o', label='Start')
        axs_ind[1].scatter(r_y[-1], r_z[-1], color='k', marker='o', label='End')
        axs_ind[1].set_xlabel('Along-Track (y) [km]')
        axs_ind[1].set_ylabel('Cross-Track (z) [km]')
        axs_ind[1].set_title('Side View (y vs z)')
        axs_ind[1].grid(True)
        axs_ind[1].axis('equal')
        axs_ind[1].invert_xaxis()

        # Approach View (Cross-Track vs Radial)
        axs_ind[2].plot(r_z, r_x, color='b')
        axs_ind[2].scatter(0, 0, color='r', marker='o', label='Chief')
        axs_ind[2].scatter(r_z[0], r_x[0], color='g', marker='o', label='Start')
        axs_ind[2].scatter(r_z[-1], r_x[-1], color='k', marker='o', label='End')
        axs_ind[2].set_xlabel('Cross-Track (z) [km]')
        axs_ind[2].set_ylabel('Radial (x) [km]')
        axs_ind[2].set_title('Approach View (z vs x)')
        axs_ind[2].grid(True)
        axs_ind[2].axis('equal')

        axs_ind[0].legend(loc='upper right')

        # Save individual figure to the deputy-specific folder
        filename_ind = os.path.join(sat_output_dir, f"RIC_plane_views_{sat_name}.png")
        plt.tight_layout()
        fig_ind.savefig(filename_ind, dpi=300)
        plt.close(fig_ind)

    # ==========================================
    # 3. Format and Save Combined Plot
    # ==========================================
    # Format In-Plane
    axs_all[0].set_xlabel('Along-Track (y) [km]')
    axs_all[0].set_ylabel('Radial (x) [km]')
    axs_all[0].set_title('In-Plane Motion (y vs x)')
    axs_all[0].grid(True)
    axs_all[0].axis('equal')  
    axs_all[0].invert_xaxis() 

    # Format Side View
    axs_all[1].set_xlabel('Along-Track (y) [km]')
    axs_all[1].set_ylabel('Cross-Track (z) [km]')
    axs_all[1].set_title('Side View (y vs z)')
    axs_all[1].grid(True)
    axs_all[1].axis('equal')
    axs_all[1].invert_xaxis()

    # Format Approach View
    axs_all[2].set_xlabel('Cross-Track (z) [km]')
    axs_all[2].set_ylabel('Radial (x) [km]')
    axs_all[2].set_title('Approach View (z vs x)')
    axs_all[2].grid(True)
    axs_all[2].axis('equal')

    # Ensure legend fits nicely on the combined plot
    # axs_all[0].legend(loc='best', fontsize='small')

    # Save combined figure to main output directory
    filename_all = os.path.join(main_output_dir, "RIC_plane_views_combined.png")
    plt.tight_layout()
    fig_all.savefig(filename_all, dpi=300)
    plt.close(fig_all)


# ==============================================================================
# 3D Isometric View (Combined Plot)
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
    ax.set_title('3D Isometric View (Hill Frame)')
    
    # Calculate Equal Bounding Box for true geometry
    combined_rho = np.vstack(all_rho)
    _set_axes_equal(ax, combined_rho)

    # Set view angle
    ax.view_init(elev=30, azim=135)
    # ax.legend()

    filename = os.path.join(output_dir, "RIC_iso_view.png")
    plt.savefig(filename, dpi=300)
    plt.close(fig)


def _set_axes_equal(ax, data):
    """
    Helper function to force equal aspect ratio for 3D plots.
    Uses the provided data to calculate the maximum bounds.
    """
    max_val = np.max(np.abs(data))
    plot_radius = max_val * 1.1 

    ax.set_xlim3d([-plot_radius, plot_radius])
    ax.set_ylim3d([-plot_radius, plot_radius])
    ax.set_zlim3d([-plot_radius, plot_radius])