import os
import numpy as np
import matplotlib.pyplot as plt

def plot_ECI_trajectories(results_serializable, output_dir):
    """
    Plots the ECI trajectories of the Chief and all Deputies in 3D around Earth.
    """
    # Extract Chief data
    chief_r = np.array(results_serializable.get("chief", {}).get("r", []), dtype=float)
    deputies = results_serializable.get("deputies", {})

    if len(chief_r) == 0 and not deputies:
        print("No ECI data to plot.")
        return

    # 1. Setup Plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1]) 

    # 2. Earth Sphere
    R_earth_km = 6378.137
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    x_earth = R_earth_km * np.outer(np.cos(u), np.sin(v))
    y_earth = R_earth_km * np.outer(np.sin(u), np.sin(v))
    z_earth = R_earth_km * np.outer(np.ones(np.size(u)), np.cos(v))

    # Surface (transparent blue)
    ax.plot_surface(x_earth, y_earth, z_earth, color='b', alpha=0.1, 
                    linewidth=0, shade=True)
    # Wireframe (solid blue outline)
    ax.plot_wireframe(x_earth, y_earth, z_earth, color='b', alpha=0.6, 
                      rstride=5, cstride=5, linewidth=0.5, label='Earth')

    # 3. Trajectories & Data Collection (for axis limits)
    all_data = []

    # Plot Chief
    if len(chief_r) > 0:
        ax.plot(chief_r[:, 0], chief_r[:, 1], chief_r[:, 2], 
                color='red', label='Chief', linewidth=2)
        all_data.append(chief_r)

    # Plot Deputies with distinct colors
    colors = plt.cm.tab10.colors  # Grab a nice colormap
    for i, (sat_name, sat_data) in enumerate(deputies.items()):
        dep_r = np.array(sat_data.get("r", []), dtype=float)
        if len(dep_r) > 0:
            color = colors[i % len(colors)]
            ax.plot(dep_r[:, 0], dep_r[:, 1], dep_r[:, 2], 
                    color=color, label=sat_name, linewidth=1.5, linestyle='--')
            all_data.append(dep_r)

    # 4. Calculate Equal Limits dynamically based on ALL plotted satellites
    if all_data:
        combined_data = np.vstack(all_data)
        max_val = max(np.max(np.abs(combined_data)), R_earth_km) * 1.1

        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_zlim(-max_val, max_val)

    # 5. Labels and Formatting
    ax.set_xlabel("ECI X (km)")
    ax.set_ylabel("ECI Y (km)")
    ax.set_zlabel("ECI Z (km)")
    ax.set_title("ECI Trajectories")
    
    # Earth wireframe adds an empty legend patch, so we filter it out 
    # to keep the legend looking clean and professional.
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "trajectory_ECI.png"), dpi=300)
    plt.close()