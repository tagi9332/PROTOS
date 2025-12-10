import os
import numpy as np
import matplotlib.pyplot as plt

def plot_ECI_trajectories(results_serializable, output_dir):
    states = np.array(results_serializable.get("full_state", []), dtype=float)
    if len(states) == 0:
        print("No ECI data to plot.")
        return

    # 1. Extract Data
    chief_r = states[:, 0:3]
    deputy_r = states[:, 6:9]
    x_c, y_c, z_c = chief_r.T
    x_d, y_d, z_d = deputy_r.T

    # 2. Setup Plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1]) 

    # 3. Earth Sphere
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

    # 4. Trajectories
    ax.plot(x_c, y_c, z_c, color='red', label='Chief', linewidth=2)
    ax.plot(x_d, y_d, z_d, color='blue', label='Deputy', linewidth=1, linestyle='--')

    # 5. Calculate Equal Limits
    all_data = np.concatenate([x_c, y_c, z_c, x_d, y_d, z_d])
    max_val = max(np.max(np.abs(all_data)), R_earth_km) * 1.1

    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_zlim(-max_val, max_val)

    # 6. Labels
    ax.set_xlabel("ECI X (km)")
    ax.set_ylabel("ECI Y (km)")
    ax.set_zlabel("ECI Z (km)")
    ax.set_title("ECI Trajectories")
    ax.legend()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "trajectory_ECI.png"), dpi=300)
    plt.close()