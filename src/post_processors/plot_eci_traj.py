import os
import numpy as np
import matplotlib.pyplot as plt

def plot_ECI_trajectories(results_serializable, output_dir):
    states = np.array(results_serializable.get("full_state", []), dtype=float)
    if len(states) == 0:
        print("No ECI data to plot.")
        return

    chief_r = states[:, 0:3]
    deputy_r = states[:, 6:9]
    x_c, y_c, z_c = chief_r.T
    x_d, y_d, z_d = deputy_r.T

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_c, y_c, z_c, color='red', label='Chief')
    ax.plot(x_d, y_d, z_d, color='blue', label='Deputy')

    ax.set_xlabel("ECI X (km)")
    ax.set_ylabel("ECI Y (km)")
    ax.set_zlabel("ECI Z (km)")
    ax.set_title("ECI Trajectories")
    ax.grid(True)
    ax.legend()

    plt.savefig(os.path.join(output_dir, "trajectory_ECI.png"))
    plt.close()