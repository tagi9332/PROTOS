import os
import numpy as np
import matplotlib.pyplot as plt

def plot_RIC_trajectory(results_serializable, output_dir):
    states = np.array(results_serializable.get("full_state", []), dtype=float)
    if len(states) == 0:
        print("No RIC data to plot.")
        return

    rho = states[:, 12:15]
    x_r, y_r, z_r = rho.T

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_r, y_r, z_r, color='blue', label='Deputy')

    ax.scatter([0], [0], [0], color='red', s=60) # type: ignore

    ax.set_xlabel("Radial (km)")
    ax.set_ylabel("In-track (km)")
    ax.set_zlabel("Cross-track (km)")
    ax.set_title("RIC Relative Trajectory")
    ax.grid(True)
    ax.legend()

    plt.savefig(os.path.join(output_dir, "trajectory_RIC.png"))
    plt.close()