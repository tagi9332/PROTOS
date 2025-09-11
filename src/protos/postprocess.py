import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting


def postprocess(gnc_results: dict, output_dir: str):
    """
    Save GNC results and generate trajectory plots in the RIC frame.
    
    gnc_results: dict with keys 'time' and 'state'
    output_dir: directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save GNC results to JSON
    output_json = os.path.join(output_dir, "gnc_results.json")
    with open(output_json, 'w') as f:
        json.dump(gnc_results, f, indent=4)
    print(f"GNC results saved to {output_json}")
    
    # Extract time and states
    time = gnc_results.get("time", [])
    states = gnc_results.get("state", [])  # [x, y, z, vx, vy, vz]
    
    if not time or not states:
        print("No trajectory data found in gnc_results. Skipping plots.")
        return
    
    states = [list(map(float, s)) for s in states]  # ensure floats
    states = list(zip(*states))  # transpose for easy access
    x, y, z, vx, vy, vz = states
    
    # Initial and final positions
    x0, y0, z0 = x[0], y[0], z[0]
    xf, yf, zf = x[-1], y[-1], z[-1]
    
    # 3D Trajectory Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='Deputy Trajectory', color='blue')
    ax.scatter([0], [0], [0], color='red', s=60, label='Chief (origin)') # type: ignore
    ax.scatter([x0], [y0], [z0], color='green', s=60, label='Deputy Start') # type: ignore
    ax.scatter([xf], [yf], [zf], color='black', s=60, label='Deputy End') # type: ignore
    ax.set_xlabel('Radial (km)')
    ax.set_ylabel('In-track (km)')
    ax.set_zlabel('Cross-track (km)')
    ax.set_title('Deputy Trajectory in RIC Frame')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    traj_plot_path = os.path.join(output_dir, "trajectory_RIC.png")
    plt.savefig(traj_plot_path)
    plt.close()
    print(f"Trajectory plot saved to {traj_plot_path}")
    
    # Optional: 2D projections
    projections = [
        ('Radial vs In-track', x, y, 'x (km)', 'y (km)', 'RIC_xy.png'),
        ('Radial vs Cross-track', x, z, 'x (km)', 'z (km)', 'RIC_xz.png'),
        ('In-track vs Cross-track', y, z, 'y (km)', 'z (km)', 'RIC_yz.png')
    ]
    
    for title, X, Y, xlabel, ylabel, filename in projections:
        plt.figure(figsize=(7, 6))
        plt.plot(X, Y, color='blue')
        plt.scatter([0], [0], color='red', s=60, label='Chief')
        plt.scatter([X[0]], [Y[0]], color='green', s=60, label='Start')
        plt.scatter([X[-1]], [Y[-1]], color='black', s=60, label='End')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"Projection plot saved to {os.path.join(output_dir, filename)}")
