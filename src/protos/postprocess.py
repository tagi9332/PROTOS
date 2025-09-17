import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def _convert_ndarray(obj):
    """Recursively convert any np.ndarray in obj to a list."""
    if isinstance(obj, dict):
        return {k: _convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_ndarray(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def postprocess(gnc_results: dict, output_dir: str):
    """
    Save GNC results and generate trajectory plots.
    Stores full state: chief, deputy inertial, deputy LVLH.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Recursively convert any ndarray to list for JSON
    gnc_serializable = _convert_ndarray(gnc_results)

    # Save GNC results to JSON
    output_json = os.path.join(output_dir, "gnc_results.json")
    with open(output_json, 'w') as f:
        json.dump(gnc_serializable, f, indent=4)
    print(f"GNC results saved to {output_json}")

    # Extract time and full states
    time = gnc_serializable.get("time", [])
    state_inertial = gnc_serializable.get("state_inertial", [])
    state_LVLH = gnc_serializable.get("state_LVLH", [])
    chief_state = gnc_serializable.get("chief_state", [])

    if not time or not state_inertial or not state_LVLH or not chief_state:
        print("Incomplete trajectory data. Skipping plots.")
        return

    # Convert to numpy arrays for plotting
    deputy_inertial = np.array(state_inertial, dtype=float)
    deputy_LVLH = np.array(state_LVLH, dtype=float)
    chief = np.array(chief_state, dtype=float)

    # -------------------------------
    # 3D Plot: Deputy in LVLH
    # -------------------------------
    x, y, z, vx, vy, vz = deputy_LVLH.T
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, color='blue', label='Deputy LVLH Trajectory')
    ax.scatter([0], [0], [0], color='red', s=60, label='Chief (origin)')
    ax.scatter([x[0]], [y[0]], [z[0]], color='green', s=60, label='Deputy Start')
    ax.scatter([x[-1]], [y[-1]], [z[-1]], color='black', s=60, label='Deputy End')
    ax.set_xlabel('Radial (km)')
    ax.set_ylabel('In-track (km)')
    ax.set_zlabel('Cross-track (km)') # type: ignore
    ax.set_title('Deputy Trajectory in LVLH Frame')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_LVLH.png"))
    plt.close()
    print(f"LVLH trajectory plot saved to {os.path.join(output_dir, 'trajectory_LVLH.png')}")

    # -------------------------------
    # Optional 2D Projections in LVLH
    # -------------------------------
    projections = [
        ('Radial vs In-track', x, y, 'x (km)', 'y (km)', 'LVLH_xy.png'),
        ('Radial vs Cross-track', x, z, 'x (km)', 'z (km)', 'LVLH_xz.png'),
        ('In-track vs Cross-track', y, z, 'y (km)', 'z (km)', 'LVLH_yz.png')
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

    # -------------------------------
    # Optional: Save all states in a CSV for analysis
    # -------------------------------
    full_state_array = np.hstack((chief, deputy_inertial, deputy_LVLH))
    csv_file = os.path.join(output_dir, "full_state.csv")
    header = "chief_x,chief_y,chief_z,chief_vx,chief_vy,chief_vz," \
             "deputy_x,deputy_y,deputy_z,deputy_vx,deputy_vy,deputy_vz," \
             "deputy_LVLH_x,deputy_LVLH_y,deputy_LVLH_z,deputy_LVLH_vx," \
             "deputy_LVLH_vy,deputy_LVLH_vz"
    np.savetxt(csv_file, full_state_array, delimiter=',', header=header, comments='')
    print(f"Full state saved to CSV: {csv_file}")
