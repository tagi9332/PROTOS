import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Dict, Any

_MAX_FRAMES = 200

def get_eci_to_ric_dcm(r_chief: np.ndarray, v_chief: np.ndarray) -> np.ndarray:
    r_norm = np.linalg.norm(r_chief, axis=1)[:, None]
    u_R = r_chief / r_norm
    
    h = np.cross(r_chief, v_chief)
    h_norm = np.linalg.norm(h, axis=1)[:, None]
    u_C = h / h_norm
    
    u_I = np.cross(u_C, u_R)
    
    dcm = np.empty((len(r_chief), 3, 3))
    dcm[:, 0, :] = u_R
    dcm[:, 1, :] = u_I
    dcm[:, 2, :] = u_C
    return dcm

def get_body_to_eci_dcm(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    dcm = np.zeros((len(w), 3, 3))
    
    dcm[:, 0, 0] = 1 - 2*y**2 - 2*z**2
    dcm[:, 0, 1] = 2*x*y - 2*w*z
    dcm[:, 0, 2] = 2*x*z + 2*w*y
    
    dcm[:, 1, 0] = 2*x*y + 2*w*z
    dcm[:, 1, 1] = 1 - 2*x**2 - 2*z**2
    dcm[:, 1, 2] = 2*y*z - 2*w*x
    
    dcm[:, 2, 0] = 2*x*z - 2*w*y
    dcm[:, 2, 1] = 2*y*z + 2*w*x
    dcm[:, 2, 2] = 1 - 2*x**2 - 2*y**2
    return dcm

def generate_base_cone(fov_deg: float, length: float, resolution: int = 15):
    half_angle = np.radians(fov_deg / 2.0)
    z = np.linspace(0, length, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    Z, THETA = np.meshgrid(z, theta)
    R = Z * np.tan(half_angle)
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    return X, Y, Z

def generate_base_cube(size: float):
    """Generates the 8 vertices and 6 faces of a cube centered at the origin."""
    s = size / 2.0
    vertices = np.array([
        [-s, -s, -s], [ s, -s, -s], [ s,  s, -s], [-s,  s, -s],
        [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s]
    ])
    
    faces_idx = [
        [0, 1, 2, 3], # Bottom
        [4, 5, 6, 7], # Top
        [0, 1, 5, 4], # Front
        [1, 2, 6, 5], # Right
        [2, 3, 7, 6], # Back
        [3, 0, 4, 7]  # Left
    ]
    return vertices, faces_idx

def animate_6dof_states(results: Dict[str, Any], output_dir: str) -> None:
    time = np.array(results.get("time", []), dtype=float)
    if len(time) == 0:
        return

    chief_data = results.get("chief", {})
    deputies_dict = results.get("deputies", {})
    if not chief_data or not deputies_dict:
        return

    is_6dof = results.get("is_6dof", False)
    r_chief = np.array(chief_data.get("r", []), dtype=float)
    v_chief = np.array(chief_data.get("v", []), dtype=float)

    n_total = len(time)
    step = max(1, n_total // _MAX_FRAMES)
    frame_indices = list(range(0, n_total, step))
    n_frames = len(frame_indices)

    # 1. Pre-compute rotations and bounds
    dcm_eci2ric = get_eci_to_ric_dcm(r_chief, v_chief)
    
    deputy_trajectories_ric = {}
    deputy_full_dcms = {}
    max_ric_dist = 0.0

    for name, sat_data in deputies_dict.items():
        r_dep = np.array(sat_data.get("r", []), dtype=float)
        if len(r_dep) != n_total:
            continue
            
        dr_eci = r_dep - r_chief
        dr_ric = np.einsum('nij,nj->ni', dcm_eci2ric, dr_eci)
        deputy_trajectories_ric[name] = dr_ric
        max_ric_dist = max(max_ric_dist, np.max(np.abs(dr_ric)))

        if is_6dof and "q" in sat_data:
            q_dep = np.array(sat_data["q"], dtype=float)
            dcm_b2eci = get_body_to_eci_dcm(q_dep)
            dcm_b2ric = np.einsum('nij,njk->nik', dcm_eci2ric, dcm_b2eci)
            deputy_full_dcms[name] = dcm_b2ric

    # 2. Setup Figure and Geometries
    fig = plt.figure(figsize=(8, 8))
    ax3d = fig.add_subplot(111, projection="3d")
    palette = plt.cm.tab10.colors

    # Scale visuals based on the max trajectory distance
    cone_length = max_ric_dist * 0.20 if max_ric_dist > 0 else 1.0
    cube_size = max_ric_dist * 0.04 if max_ric_dist > 0 else 0.2

    X_base, Y_base, Z_base = generate_base_cone(30.0, cone_length)
    v_base_cube, faces_idx = generate_base_cube(cube_size)

    # Draw static axis-aligned Chief cube at origin
    chief_faces = [[[v_base_cube[idx][0], v_base_cube[idx][1], v_base_cube[idx][2]] for idx in face] for face in faces_idx]
    ax3d.add_collection3d(Poly3DCollection(chief_faces, facecolors='black', edgecolors='white', alpha=0.9))

    lines_traj = {}
    cone_artists = {}
    cube_artists = {}

    for i, name in enumerate(deputy_trajectories_ric.keys()):
        color = palette[i % len(palette)]
        lines_traj[name], = ax3d.plot([], [], [], color=color, linewidth=1.2, alpha=0.6, label=name)
        cone_artists[name] = None
        cube_artists[name] = None

    bound = max_ric_dist * 1.1 if max_ric_dist > 0 else 10.0
    ax3d.set_xlim(-bound, bound)
    ax3d.set_ylim(-bound, bound)
    ax3d.set_zlim(-bound, bound)
    
    try:
        ax3d.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass

    ax3d.set_xlabel("Radial (km)")
    ax3d.set_ylabel("In-track (km)")
    ax3d.set_zlabel("Cross-track (km)")
    ax3d.set_title("Relative Motion & Attitude in RIC", fontweight="bold")
    ax3d.legend(loc="upper right", fontsize=9)
    time_text = fig.text(0.15, 0.90, "", fontsize=10, fontweight="bold")

    # 3. Animation Update Function
    def _update(frame_num):
        idx = frame_indices[frame_num]
        time_text.set_text(f"Time: {time[idx]:.1f} s")

        for i, name in enumerate(deputy_trajectories_ric.keys()):
            r_ric = deputy_trajectories_ric[name]
            color = palette[i % len(palette)]
            
            # Update trajectory
            lines_traj[name].set_data(r_ric[:idx + 1, 0], r_ric[:idx + 1, 1])
            lines_traj[name].set_3d_properties(r_ric[:idx + 1, 2])
            
            # Remove old dynamic artists
            if cone_artists[name] is not None:
                cone_artists[name].remove()
            if cube_artists[name] is not None:
                cube_artists[name].remove()
            
            r_c = r_ric[idx]

            # Determine rotation (default to identity if no attitude data)
            dcm = deputy_full_dcms[name][idx] if name in deputy_full_dcms else np.eye(3)

            # Draw Deputy Cube
            v_rot = (dcm @ v_base_cube.T).T + r_c
            faces = [[[v_rot[vi][0], v_rot[vi][1], v_rot[vi][2]] for vi in face] for face in faces_idx]
            
            cube_col = Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors='black', alpha=0.8)
            cube_artists[name] = ax3d.add_collection3d(cube_col)

            # Draw FOV Cone
            if name in deputy_full_dcms:
                pts = np.vstack((X_base.flatten(), Y_base.flatten(), Z_base.flatten()))
                pts_rot = dcm @ pts 
                
                X_new = pts_rot[0, :].reshape(X_base.shape) + r_c[0]
                Y_new = pts_rot[1, :].reshape(Y_base.shape) + r_c[1]
                Z_new = pts_rot[2, :].reshape(Z_base.shape) + r_c[2]
                
                cone_artists[name] = ax3d.plot_surface(
                    X_new, Y_new, Z_new, 
                    color=color, alpha=0.2, linewidth=0, shade=False, antialiased=True
                )

        return []

    # 4. Save
    fps = max(5, min(20, n_frames // 10))
    anim = animation.FuncAnimation(fig, _update, frames=n_frames, interval=1000 // fps, blit=False)

    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, "animation_ric_relative_fov.gif")

    writer = animation.PillowWriter(fps=fps)
    anim.save(gif_path, writer=writer)
    plt.close(fig)