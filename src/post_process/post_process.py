import os
import numpy as np
from src.post_process import (
    plot_relative_separation,
    save_state_csv,
    save_orbital_elements,
    plot_ECI_trajectories,
    plot_3d_RIC_trajectory,
    save_plane_views,
    save_iso_view,
    save_control_accel,
    plot_delta_v,
    plot_orbital_elements,
    plot_attitude_control,
    plot_control_accel,
    plot_attitude
)

def _convert_ndarray(obj):
    if isinstance(obj, dict):
        return {k: _convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_ndarray(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def post_process(results, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    results_serializable = _convert_ndarray(results)
    assert (isinstance(results_serializable, dict))

    # Detect 6DOF by length of full_state vector
    init_state = results_serializable["full_state"][0]
    results_serializable["is_6dof"] = len(init_state) >= 32 # 6DOF has at least 32 elements

    # Save raw state vectors
    save_state_csv(results_serializable, output_dir)

    # Save classical & differential COEs
    coes = save_orbital_elements(results_serializable, output_dir)

    # Trajectory plots
    plot_ECI_trajectories(results_serializable, output_dir)

    # Attitude plots
    if results_serializable["is_6dof"]:
        plot_attitude(results_serializable, output_dir)
        plot_attitude_control(results_serializable, output_dir)
        
    # Control profiles
    save_control_accel(results_serializable, output_dir)
    plot_delta_v(results_serializable, output_dir)
    plot_control_accel(results_serializable, output_dir) # type: ignore


    # COE plots
    if coes is not None:
        chief, deputy, delta = coes
        time = np.array(results_serializable["time"], dtype=float) # type: ignore
        plot_orbital_elements(time, chief, deputy, delta, output_dir)

    # Save static plane views of RIC-frame trajectory
    save_plane_views(results_serializable, output_dir)
    save_iso_view(results_serializable, output_dir)

    # Plot relative separation
    plot_relative_separation(results_serializable, output_dir)

    # Open RIC-frame plot in interactive window
    plot_3d_RIC_trajectory(results_serializable, output_dir, show_plot=True)

    print(f"Postprocess completed. Output saved in {output_dir}")
