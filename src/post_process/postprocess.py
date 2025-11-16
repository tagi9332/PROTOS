import os
import numpy as np
from src.post_process.save_full_state import save_state_csv
from src.post_process.save_element_vectors import save_orbital_elements
from src.post_process.plot_eci_traj import plot_ECI_trajectories
from src.post_process.plot_3d_ric_traj import plot_3d_RIC_trajectory
from src.post_process.save_control_vector import save_control_accel
from src.post_process.plot_dv import plot_delta_v
from src.post_process.plot_oes import plot_orbital_elements
from src.post_process.plot_control_effort import plot_control_accel

def _convert_ndarray(obj):
    if isinstance(obj, dict):
        return {k: _convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_ndarray(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def postprocess(results, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    results_serializable = _convert_ndarray(results)

    # Save raw state vectors
    save_state_csv(results_serializable, output_dir)

    # Save classical & differential COEs
    coes = save_orbital_elements(results_serializable, output_dir)

    # Trajectory plots
    plot_ECI_trajectories(results_serializable, output_dir)

    # Control profiles
    save_control_accel(results_serializable, output_dir)
    plot_delta_v(results_serializable, output_dir)
    plot_control_accel(results_serializable, output_dir) # type: ignore



    # COE plots
    if coes is not None:
        chief, deputy, delta = coes
        time = np.array(results_serializable["time"], dtype=float) # type: ignore
        plot_orbital_elements(time, chief, deputy, delta, output_dir)

    print(f"Postprocess completed. Output saved in {output_dir}")

    # Open RIC-frame plot in interactive window
    plot_3d_RIC_trajectory(results_serializable, output_dir, show_plot=True)
