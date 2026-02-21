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

def post_process(results, output_dir):
    """
    Master post-processing wrapper. 
    Expects 'results' as a dictionary containing 'time', 'chief', and 'deputies'.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Detect 6DOF dynamically
    is_6dof = "q" in results.get("chief", {})
    results["is_6dof"] = is_6dof

    print(f"Postprocessing results")

    # 2. Extract and Save CSV Data
    save_state_csv(results, output_dir)
    save_control_accel(results, output_dir)
    
    # save_orbital_elements now dynamically returns the coes_dict we need!
    coes_dict = save_orbital_elements(results, output_dir)

    # 3. Generate Trajectory Plots
    plot_ECI_trajectories(results, output_dir)
    save_plane_views(results, output_dir)
    save_iso_view(results, output_dir)
    plot_relative_separation(results, output_dir)

    # 4. Generate Attitude & Control Plots
    if is_6dof:
        plot_attitude(results, output_dir)
        plot_attitude_control(results, output_dir)
        
    plot_delta_v(results, output_dir)
    plot_control_accel(results, output_dir)

    # 5. Generate COE Plots
    if coes_dict:  # Safely checks for non-empty dictionary
        time = np.array(results.get("time", []), dtype=float)
        plot_orbital_elements(time, coes_dict, output_dir)

    # 6. Interactive Plot (Usually best kept last so it doesn't block CSV generation)
    plot_3d_RIC_trajectory(results, output_dir, show_plot=True)

    print(f"--- Postprocess completed. Output saved in {output_dir} ---")