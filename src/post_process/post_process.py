import os
import numpy as np
from datetime import datetime

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
    plot_attitude,
    animate_6dof_states,
    setup_output_dir
)

def post_process(results, output_config):
    """
    Master post-processing wrapper. 
    Expects 'results' as a dictionary containing 'time', 'chief', and 'deputies'.
    """
    print("Postprocessing results...")

    # Initialize directories and get paths
    main_output_dir, vehicle_dirs = setup_output_dir.setup_output_directories(results)

    is_6dof = results.get("is_6dof", False)
    time_data = np.array(results.get("time", []), dtype=float)

    # Initialize post_process input parameters
    save_gnc_results = output_config.get("save_gnc_results", True)
    animate_trajectories = output_config.get("animate_trajectories", True)
    save_orbit_elements = output_config.get("save_orbit_elements", True)
    plots = output_config.get("plots", True)
    
    # ==========================================
    # A. Combined Results
    # ==========================================
    if plots:
        plot_ECI_trajectories(results, main_output_dir)
        save_iso_view(results, main_output_dir)
        
    if animate_trajectories:
        animate_6dof_states(results, main_output_dir)

    # ==========================================
    # B. Vehicle-Specific Results 
    # ==========================================
    save_state_csv(results, vehicle_dirs)
    if plots:
        save_plane_views(results, vehicle_dirs)
        plot_relative_separation(results, vehicle_dirs)
    if save_gnc_results: save_control_accel(results, vehicle_dirs)
    if save_orbit_elements:
        coes_dict = save_orbital_elements(results, vehicle_dirs)
        if plots:
            plot_orbital_elements(time_data, coes_dict, vehicle_dirs)
    if is_6dof:
        if plots:
            plot_attitude(results, vehicle_dirs)
            if save_gnc_results:
                plot_attitude_control(results, vehicle_dirs)
        
    if plots and save_gnc_results:
        plot_delta_v(results, vehicle_dirs)
        plot_control_accel(results, vehicle_dirs)       

    # ==========================================
    # C. Interactive Plots
    # ==========================================
    if plots:
        plot_3d_RIC_trajectory(results, main_output_dir, show_plot=True)

    print(f"Postprocess completed. Outputs saved in {main_output_dir}")