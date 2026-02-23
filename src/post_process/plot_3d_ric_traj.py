import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import webbrowser

def plot_3d_RIC_trajectory(results_serializable, output_dir, show_plot=True):
    """
    Plots the deputies' relative trajectories in the chief's Hill (RIC) frame.
    Saves an interactive HTML 3D plot and optionally displays it.
    Forces a 1:1:1 physical aspect ratio by creating a cubic bounding box.
    """
    deputies = results_serializable.get("deputies", {})
    if not deputies:
        print("No deputy data available. Skipping plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    fig = go.Figure()

    # Plot Chief origin (anchor point)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(color='red', size=6),
        name='Chief (origin)'
    ))

    colors = px.colors.qualitative.Plotly 

    # --- Variables for tracking the bounding box ---
    x_min, x_max = 0.0, 0.0
    y_min, y_max = 0.0, 0.0
    z_min, z_max = 0.0, 0.0

    # Loop through each deputy and plot its trajectory
    for i, (sat_name, sat_data) in enumerate(deputies.items()):
        rel_H = np.array(sat_data.get("rho", []), dtype=float)
        
        if len(rel_H) == 0:
            continue
            
        color = colors[i % len(colors)]

        # Trajectory line
        fig.add_trace(go.Scatter3d(
            x=rel_H[:, 0], y=rel_H[:, 1], z=rel_H[:, 2],
            mode='lines',
            line=dict(color=color, width=4),
            name=f'{sat_name.capitalize()} Trajectory',
            legendgroup=sat_name 
        ))

        # Start marker (Now shown in legend)
        fig.add_trace(go.Scatter3d(
            x=[rel_H[0, 0]], y=[rel_H[0, 1]], z=[rel_H[0, 2]],
            mode='markers',
            marker=dict(color='green', size=5, symbol='circle'),
            name=f'{sat_name.capitalize()} Start',
            legendgroup=sat_name,
            showlegend=True
        ))

        # End marker (Now shown in legend)
        fig.add_trace(go.Scatter3d(
            x=[rel_H[-1, 0]], y=[rel_H[-1, 1]], z=[rel_H[-1, 2]],
            mode='markers',
            marker=dict(color='black', size=5),
            name=f'{sat_name.capitalize()} End',
            legendgroup=sat_name,
            showlegend=True
        ))

        # Update min/max for bounding box
        x_min, x_max = min(x_min, rel_H[:, 0].min()), max(x_max, rel_H[:, 0].max())
        y_min, y_max = min(y_min, rel_H[:, 1].min()), max(y_max, rel_H[:, 1].max())
        z_min, z_max = min(z_min, rel_H[:, 2].min()), max(z_max, rel_H[:, 2].max())

    # --- Calculate Cubic Bounding Box Limits ---
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    
    x_center = (x_max + x_min) / 2.0
    y_center = (y_max + y_min) / 2.0
    z_center = (z_max + z_min) / 2.0

    half_range = (max_range / 2.0) * 1.05 
    
    fig.update_layout(
        title="Deputy Relative Motion in Chief's Hill Frame",
        scene=dict(
            xaxis_title='Radial (x) [km]',
            yaxis_title='Along-track (y) [km]',
            zaxis_title='Cross-track (z) [km]',
            
            # Apply our calculated uniform limits
            xaxis=dict(range=[x_center - half_range, x_center + half_range]),
            yaxis=dict(range=[y_center - half_range, y_center + half_range]),
            zaxis=dict(range=[z_center - half_range, z_center + half_range]),
            
            aspectmode='cube' 
        ),
        width=1000,
        height=800
    )

    # Save as HTML and automatically open
    plot_path = os.path.join(output_dir, 'interactive_RIC_trajectory.html')
    fig.write_html(plot_path)
    
    if show_plot:
        webbrowser.open('file://' + os.path.realpath(plot_path))