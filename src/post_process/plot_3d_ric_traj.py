import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def plot_3d_RIC_trajectory(results_serializable, output_dir, show_plot=True):
    """
    Plots the deputies' relative trajectories in the chief's Hill (RIC) frame.
    Saves a PNG and optionally displays an interactive 3D plot.

    Args:
        results_serializable (dict): Simulation results containing 'deputies'.
        output_dir (str): Directory to save the figure.
        show_plot (bool): Whether to display an interactive plot.
    """
    deputies = results_serializable.get("deputies", {})
    if not deputies:
        print("No deputy data available. Skipping plot.")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------
    # Interactive 3D plot with Plotly
    # ----------------------------
    fig = go.Figure()

    # Plot Chief origin (anchor point)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(color='red', size=6, symbol='cross'),
        name='Chief (origin)'
    ))

    # Get a nice color palette for N deputies
    colors = px.colors.qualitative.Plotly 

    # Loop through each deputy and plot its trajectory
    for i, (sat_name, sat_data) in enumerate(deputies.items()):
        rel_H = np.array(sat_data.get("rho", []), dtype=float)
        
        if len(rel_H) == 0:
            continue
            
        color = colors[i % len(colors)] # Cycle colors if N > 10

        # Trajectory line
        fig.add_trace(go.Scatter3d(
            x=rel_H[:, 0],
            y=rel_H[:, 1],
            z=rel_H[:, 2],
            mode='lines',
            line=dict(color=color, width=4),
            name=f'{sat_name} Trajectory',
            legendgroup=sat_name # Groups line, start, and end markers in legend
        ))

        # Start marker
        fig.add_trace(go.Scatter3d(
            x=[rel_H[0, 0]],
            y=[rel_H[0, 1]],
            z=[rel_H[0, 2]],
            mode='markers',
            marker=dict(color='green', size=5, symbol='circle'),
            name=f'{sat_name} Start',
            legendgroup=sat_name,
            showlegend=False # Hide to keep the legend clean
        ))

        # End marker
        fig.add_trace(go.Scatter3d(
            x=[rel_H[-1, 0]],
            y=[rel_H[-1, 1]],
            z=[rel_H[-1, 2]],
            mode='markers',
            marker=dict(color='black', size=5, symbol='square'),
            name=f'{sat_name} End',
            legendgroup=sat_name,
            showlegend=False # Hide to keep the legend clean
        ))

    fig.update_layout(
        title="Deputy Relative Motion in Chief's Hill Frame",
        scene=dict(
            xaxis_title='Radial (x) [km]',
            yaxis_title='Along-track (y) [km]',
            zaxis_title='Cross-track (z) [km]',
            aspectmode='cube'  # equal scaling
        ),
        width=900,
        height=700
    )

    # Show interactive plot
    if show_plot:
        fig.show()