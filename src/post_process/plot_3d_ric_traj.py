import os
import numpy as np
import plotly.graph_objects as go

def plot_3d_RIC_trajectory(results_serializable, output_dir, show_plot=True, filename="hill_frame_trajectory.png"):
    """
    Plots the deputy's relative trajectory in the chief's Hill (RIC) frame.
    Saves a PNG and optionally displays an interactive 3D plot.

    Args:
        results_serializable (dict): Simulation results containing 'full_state'.
        output_dir (str): Directory to save the figure.
        show_plot (bool): Whether to display an interactive plot.
        filename (str): Name of the saved static figure (PNG).
    """
    # Get state data
    states = np.array(results_serializable.get("full_state", []), dtype=float)
    if len(states) == 0:
        print("No Hill-frame data available. Skipping plot.")
        return

    rel_H = states[:, 12:15]        # Deputy relative position (rho)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------
    # Interactive 3D plot with Plotly
    # ----------------------------
    fig = go.Figure()

    # Trajectory line
    fig.add_trace(go.Scatter3d(
        x=rel_H[:, 0],
        y=rel_H[:, 1],
        z=rel_H[:, 2],
        mode='lines',
        line=dict(color='blue', width=4),
        name='Deputy Trajectory'
    ))

    # Start marker
    fig.add_trace(go.Scatter3d(
        x=[rel_H[0, 0]],
        y=[rel_H[0, 1]],
        z=[rel_H[0, 2]],
        mode='markers',
        marker=dict(color='green', size=6),
        name='Start'
    ))

    # End marker
    fig.add_trace(go.Scatter3d(
        x=[rel_H[-1, 0]],
        y=[rel_H[-1, 1]],
        z=[rel_H[-1, 2]],
        mode='markers',
        marker=dict(color='black', size=6),
        name='End'
    ))

    # Chief origin
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(color='red', size=6),
        name='Chief (origin)'
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
