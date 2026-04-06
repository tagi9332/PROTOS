import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_2d_ground_tracks(results, output_dir, map_path=os.path.join("data", "resources", "8k_earth_daymap.jpg")):
    """
    Plots the 2D ground tracks of the chief and deputies over an Earth map.
    Converts ECI coordinates to LLA, accounting for Earth's rotation.
    """
    time = results.get("time", [])
    if len(time) == 0:
        print("No time data available. Skipping ground track plot.")
        return

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    # Load and display the Earth map background
    try:
        img = mpimg.imread(map_path)
        # Extent binds the image strictly to standard lat/lon boundaries
        ax.imshow(img, extent=[-180, 180, -90, 90])
    except FileNotFoundError:
        print(f"Warning: Earth map not found at {map_path}. Plotting without background.")
        ax.set_facecolor('#e0e0e0') 

    # Helper function: ECI to Lat/Lon conversion
    def get_lat_lon(r_eci, t):
        x, y, z = r_eci[:, 0], r_eci[:, 1], r_eci[:, 2]
        r_mag = np.linalg.norm(r_eci, axis=1)
        
        # Latitude
        lat = np.degrees(np.arcsin(z / r_mag))
        
        # ECI Longitude
        lon_eci = np.degrees(np.arctan2(y, x))
        
        # Earth rotation rate ~7.2921159e-5 rad/s -> ~0.004178 deg/s
        omega_e_deg = np.degrees(7.2921159e-5) 
        
        # Convert ECI longitude to ECEF (Earth-fixed) longitude
        lon_ecef = (lon_eci - omega_e_deg * t + 180) % 360 - 180
        
        # Insert NaNs where the track wraps around the anti-meridian to prevent horizontal streaking
        diffs = np.abs(np.diff(lon_ecef))
        wrap_idx = np.where(diffs > 180)[0] + 1
        
        lon_plot = np.insert(lon_ecef, wrap_idx, np.nan)
        lat_plot = np.insert(lat, wrap_idx, np.nan)
        
        return lon_plot, lat_plot

    # Plot Chief Trajectory
    chief_r = results.get("chief", {}).get("r", [])
    if len(chief_r) > 0:
        lon, lat = get_lat_lon(np.array(chief_r), time)
        ax.plot(lon, lat, label="Chief", color='red', linewidth=2, zorder=3)
        ax.scatter(lon[0], lat[0], color='green', marker='o', s=40, label="Start", zorder=4)
        ax.scatter(lon[-1], lat[-1], color='black', marker='X', s=40, label="End", zorder=4)

    # Plot Deputy Trajectories
    deputies = results.get("deputies", {})
    colors = plt.cm.tab10.colors 
    for i, (sat_name, sat_data) in enumerate(deputies.items()):
        dep_r = sat_data.get("r", [])
        if len(dep_r) > 0:
            c = colors[i % len(colors)]
            lon, lat = get_lat_lon(np.array(dep_r), time)
            ax.plot(lon, lat, label=f"{sat_name.capitalize()}", color=c, linewidth=1.5, zorder=2)
            # Add small start/end markers for deputies
            ax.scatter(lon[0], lat[0], color='green', marker='o', s=20, zorder=4)
            ax.scatter(lon[-1], lat[-1], color='black', marker='X', s=20, zorder=4)

    # Format Plot Constraints & Aesthetics
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.set_title('2D Ground Tracks')
    
    # Place legend outside the plot to avoid obscuring tracks
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    # Save Output
    plot_path = os.path.join(output_dir, '2D_ground_tracks.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()