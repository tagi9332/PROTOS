import os
import matplotlib.pyplot as plt

def plot_orbital_elements(time, coes_dict, output_dir):
    """
    Plots the Classical Orbital Elements (COEs) and differential COEs 
    for the Chief and all N-Deputies.
    """
    if not coes_dict:
        print("No COE data provided. Skipping plots.")
        return

    labels = ["a (km)", "e", r"i ($^\circ$)", r"RAAN ($^\circ$)", r"ARGP ($^\circ$)", r"TA ($^\circ$)"]

    def plot_set(data, title, filename):
        if data is None or len(data) == 0:
            return
            
        fig, axes = plt.subplots(6, 1, figsize=(10, 14), sharex=True)
        for i in range(6):
            axes[i].plot(time, data[:, i], linewidth=1.5, color='b')
            axes[i].set_ylabel(labels[i])
            axes[i].grid(True)
            axes[i].set_xlim([time[0], time[-1]])
            
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(title, fontsize=16)
        
        # Prevent the suptitle from overlapping the top subplot
        fig.tight_layout(rect=[0, 0.03, 1, 0.97]) 
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close(fig)

    # 1. Plot Chief COEs
    chief_coes = coes_dict.get("chief")
    if chief_coes is not None:
        plot_set(chief_coes, "Chief Orbital Elements", "coes_chief.png")

    # 2. Plot Deputies COEs & Differential COEs
    deputies = coes_dict.get("deputies", {})
    for sat_name, sat_data in deputies.items():
        safe_name = sat_name.replace(" ", "_").lower()
        
        # Absolute COEs
        dep_coes = sat_data.get("coes")
        plot_set(dep_coes, f"{sat_name} Orbital Elements", f"coes_{safe_name}.png")
        
        # Differential COEs (relative to Chief)
        delta_coes = sat_data.get("delta_coes")
        plot_set(delta_coes, f"{sat_name} Differential COEs", f"diff_coes_{safe_name}.png")