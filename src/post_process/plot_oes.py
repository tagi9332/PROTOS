import os
import matplotlib.pyplot as plt

def plot_orbital_elements(time, chief_coes, deputy_coes, delta_coes, output_dir):

    labels = ["a (km)", "e", "i (deg)", "RAAN (deg)", "ARGP (deg)", "TA (deg)"]

    def plot_set(data, title, filename):
        fig, axes = plt.subplots(6, 1, figsize=(10, 14), sharex=True)
        for i in range(6):
            axes[i].plot(time, data[:, i])
            axes[i].set_ylabel(labels[i])
            axes[i].grid(True)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(title)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    plot_set(chief_coes,  "Chief COEs",  "chief_orbital_elements.png")
    plot_set(deputy_coes, "Deputy COEs", "deputy_orbital_elements.png")
    plot_set(delta_coes,  "Differential COEs", "diff_orbital_elements.png")