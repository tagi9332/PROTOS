import os
import numpy as np
import matplotlib.pyplot as plt


def _extract_quat_and_rates(full_state, is_6dof):
    """
    From full_state row, extract quaternion and body rates for chief & deputy.
    full_state format (6DOF):
        [ rC(3), vC(3), qC(4), wC(3),
          rD(3), vD(3), qD(4), wD(3) ]
    """
    if not is_6dof:
        return None

    fs = np.array(full_state, dtype=float)

    qC = fs[18:22]
    wC = fs[22:25]

    qD = fs[25:29]
    wD = fs[29:32]

    return qC, wC, qD, wD


def plot_attitude(results_serializable, output_dir):
    """
    Plot quaternions and angular rates for Chief & Deputy.
    Each plot has two stacked subplots: Chief on top, Deputy on bottom.
    Saves:
      attitude_quaternions.png
      attitude_rates.png
    """
    if not results_serializable.get("is_6dof", False):
        return

    time = np.array(results_serializable["time"], dtype=float)
    full_state = results_serializable["full_state"]

    qC_list, qD_list = [], []
    wC_list, wD_list = [], []

    for fs in full_state:
        qC, wC, qD, wD = _extract_quat_and_rates(fs, True) # type: ignore
        qC_list.append(qC)
        qD_list.append(qD)
        wC_list.append(wC)
        wD_list.append(wD)

    qC = np.array(qC_list)
    qD = np.array(qD_list)
    wC = np.array(wC_list)
    wD = np.array(wD_list)

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # 1) Quaternion Plot (Chief & Deputy stacked)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    q_labels = [r"$q_0$", r"$q_1$", r"$q_2$", r"$q_3$"]

    # Chief
    for i in range(4):
        axes[0].plot(time, qC[:, i], label=q_labels[i])
    axes[0].set_ylabel("Chief Quaternions")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    # Deputy
    for i in range(4):
        axes[1].plot(time, qD[:, i], label=q_labels[i])
    axes[1].set_ylabel("Deputy Quaternions")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    fig.suptitle("Quaternion Components for Chief and Deputy")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "attitude_quaternions.png"), dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------
    # 2) Angular Rates Plot (Chief & Deputy stacked)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    rate_labels = [r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]

    # Chief
    for i in range(3):
        axes[0].plot(time, wC[:, i], label=rate_labels[i])
    axes[0].set_ylabel("Chief Angular Rates (rad/s)")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    # Deputy
    for i in range(3):
        axes[1].plot(time, wD[:, i], label=rate_labels[i])
    axes[1].set_ylabel("Deputy Angular Rates (rad/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    fig.suptitle("Body Angular Rates for Chief and Deputy")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "attitude_rates.png"), dpi=200)

