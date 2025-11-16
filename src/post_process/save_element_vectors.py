import os
import csv
import numpy as np
from data.resources.constants import MU_EARTH
from utils.orbital_element_conversions.oe_conversions import inertial_to_orbital_elements

def save_orbital_elements(results_serializable, output_dir):
    states = np.array(results_serializable.get("full_state", []), dtype=float)
    if len(states) == 0:
        print("No COE data. Skipping orbital_elements.csv.")
        return

    time = np.array(results_serializable["time"], dtype=float)
    chief_r = states[:, 0:3]
    chief_v = states[:, 3:6]
    deputy_r = states[:, 6:9]
    deputy_v = states[:, 9:12]

    chief_coes = np.zeros((len(time), 6))
    deputy_coes = np.zeros((len(time), 6))
    delta_coes  = np.zeros((len(time), 6))

    for k in range(len(time)):
        a_c, e_c, i_c, raan_c, argp_c, TA_c = inertial_to_orbital_elements(
            chief_r[k], chief_v[k], MU_EARTH, 'deg'
        )
        a_d, e_d, i_d, raan_d, argp_d, TA_d = inertial_to_orbital_elements(
            deputy_r[k], deputy_v[k], MU_EARTH, 'deg'
        )

        chief_coes[k]  = [a_c, e_c, i_c, raan_c, argp_c, TA_c]
        deputy_coes[k] = [a_d, e_d, i_d, raan_d, argp_d, TA_d]
        delta_coes[k]  = chief_coes[k] - deputy_coes[k]

    # Save to CSV
    header = [
        "time [s]", 
        "a_c [km]", "e_c [-]", "i_c [deg]", "RAAN_c [deg]", "ARGP_c [deg]", "TA_c [deg]",
        "a_d [km]", "e_d [-]", "i_d [deg]", "RAAN_d [deg]", "ARGP_d [deg]", "TA_d [deg]",
        "delta_a [km]", "delta_e [-]", "delta_i [deg]", "delta_RAAN [deg]",
        "delta_ARGP [deg]", "delta_TA [deg]"
    ]

    coe_data = np.column_stack([time, chief_coes, deputy_coes, delta_coes])

    out_csv = os.path.join(output_dir, "orbital_elements.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(coe_data)

    print(f"Orbital element sets saved to: {out_csv}")

    return chief_coes, deputy_coes, delta_coes