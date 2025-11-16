import os
import csv

def save_state_csv(results_serializable, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "state_results.csv")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time", "chief_r_x", "chief_r_y", "chief_r_z",
            "chief_v_x", "chief_v_y", "chief_v_z",
            "deputy_r_x", "deputy_r_y", "deputy_r_z",
            "deputy_v_x", "deputy_v_y", "deputy_v_z",
            "deputy_rho_x", "deputy_rho_y", "deputy_rho_z",
            "deputy_rho_dot_x", "deputy_rho_dot_y", "deputy_rho_dot_z"
        ])

        for t, state_vector in zip(results_serializable["time"], results_serializable["full_state"]):
            writer.writerow([t] + state_vector)

    print(f"State vector saved to: {output_csv}")