import os
import csv

def save_state_csv(results_serializable, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "state_results.csv")

    sample_state = results_serializable["full_state"][0]
    state_length = len(sample_state)

    # Determine if attitude states are included (3DOF = 18 states, 6DOF = 32)
    has_attitude = (state_length > 18)

    # --------------------------------------
    # Build CSV header dynamically
    # --------------------------------------
    header = [
        "time",
        "chief_r_x", "chief_r_y", "chief_r_z",
        "chief_v_x", "chief_v_y", "chief_v_z",
        "deputy_r_x", "deputy_r_y", "deputy_r_z",
        "deputy_v_x", "deputy_v_y", "deputy_v_z",
        "deputy_rho_x", "deputy_rho_y", "deputy_rho_z",
        "deputy_rho_dot_x", "deputy_rho_dot_y", "deputy_rho_dot_z",
    ]

    # If 6DOF, append attitude columns
    if has_attitude:
        header += [
            # Chief attitude
            "chief_q_w", "chief_q_x", "chief_q_y", "chief_q_z",
            "chief_omega_x", "chief_omega_y", "chief_omega_z",

            # Deputy attitude
            "deputy_q_w", "deputy_q_x", "deputy_q_y", "deputy_q_z",
            "deputy_omega_x", "deputy_omega_y", "deputy_omega_z"
        ]

    # --------------------------------------
    # Write the CSV
    # --------------------------------------
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for t, state_vector in zip(results_serializable["time"],
                                   results_serializable["full_state"]):
            writer.writerow([t] + state_vector)

    print(f"State vector saved to: {output_csv}")
