import os
import csv
import numpy as np
from data.resources.constants import MU_EARTH
from utils.orbital_element_conversions.oe_conversions import inertial_to_oes

def _compute_coes(r_array, v_array):
    """
    Helper function to compute an Nx6 array of Classical Orbital Elements 
    given Nx3 position and velocity arrays.
    """
    n_steps = len(r_array)
    coes = np.zeros((n_steps, 6))
    
    # We keep the loop here assuming the external inertial_to_oes 
    # utility function is not natively vectorized.
    for k in range(n_steps):
        a, e, i, raan, argp, ta = inertial_to_oes(
            r_array[k], v_array[k], MU_EARTH, 'deg'
        )
        coes[k] = [a, e, i, raan, argp, ta]
        
    return coes

def save_orbital_elements(results_serializable, output_dir):
    """
    Computes Classical Orbital Elements (COEs) for the Chief and all Deputies.
    Saves individual CSV files per spacecraft and returns a nested dictionary
    of COE data for plotting.
    """
    time = np.array(results_serializable.get("time", []), dtype=float)
    if len(time) == 0:
        print("No time data. Skipping orbital_elements.csv.")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    coes_dict = {"chief": None, "deputies": {}}

    # =========================================================
    # 1. Process Chief
    # =========================================================
    chief_r = np.array(results_serializable.get("chief", {}).get("r", []), dtype=float)
    chief_v = np.array(results_serializable.get("chief", {}).get("v", []), dtype=float)
    
    if len(chief_r) == len(time) and len(chief_v) == len(time):
        chief_coes = _compute_coes(chief_r, chief_v)
        coes_dict["chief"] = chief_coes
        
        # Save Chief CSV
        out_csv = os.path.join(output_dir, "orbital_elements_chief.csv")
        header = ["time_s", "a_km", "e", "i_deg", "RAAN_deg", "ARGP_deg", "TA_deg"]
        coe_data = np.column_stack([time, chief_coes])
        
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(coe_data)
    else:
        print("Missing or invalid Chief state data. Cannot compute Chief COEs.")
        return coes_dict # Cannot compute relative elements without the Chief

    # =========================================================
    # 2. Process Deputies
    # =========================================================
    deputies = results_serializable.get("deputies", {})
    
    for sat_name, sat_data in deputies.items():
        dep_r = np.array(sat_data.get("r", []), dtype=float)
        dep_v = np.array(sat_data.get("v", []), dtype=float)
        
        if len(dep_r) != len(time) or len(dep_v) != len(time):
            print(f"[{sat_name}] Missing or invalid state data. Skipping.")
            continue
            
        # Absolute COEs
        dep_coes = _compute_coes(dep_r, dep_v)
        
        # Differential COEs (wrapping angular differences cleanly)
        delta_coes = dep_coes - chief_coes
        
        # Save to Dictionary
        coes_dict["deputies"][sat_name] = {
            "coes": dep_coes,
            "delta_coes": delta_coes
        }
        
        # Save Deputy CSV
        safe_name = sat_name.replace(" ", "_").lower()
        out_csv = os.path.join(output_dir, f"orbital_elements_{safe_name}.csv")
        
        header = [
            "time_s", 
            "a_km", "e", "i_deg", "RAAN_deg", "ARGP_deg", "TA_deg",
            "delta_a_km", "delta_e", "delta_i_deg", "delta_RAAN_deg", 
            "delta_ARGP_deg", "delta_TA_deg"
        ]
        
        dep_coe_data = np.column_stack([time, dep_coes, delta_coes])
        
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(dep_coe_data)
            
    return coes_dict