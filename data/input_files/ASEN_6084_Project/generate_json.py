import json
import random

def generate_breakup_scenario(num_debris, max_vel, output_filename="breakup_config.json"):
    # 1. Define the static parts of the configuration
    config = {
        "scenario": "Test Configuration for Orbit Propagation",
        "simulation": {
            "epoch": "2000-01-01T12:00:00Z",
            "duration": 172800,
            "time_step": 360,
            "propagator": "2BODY",
            "perturbations": {
                "J2": True,
                "drag": False,
                "SRP": False
            }
        },
        "satellites": [
            {
                "name": "Chief",
                "initial_state": {
                    "state": [42164.14, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "frame": "OEs"
                },
                "mass": 500.0,
                "properties": {
                    "area": 10.0,
                    "Cd": 2.2
                }
            }
        ],
        "output": {
            "save_orbit_elements": False,
            "save_gnc_results": False,
            "plots": False,
            "animate_trajectory": False
        }
    }

    # 2. Generate the randomized debris objects
    for i in range(1, num_debris + 1):
        # Generate random velocities between -max_vel and +max_vel
        vx = random.uniform(-max_vel, max_vel)
        vy = random.uniform(-max_vel, max_vel)
        vz = random.uniform(-max_vel, max_vel)

        debris_obj = {
            "name": f"Debris_{i:02d}_Random",
            "initial_state": {
                "state": [0.0, 0.0, 0.0, vx, vy, vz],
                "frame": "LVLH"
            }
        }
        
        config["satellites"].append(debris_obj)

    # 3. Export to a JSON file
    with open(output_filename, "w") as outfile:
        json.dump(config, outfile, indent=2)
        
    print(f"[+] Successfully generated {num_debris} debris objects.")
    print(f"[+] Saved to {output_filename}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    NUM_DEBRIS = 10                 # Number of debris pieces to generate
    MAX_REL_VELOCITY = 0.000015       # Max velocity in km/s
    # ---------------------
    
    generate_breakup_scenario(
        num_debris=NUM_DEBRIS, 
        max_vel=MAX_REL_VELOCITY, 
        output_filename="data/input_files/ASEN_6084_Project/breakup_config.jsonx"
    )