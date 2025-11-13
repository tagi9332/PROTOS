import commentjson as json
import dateutil
import numpy as np
from utils.frame_convertions.rel_to_inertial_functions import rel_vector_to_inertial, LVLH_DCM, compute_omega, inertial_to_rel_LVLH
from utils.orbital_element_conversions.oe_conversions import inertial_to_orbital_elements, orbital_elements_to_inertial, lroes_to_inertial

def parse_input(file_path: str) -> dict:
    """
    Parse the JSONX input file and prepare input dictionaries
    for dynamics, GNC, and postprocessing.
    """
    with open(file_path, "r") as f:
        raw_config = json.load(f) 
    
    # Extract simulation parameters
    sim_config = raw_config.get("simulation", {})

    # Parse epoch if present
    epoch_str = sim_config.get("epoch", None)
    if epoch_str is not None:
        sim_config["epoch"] = dateutil.parser.isoparse(epoch_str)
    else:
        # Default to J2000: 2000-01-01 12:00:00 UTC
        sim_config["epoch"] = dateutil.parser.isoparse("2000-01-01T12:00:00Z")

    # Extract output parameters
    output_config = raw_config.get("output", {})

    # Extract propagator selection (default to 2BODY)
    propagator = sim_config.get("propagator", "2BODY").upper()
    sim_config["propagator"] = propagator  # store in sim_config for downstream use

    # Process satellites
    satellites = raw_config.get("satellites", [])

    # Separate chief and deputy
    chief = next(sat for sat in satellites if sat["name"].lower() == "chief")
    deputy = next(sat for sat in satellites if sat["name"].lower() == "deputy")

    # Chief initialization
    chief_initial = chief["initial_state"]
    frame = chief_initial.get("frame", "").upper()
    chief_vector = np.array(chief_initial["state"])

    if frame == "ECI":
        # Chief given directly in ECI
        chief_r = chief_vector[:3]
        chief_v = chief_vector[3:]

    elif frame == "OES":
        # Chief given in orbital elements vector [a, e, i, RAAN, AOP, TA]
        a, e, i, RAAN, AOP, TA = chief_vector
        chief_r, chief_v = orbital_elements_to_inertial(a, e, i, RAAN, AOP, TA, mu=398600.4418, units='deg')

    else:
            raise ValueError("Chief initial state must be either ECI or ORBITAL_ELEMENTS")

    # Deputy state
    deputy_state = np.array(deputy["initial_state"]["state"])
    frame = deputy["initial_state"].get("frame", "").upper()

    if frame == "ECI":
        # Deputy is given in ECI frame
        deputy_r = deputy_state[:3]
        deputy_v = deputy_state[3:]
        # Convert to LVLH relative position and velocity
        C_HN = LVLH_DCM(chief_r, chief_v) 
        deputy_rho = C_HN @ (deputy_r - chief_r)
        omega = compute_omega(chief_r, chief_v)
        deputy_rho_dot = C_HN @ (deputy_v - chief_v) - np.cross(omega, deputy_rho)

    elif frame == "LVLH":
        # Deputy is given relative to chief in LVLH
        deputy_rho = deputy_state[:3]
        deputy_rho_dot = deputy_state[3:]
        # Convert to inertial frame
        deputy_r, deputy_v = rel_vector_to_inertial(deputy_rho, deputy_rho_dot, chief_r, chief_v)

    elif frame == "LROES":
        # Deputy is given in Linearized Relative Orbital Elements (LROEs) relative to chief
        lroes = deputy_state  # [A_0, B_0, alpha, beta, x_offset, y_offset]
        deputy_r, deputy_v = lroes_to_inertial(0, chief_r, chief_v, lroes)  # time t=0
        # Convert to LVLH relative position and velocity
        C_HN = LVLH_DCM(chief_r, chief_v) 
        deputy_rho = C_HN @ (deputy_r - chief_r)
        omega = compute_omega(chief_r, chief_v)
        deputy_rho_dot = C_HN @ (deputy_v - chief_v) - np.cross(omega, deputy_rho)

    elif frame == "DOES":
        # Deputy is given in delta Orbital Elements (dOEs) relative to chief
        dOEs = deputy_state  # [d_a, d_e, d_i, d_RAAN, d_AOP, d_TA]

        # Compute chief's orbital elements
        # If chief given in OEs, use those directly
        if chief_initial.get("frame", "").upper() == "OES":
            a_chief, e_chief, i_chief, RAAN_chief, AOP_chief, TA_chief = chief_vector
        else:
            a_chief, e_chief, i_chief, RAAN_chief, AOP_chief, TA_chief = inertial_to_orbital_elements(chief_r, chief_v, units='deg')

        # Apply delta OEs
        a_dep = a_chief + dOEs[0]
        e_dep = e_chief + dOEs[1]
        i_dep = i_chief + dOEs[2]
        RAAN_dep = RAAN_chief + dOEs[3]
        AOP_dep = AOP_chief + dOEs[4]
        TA_dep = TA_chief + dOEs[5]

        # Convert deputy OEs to inertial
        deputy_r, deputy_v = orbital_elements_to_inertial(a_dep, e_dep, i_dep, RAAN_dep, AOP_dep, TA_dep, units='deg')

        # Convert to LVLH relative position and velocity
        deputy_rho, deputy_rho_dot = inertial_to_rel_LVLH(deputy_r, deputy_v, chief_r, chief_v)

    elif frame == "RIC":
        # TODO: implement RIC to inertial conversion
        raise ValueError("RIC frame conversion not implemented yet")

    else:
        raise ValueError("Deputy initial state frame not properly defined")
    
    # Dynamics input: inertial positions/velocities + simulation config
    dynamics_input = {
        "chief_r": chief_r,
        "chief_v": chief_v,
        "deputy_r": deputy_r,
        "deputy_v": deputy_v,
        "deputy_rho": deputy_rho,
        "deputy_rho_dot": deputy_rho_dot,
        "satellite_properties": {
            "chief": chief.get("properties", {}),
            "deputy": deputy.get("properties", {})
        },
        "simulation": sim_config  # includes propagator
    }

    # --- GNC Section Handling ---
    gnc_section = raw_config.get("gnc", {})

    # Default desired relative state (zeros)
    deputy_rho_des = np.zeros(3)
    deputy_rho_dot_des = np.zeros(3)

    if gnc_section:
        guidance = gnc_section.get("guidance", {})
        navigation = gnc_section.get("navigation", {})
        control = gnc_section.get("control", {})

        # Parse desired state if guidance is RPO-type
        if guidance.get("type", "").upper() == "RPO":
            rpo = guidance.get("rpo", {})
            desired_state = np.array(rpo.get("deputy_desired_relative_state", [0,0,0,0,0,0]))
            frame = rpo.get("frame", "DOES").upper()

            if frame == "LVLH":
                deputy_rho_des = desired_state[:3]
                deputy_rho_dot_des = desired_state[3:]
            elif frame == "DOES":
                # Compute desired relative from delta OEs
                if chief_initial.get("frame", "").upper() == "OES":
                    a_c, e_c, i_c, RAAN_c, AOP_c, TA_c = chief_vector
                else:
                    a_c, e_c, i_c, RAAN_c, AOP_c, TA_c = inertial_to_orbital_elements(chief_r, chief_v, units='deg')

                dOEs = desired_state
                a_d, e_d, i_d = a_c + dOEs[0], e_c + dOEs[1], i_c + dOEs[2]
                RAAN_d, AOP_d, TA_d = RAAN_c + dOEs[3], AOP_c + dOEs[4], TA_c + dOEs[5]
                deputy_r_des, deputy_v_des = orbital_elements_to_inertial(a_d, e_d, i_d, RAAN_d, AOP_d, TA_d, units='deg')
                deputy_rho_des, deputy_rho_dot_des = inertial_to_rel_LVLH(deputy_r_des, deputy_v_des, chief_r, chief_v)
    else:
        # No GNC section → create empty placeholders
        guidance, navigation, control = {}, {}, {}

    gnc_input = {
        "trajectory": None,
        "satellites": {"chief": chief, "deputy": deputy},
        "simulation": sim_config,
        "output": output_config,
        "desired_relative_state": {
            "deputy_rho_des": deputy_rho_des,
            "deputy_rho_dot_des": deputy_rho_dot_des
        },
        "guidance": guidance,
        "navigation": navigation,
        "control": control
    }

    # Postprocessing
    postprocess_input = {
        "trajectory_file": output_config.get("trajectory_file", "data/results/trajectory.csv"),
        "gnc_file": output_config.get("gnc_file", "data/results/gnc_results.csv"),
        "plots": output_config.get("plots", True),
        "propagator": propagator
    }

    return {
        "simulation": sim_config,
        "dynamics": dynamics_input,
        "gnc": gnc_input,
        "postprocess": postprocess_input,
        "raw": raw_config
    }