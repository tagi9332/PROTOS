import numpy as np
from utils.frame_conversions.rel_to_inertial_functions import inertial_to_rel_LVLH
from utils.orbital_element_conversions.oe_conversions import inertial_to_orbital_elements, orbital_elements_to_inertial

def init_gnc(raw_config, sim_config, chief, deputy, chief_r_init, chief_v_init, output_config):
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
                a_c, e_c, i_c, RAAN_c, AOP_c, TA_c = inertial_to_orbital_elements(chief_r_init, chief_v_init, units='deg')

                dOEs = desired_state
                a_d, e_d, i_d = a_c + dOEs[0], e_c + dOEs[1], i_c + dOEs[2]
                RAAN_d, AOP_d, TA_d = RAAN_c + dOEs[3], AOP_c + dOEs[4], TA_c + dOEs[5]
                deputy_r_des, deputy_v_des = orbital_elements_to_inertial(a_d, e_d, i_d, RAAN_d, AOP_d, TA_d, units='deg')
                deputy_rho_des, deputy_rho_dot_des = inertial_to_rel_LVLH(deputy_r_des, deputy_v_des, chief_r_init, chief_v_init)
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

    return gnc_input