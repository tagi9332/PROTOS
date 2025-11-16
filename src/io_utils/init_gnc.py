import numpy as np

def init_gnc(raw_config, sim_config, chief, deputy, output_config):
    gnc_section = raw_config.get("gnc", {})

    # Defaults
    desired_relative_state = np.zeros(6)
    desired_frame = "LVLH"     # default frame for RPO
    guidance = navigation = control = {}

    if gnc_section:
        guidance = gnc_section.get("guidance", {})
        navigation = gnc_section.get("navigation", {})
        control = gnc_section.get("control", {})

        if guidance.get("type", "").upper() == "RPO":
            rpo = guidance.get("rpo", {})

            # Directly store the raw desired state and frame
            desired_relative_state = np.array(
                rpo.get("deputy_desired_relative_state")
            )
            desired_frame = rpo.get("frame").upper()

    # Assemble GNC input without converting desired state
    gnc_input = {
        "trajectory": None,
        "satellites": {"chief": chief, "deputy": deputy},
        "simulation": sim_config,
        "output": output_config,
        "desired_relative_state": {
            "state": desired_relative_state,     # store raw
            "frame": desired_frame      # store frame (LVLH, DOES, etc.)
        },
        "guidance": guidance,
        "navigation": navigation,
        "control": control
    }

    return gnc_input
