def init_gnc(raw_config):
    gnc_section = raw_config.get("gnc", {})

    # Defaults
    guidance = navigation = control = {}

    if gnc_section:
        guidance = gnc_section.get("guidance", {})
        navigation = gnc_section.get("navigation", {})
        control = gnc_section.get("control", {})

    # Assemble GNC input without converting desired state
    gnc_input = {
        "guidance": guidance,
        "navigation": navigation,
        "control": control,
        "simulation_mode": raw_config.get("simulation", {}).get("simulation_mode", "3DOF").upper()
    }

    return gnc_input
