def print_sim_header(file_path, sim_config, init_state):
    """
    Prints a professional styled header and simulation summary to the terminal.
    """
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    # Count deputies
    num_deputies = len(init_state.get("deputies", {}))
    if num_deputies >1:
        deputy_str = f"{num_deputies} Deputies"
    elif num_deputies == 1:
        deputy_str = "1 Deputy"
    else:
        deputy_str = "No Deputies"
    mode = sim_config.simulation_mode.upper()

    # --- Main Banner ---
    print(f"\n{CYAN}{'='*65}{RESET}")
    print(f"{CYAN}{BOLD}  PROTOS Satellite Simulation Framework{RESET}")
    print(f"{CYAN}{'='*65}{RESET}")
    
    # --- Metadata Table ---
    print(f" {BOLD}Config File:{RESET}  {file_path}")
    print(f" {BOLD}Mode:{RESET}         {mode}")
    print(f" {BOLD}Time Step:{RESET}    {sim_config.time_step}s")
    print(f" {BOLD}Formation:{RESET}    Chief + {deputy_str}")
    print(f"{CYAN}{'-'*65}{RESET}")

    print(f"{CYAN}{'-'*65}{RESET}")
    print("") # Final spacer