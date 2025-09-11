def run(trajectory: dict, config: dict):
    """
    Placeholder GNC module.
    Currently just passes through the trajectory without modification.
    
    trajectory: dict from dynamics.propagate
    config: GNC config dictionary (currently unused)
    
    Returns a dictionary compatible with postprocess.
    """
    # Wrap the trajectory in a results dict for compatibility
    gnc_results = {
        "status": "pass-through",
        "time": trajectory.get("time", []),
        "state": trajectory.get("state", [])
    }
    
    return gnc_results