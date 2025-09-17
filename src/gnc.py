def step(state: dict, config: dict) -> dict:
    """
    Placeholder GNC step.
    Right now just echoes back the deputy state without modification.
    
    state: dict with chief/deputy positions and velocities
    config: GNC config dictionary (currently unused)
    
    Returns a dictionary for this timestep.
    """
    return {
        "status": "pass-through",
        "chief_r": state["chief_r"].tolist(),
        "chief_v": state["chief_v"].tolist(),
        "deputy_r": state["deputy_r"].tolist(),
        "deputy_v": state["deputy_v"].tolist(),
        "deputy_rho": state["deputy_rho"].tolist(),
        "deputy_rho_dot": state["deputy_rho_dot"].tolist()
    }


def run(trajectory: dict, config: dict) -> dict:
    """
    Backward-compatible wrapper for batch processing.
    Calls step() on each trajectory point.
    """
    times = trajectory.get("time", [])
    states = trajectory.get("state", [])

    results = []
    for t, y in zip(times, states):
        step_out = {
            "t": t,
            "status": "pass-through",
            "state": y
        }
        results.append(step_out)

    return {
        "status": "batch-pass-through",
        "time": times,
        "state": states,
        "results": results
    }
