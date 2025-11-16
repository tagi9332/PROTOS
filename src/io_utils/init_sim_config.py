import dateutil.parser

def init_sim_config(raw_config: dict) -> dict:
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

    return {
        "output": output_config,
        "simulation": sim_config
    }