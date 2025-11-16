def init_postprocess(output_config, propagator):
    postprocess_input = {
        "trajectory_file": output_config.get("trajectory_file", "data/results/trajectory.csv"),
        "gnc_file": output_config.get("gnc_file", "data/results/gnc_results.csv"),
        "plots": output_config.get("plots", True),
        "propagator": propagator
    }

    return postprocess_input