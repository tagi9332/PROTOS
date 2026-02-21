import numpy as np
import dateutil.parser
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any


@dataclass
class OutputConfig:
    trajectory_file: str = "data/results/trajectory.csv"
    gnc_file: str = "data/results/gnc_results.csv"
    plots: bool = True

@dataclass
class SimulationConfig:
    time_step: float = 1.0
    duration: float = 3600.0
    epoch: datetime = field(default_factory=lambda: dateutil.parser.isoparse("2000-01-01T12:00:00Z"))
    propagator: str = "2BODY"
    simulation_mode: str = "3DOF"
    t_eval: np.ndarray = field(init=False)

    def __post_init__(self):
        self.propagator = self.propagator.upper()
        steps = int(self.duration / self.time_step) + 1
        self.t_eval = np.linspace(0, self.duration, steps)

@dataclass
class SimConfig:
    simulation: SimulationConfig
    output: OutputConfig  

def init_sim_config(raw_config: dict) -> SimConfig:
    """Initialize the simulation configuration by parsing the raw input dictionary."""
    # Extract simulation parameters
    sim_raw = raw_config.get("simulation", {})
    out_raw = raw_config.get("output", {})

    # Handle the epoch string parsing safely
    epoch_str = sim_raw.get("epoch")
    if epoch_str is not None:
        epoch_val = dateutil.parser.isoparse(epoch_str)
    else:
        epoch_val = dateutil.parser.isoparse("2000-01-01T12:00:00Z")

    # Instantiate the simulation config
    sim_config = SimulationConfig(
        time_step=sim_raw.get("time_step", 1.0),
        duration=sim_raw.get("duration", 3600.0),
        epoch=epoch_val,
        propagator=sim_raw.get("propagator", "2BODY"),
        simulation_mode=sim_raw.get("simulation_mode", "3DOF")
    )

    # Instantiate the output config
    output_raw = raw_config.get("output", {})
    output_config = OutputConfig(
        trajectory_file=output_raw.get("trajectory_file", "data/results/trajectory.csv"),
        gnc_file=output_raw.get("gnc_file", "data/results/gnc_results.csv"),
        plots=output_raw.get("plots", True)
    )

    # Return the unified configuration
    return SimConfig(simulation=sim_config, output=output_config)