from typing import Any, Dict, List

from src.io_utils.init_sim_config import SimulationConfig
from src.io_utils.init_satellites import SatelliteConfig

def init_gnc(satellites: List[SatelliteConfig], sim_config: SimulationConfig) -> Dict[str, Any]:
    """Initialize the GNC configuration based on the parsed satellite configurations and simulation settings."""
    gnc_input = {}
    sim_mode = sim_config.simulation_mode.upper()

    for sat in satellites:
            # sat.gnc is guaranteed to be at least an empty dict thanks to our dataclass
            gnc_section = sat.gnc

            # Safely extract sub-components, defaulting to new, separate empty dicts
            guidance = gnc_section.get("guidance", {})
            navigation = gnc_section.get("navigation", {})
            control = gnc_section.get("control", {})

            # Assemble GNC input for this specific satellite
            gnc_input[sat.name] = {
                "guidance": guidance,
                "navigation": navigation,
                "control": control,
                "simulation_mode": sim_mode
            }

    return gnc_input