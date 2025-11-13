===============================================================
PROTOS: Proximity Operations Trajectory and Orbit Simulation
===============================================================

Overview
--------
PROTOS is a modular Python-based simulation framework for modeling 
spacecraft relative motion, guidance, navigation, and control (GNC) 
in multi-satellite scenarios. It provides time-stepped execution of 
dynamics with selectable propagators, optional perturbations, and 
postprocessing tools for trajectory visualization and data export.

The framework is designed for research and education in space 
domain awareness (SDA), proximity operations (RPOD), and spacecraft 
dynamics.

Key Features
------------
- Modular structure (I/O, dynamics, GNC, postprocessing).
- Configurable via JSONX input files.
- Multiple propagators:
  * CWH (Clohessy-Wiltshire-Hill equations)
  * TH (Tschauner-Hempel equations) (!TODO!)
  * 2BODY (full two-body orbital propagation)
  * LINEARIZED_2BODY (linearized relative two-body dynamics)
- Support for multiple reference frames:
  * ECI (Earth-Centered Inertial)
  * LVLH (Local-Vertical Local-Horizontal)
  * OEs (Classical Orbital Elements)
  * LROEs (Linearized Relative Orbital Elements)
- Perturbations (toggleable):
  * J2
  * Drag
  * Solar radiation pressure
- Postprocessing:
  * Export trajectory and GNC results to CSV
  * Generate trajectory plots in RIC (Radial, In-track, Cross-track)
    and ECI frames

Repository Structure
--------------------
- main.py            : Main entry point for running a simulation
- src/
  - io_utils.py      : Input parsing and frame conversions
  - dynamics.py      : Dynamics propagation step function
  - gnc.py           : Placeholder GNC logic
  - postprocess.py   : Data saving and plotting utilities
  - propagators/     : Collection of propagation models
  - controllers/     : Selectible GNC control methods
  - utils/           : Utility functions for frame and orbital conversions
- data/
  - input_files/     : Example JSONX configuration files
  - results/         : Output trajectories, GNC results, and plots

Getting Started
---------------
1. Install Python 3.8+ and required dependencies:
   - numpy
   - matplotlib
   - commentjson
   - pymsis

   You can install them with:
   pip install numpy matplotlib commentjson pymsis

2. Prepare a configuration file in JSONX format (examples in data/input_files).

3. Run the simulation:
   python main.py

4. Check the results in data/results/:
   - CSV files of trajectory and GNC results
   - PNG plots of relative and inertial trajectories

Input File Format
-----------------
Input configuration is provided in JSONX format. Key sections include:

- "simulation": global settings such as duration, timestep, propagator, 
  and perturbations.
- "satellites": definitions of chief and deputy spacecraft, including 
  initial state, frame, mass, and properties.
- "output": settings for saving trajectory, GNC results, and plots.

Example (excerpt):
{
  "simulation": {
    "duration": 2000,
    "time_step": 0.1,
    "propagator": "2BODY",
    "perturbations": { "J2": true, "drag": false }
  },
  "satellites": [
    {
      "name": "Chief",
      "initial_state": {
        "state": [ ... ],
        "frame": "ECI"
      }
    },
    {
      "name": "Deputy",
      "initial_state": {
        "state": [ ... ],
        "frame": "LVLH"
      }
    }
  ]
}

Outputs
-------
- CSV files containing time histories of states and relative motion.
- 3D and 2D plots of deputy trajectory in the RIC frame.
- 3D plots of chief and deputy trajectories in the ECI frame.

Extensibility
-------------
PROTOS is designed to be modular and extensible. Users can:
- Add new propagators by creating functions in src/propagators/.
- Implement custom GNC logic in src/gnc.py.
- Modify postprocessing routines for additional analysis or plots.

License
-------
See included license.