===============================================================
PROTOS: Proximity Operations Trajectory and Orbit Simulation
===============================================================

Overview
--------
PROTOS is a modular Python-based simulation framework for modeling 
spacecraft relative motion, guidance, navigation, and control (GNC) 
in multi-satellite scenarios. It provides time-stepped execution of 
dynamics with selectable propagators and controllers, optional perturbations, and 
postprocessing tools for trajectory visualization and data export.

Key Features
------------
- Modular structure (I/O, dynamics, GNC, postprocessing).
- Configurable via JSONX input files.
- Multiple propagators:
  * CWH (Clohessy-Wiltshire-Hill equations)
  * 2BODY (full two-body orbital propagation)
  * LINEARIZED_2BODY (linearized relative two-body dynamics)
  * TH (Tschauner-Hempel equations)
- Selectable GNC controllers:
  * Cartesian Continuous Feedback
  * Orbit Element Difference Continuous Feedback
- Support for multiple reference frames:
  * ECI (Earth-Centered Inertial)
  * LVLH (Local Vertical-Local Horizon)
  * OEs (Classical Orbital Elements) 
- Perturbations (toggleable):
  * J2
  * Drag
  * Solar radiation pressure
- Postprocessing:
  * Export trajectory and GNC results to CSV
  * Generate trajectory plots in multiple frames
    and ECI frames
  * Generate control and delta-V plots
  * Generate orbit element plots 

Repository Structure
--------------------
├── run_PROTOS.py     : Main entry point for running a simulation
├── data/
│   ├── resources/    : Contains astronomical constants
│   ├── results/      : Default file locations results save to 
├── src/
│   ├── controllers/  : Selectible GNC control methods
│   ├── io_utils/     : Input parsing and frame conversions
│   ├── post_process/ : Data saving and plotting utilities
│   ├── propagators/  : Collection of propagation models
│   ├── utils/        : Utility functions for frame and orbital conversions, etc.
├── tests/

Getting Started
---------------
1. Install Python 3.13+ and required dependencies found in requiremement.txt file

   You can install them with:
   pip install -r requirements.txt

2. Prepare a configuration file in JSONX format (example syntax in data/input_files/syntax.jsonx).

3. Run the simulation:
   python run_PROTOS.py

4. Access the results in data/results/

Input File Format
-----------------
Input configuration syntax is provided in JSONX format. Key sections include:

- "simulation": global settings such as duration, timestep, propagator, 
  and perturbations.
- "satellites": definitions of chief and deputy spacecraft, including 
  initial state, frame, mass, and properties.
- "gnc": configurable guidance and controller parameters for relative motion
- "output": settings for saving trajectory, GNC results, and plots.

Extensibility
-------------
PROTOS is designed to be modular and extensible. Users can:
- Add new propagators by creating functions in src/propagators/.
- Implement custom GNC logic in src/gnc.py and src.propagators/
- Modify postprocessing routines for additional analysis or plots.

License
-------
See included license.