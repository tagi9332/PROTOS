import numpy as np

def apply_thrust_saturation(accel_cmd: np.ndarray, sat_config: dict, mass: float, mode='omnidirectional') -> np.ndarray:
    """
    Limits the acceleration command based on the max_thrust.
    
    Modes:
    - 'independent': Limits each axis individually (cubic bounding volume).
    - 'omnidirectional': Limits the total combined magnitude (spherical bounding volume).
    """
    control_config = sat_config.get('control', {})
    max_thrust = control_config.get('max_thrust')
    # Default to independent if no mode is specified
    mode = control_config.get('saturation_mode', 'independent') 
    
    # If no limit is defined, return original command
    if max_thrust is None or max_thrust <= 0:
        return accel_cmd

    # Calculate max allowable acceleration: a_max = F_max / m
    max_accel = max_thrust / mass

    if mode == 'omnidirectional':
        # Calculate the magnitude of the requested acceleration vector
        accel_mag = np.linalg.norm(accel_cmd)
        
        # If the combined magnitude exceeds the max, scale it back proportionally
        if accel_mag > max_accel:
            saturated_accel = accel_cmd * (max_accel / accel_mag)
        else:
            saturated_accel = accel_cmd
            
    elif mode == 'independent':
        # Clip each axis individually within [-max_accel, max_accel]
        # Mimics 3 sets of independent thrusters
        saturated_accel = np.clip(
            accel_cmd, 
            -max_accel, 
            max_accel
        )
        
    else:
        raise ValueError(f"Unknown saturation_mode: '{mode}'. Use 'independent' or 'omnidirectional'.")

    return saturated_accel