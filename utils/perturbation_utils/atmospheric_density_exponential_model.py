import numpy as np
from data.resources.constants import R_EARTH # Assuming this is in km

def rho_exp_model(r_mag_km, REarth_km=R_EARTH):
    """
    Estimate mass density using a simplified exponential model.
    
    Inputs:
        r_mag_km  -> [float or array] Magnitude of satellite radius vector [km]
        REarth_km -> [float] Earth radius [km], defaults to constant.

    Outputs:
        rho       -> [float or array] Density [kg/m^3]
    """
    # Reference density at 700km altitude
    rho0 = 3.614e-13  # [kg/m^3]
    
    # Reference radius in km
    r0_km = 700.0 + REarth_km  
    
    # Scale height in km
    H_km = 88.667  
    
    # Ensure input is a numpy array for vectorization
    r_km = np.asanyarray(r_mag_km)
    
    # Calculate density: rho = rho0 * exp(-(r - r0) / H)
    rho = rho0 * np.exp(-(r_km - r0_km) / H_km)
    
    return rho