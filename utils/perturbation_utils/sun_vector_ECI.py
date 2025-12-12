from astropy.coordinates import GCRS, get_sun
from astropy.time import Time
from astropy import units as u
import numpy as np

def get_sun_vector_eci(epoch):
    """
    Compute the Sun position in ECI (GCRS) frame at a given epoch.

    Parameters
    ----------
    epoch : datetime or str
        UTC epoch (e.g., datetime object or ISO string)

    Returns
    -------
    r_sun_eci : np.ndarray
        Sun position vector in ECI [km]
    """
    # Convert to Astropy Time
    t = Time(epoch, scale='utc')

    # Get Sun position in ICRS
    sun_icrs = get_sun(t)

    # Convert to GCRS (ECI) frame
    sun_gcrs = sun_icrs.transform_to(GCRS(obstime=t))

    # Extract cartesian coordinates in meters, convert to km
    r_sun_eci = sun_gcrs.cartesian.xyz.to_value(u.Unit("km")) # type: ignore

    # Convert to unit vector
    norm = np.linalg.norm(r_sun_eci)
    if norm == 0:
        raise ValueError("Sun vector norm is zero; check input epoch.")
    r_sun_eci = r_sun_eci / norm

    return np.array(r_sun_eci)