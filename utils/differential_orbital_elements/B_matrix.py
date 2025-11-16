import numpy as np
from data.resources.constants import MU_EARTH as mu_e, R_EARTH as r_e

def B_matrix(a, e, i, argp, ta):
        """
        [B(oe)] matrix mapping accelerations [a_r, a_t, a_n] into 
        classical orbital element rates [a, e, i, Ω, ω, M].

        Inputs:
        - a     : semi-major axis
        - e     : eccentricity
        - f     : true anomaly
        - i     : inclination
        - argp  : argument of perigee
        - ta    : true anomaly
        """

        # Basic parameters
        p = a * (1 - e**2)              # semi-latus rectum
        h = np.sqrt(mu_e * p)           # specific angular momentum
        r = p / (1 + e * np.cos(ta))     # orbit radius
        true_lat = ta + argp         # true latitude

        # Define eta = sqrt(1 - e^2)
        eta = np.sqrt(1 - e**2)

        # ----- Build the full 6×3 B matrix -----
        B = np.zeros((6, 3))

        # Row 1: da/dt terms
        B[0, 0] = 2 * a**2 * e * np.sin(ta) / (h * r_e)
        B[0, 1] = 2 * a**2 * p / (h * r * r_e)
        B[0, 2] = 0

        # Row 2: de/dt terms
        B[1, 0] = p * np.sin(ta) / h
        B[1, 1] = ((p + r) * np.cos(ta) + r * e) / h
        B[1, 2] = 0

        # Row 3: di/dt terms
        B[2, 0] = 0
        B[2, 1] = 0
        B[2, 2] = (r * np.cos(true_lat)) / h

        # Row 4: dΩ/dt terms
        B[3, 0] = 0
        B[3, 1] = 0
        B[3, 2] = (r * np.sin(true_lat)) / (h * np.sin(i))

        # Row 5: dω/dt terms
        B[4, 0] = -p * np.cos(ta) / (h * e)
        B[4, 1] = (p + r) * np.sin(ta) / (h * e)
        B[4, 2] = -(r * np.sin(true_lat) * np.cos(i)) / (h * np.sin(i))

        # Row 6: dθ/dt terms
        B[5, 0] = eta * (p * np.cos(ta) - 2 * r * e) / (h * e)
        B[5, 1] = -eta * (p + r) * np.sin(ta) / (h * e)
        B[5, 2] = 0

        return B