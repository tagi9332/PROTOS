import numpy as np

from utils.frame_convertions.rel_to_inertial_functions import rel_vector_to_inertial
from data.resources.constants import MU_EARTH

def inertial_to_orbital_elements(r, v, mu=MU_EARTH):
    """
    Convert inertial state vectors (r, v) to classical orbital elements.
    r: position vector (km)
    v: velocity vector (km/s)
    mu: gravitational parameter (km^3/s^2), default is Earth's

    Returns: a, e, i, AOP, RAAN, TA
    """
    r = np.array(r)
    v = np.array(v)
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)

    # Eccentricity vector
    e_vec = (np.cross(v, h) / mu) - (r / r_norm)
    e = np.linalg.norm(e_vec)

    # Semi-major axis
    energy = v_norm**2 / 2 - mu / r_norm
    a = -mu / (2 * energy)

    # Inclination
    i = np.arccos(h[2] / h_norm)

    # Node vector
    K = np.array([0, 0, 1])
    n = np.cross(K, h)
    n_norm = np.linalg.norm(n)

    # RAAN
    if n_norm != 0:
        RAAN = np.arccos(n[0] / n_norm)
        if n[1] < 0:
            RAAN = 2 * np.pi - RAAN
    else:
        RAAN = 0

    # Argument of Periapsis
    if n_norm != 0 and e > 1e-8:
        AOP = np.arccos(np.dot(n, e_vec) / (n_norm * e))
        if e_vec[2] < 0:
            AOP = 2 * np.pi - AOP
    else:
        AOP = 0

    # True Anomaly
    if e > 1e-8:
        TA = np.arccos(np.dot(e_vec, r) / (e * r_norm))
        if np.dot(r, v) < 0:
            TA = 2 * np.pi - TA
    else:
        TA = 0

    # Convert angles to degrees
    i = np.degrees(i)
    AOP = np.degrees(AOP)
    RAAN = np.degrees(RAAN)
    TA = np.degrees(TA)

    return a, e, i, AOP, RAAN, TA


def orbital_elements_to_inertial(a, e, i, AOP, RAAN, TA, mu=398600.4418):
    p = a * (1 - e**2)
    r_mag = p / (1 + e * np.cos(TA))

    # Perifocal coordinates
    r_pf = np.array([r_mag * np.cos(TA), r_mag * np.sin(TA), 0])
    v_pf = np.array([
        -np.sqrt(mu / p) * np.sin(TA),
        np.sqrt(mu / p) * (e + np.cos(TA)),
        0
    ])

    # Rotation matrix from perifocal to inertial
    cos_O = np.cos(RAAN)
    sin_O = np.sin(RAAN)
    cos_w = np.cos(AOP)
    sin_w = np.sin(AOP)
    cos_i = np.cos(i)
    sin_i = np.sin(i)

    R = np.array([
        [cos_O * cos_w - sin_O * sin_w * cos_i, -cos_O * sin_w - sin_O * cos_w * cos_i, sin_O * sin_i],
        [sin_O * cos_w + cos_O * sin_w * cos_i, -sin_O * sin_w + cos_O * cos_w * cos_i, -cos_O * sin_i],
        [sin_w * sin_i, cos_w * sin_i, cos_i]
    ])

    r = R @ r_pf
    v = R @ v_pf

    return r, v

# compute mean motion from inertial state vectors
def compute_mean_motion_from_ECI(r, v, mu=398600.4418):
    """
    Compute the mean motion from inertial state vectors (r, v).
    r: position vector (km)
    v: velocity vector (km/s)
    mu: gravitational parameter (km^3/s^2), default is Earth's

    Returns: mean motion n (rad/s)
    """
    r = np.array(r)
    v = np.array(v)
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)

    # Specific orbital energy
    energy = v_norm**2 / 2 - mu / r_norm

    # Semi-major axis
    a = -mu / (2 * energy)

    # Mean motion
    n = np.sqrt(mu / a**3)

    return n

# Compute inertial state vector from LROEs
def lroes_to_inertial(t, chief_r, chief_v, lroes, mu=398600.4418):
    """
    Convert LROEs to inertial state vectors (r, v) of the deputy.

    """
    chief_r = np.array(chief_r)
    chief_v = np.array(chief_v)
    A_0, B_0, alpha, beta, x_offset, y_offset = lroes

    # Compute chief mean motion
    n = compute_mean_motion_from_ECI(chief_r, chief_v, mu)

    # Compute A_1 and A_2
    A_1 = A_0 * np.cos(alpha)
    A_2 = A_0 * np.sin(alpha)

    # Compute B_1 and B_2
    B_1 = B_0 * np.cos(beta)
    B_2 = B_0 * np.sin(beta)

    # Compute position state
    x = A_1 * np.cos(n*t) - A_2 * np.sin(n*t) + x_offset
    y = -2*A_1 * np.sin(n*t) - 2*A_2 * np.cos(n*t) - (3/2)*(n*x_offset*t) + y_offset
    z = B_1 * np.cos(n*t) + B_2 * np.sin(n*t)

    # Compute position state
    x_dot = -A_1*n*np.sin(n*t) - A_2*n*np.cos(n*t)
    y_dot = -2*A_1*n*np.cos(n*t) + 2*A_2*n*np.sin(n*t) - (3/2)*n*x_offset
    z_dot = -B_1*n*np.sin(n*t) + B_2*n*np.cos(n*t)

    # LVLH relative state vectors
    deputy_rho = np.array([x, y, z])
    deputy_rho_dot = np.array([x_dot, y_dot, z_dot])

    # Convert to inertial frame
    deputy_r, deputy_v = rel_vector_to_inertial(deputy_rho, deputy_rho_dot, chief_r, chief_v) 

    return deputy_r, deputy_v
