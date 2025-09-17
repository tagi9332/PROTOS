import numpy as np

def inertial_to_orbital_elements(r, v, mu=398600.4418):
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
    """
    Convert classical orbital elements to inertial state vectors (r, v).
    a: semi-major axis (km)
    e: eccentricity
    i: inclination (deg)
    AOP: argument of periapsis (deg)
    RAAN: right ascension of ascending node (deg)
    TA: true anomaly (deg)
    mu: gravitational parameter (km^3/s^2), default is Earth's

    Returns: r (km), v (km/s)
    """
    # Convert angles to radians
    i = np.radians(i)
    AOP = np.radians(AOP)
    RAAN = np.radians(RAAN)
    TA = np.radians(TA)

    # Distance from focus to satellite
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