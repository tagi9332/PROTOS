import numpy as np

from utils.frame_convertions.rel_to_inertial_functions import rel_vector_to_inertial
from data.resources.constants import MU_EARTH

def inertial_to_orbital_elements(R, V, mu=MU_EARTH):
    """
    Convert inertial state vectors (R, V) to classical orbital elements.
    R: position vector (km)
    V: velocity vector (km/s)
    mu: gravitational parameter (km^3/s^2), default is Earth's

    Returns: a, e, i, RAAN, AOP, TA (IN DEGREES)
    """
    R = np.array(R, dtype=float)
    V = np.array(V, dtype=float)

    # Eccentricity vector
    Ecc = (np.cross(V, np.cross(R, V)) / mu) - (R / np.linalg.norm(R))
    ecc = np.linalg.norm(Ecc)

    # Angular momentum
    H = np.cross(R, V)

    # Line of nodes
    N = np.cross([0, 0, 1], H)
    Nhat = N / np.linalg.norm(N)

    # Inclination
    i = np.degrees(np.arccos(np.dot(H, [0, 0, 1]) / np.linalg.norm(H)))

    # RAAN
    RAAN = np.degrees(np.arccos(Nhat[0]))
    if Nhat[1] < 0:
        RAAN = -RAAN

    # Argument of periapsis
    AOP = np.degrees(np.arccos(np.dot(Nhat, Ecc) / np.linalg.norm(Ecc)))
    if Ecc[2] < 0:
        AOP = -AOP

    # True anomaly
    TA = np.degrees(np.arccos(np.dot(Ecc, R) / (np.linalg.norm(R) * np.linalg.norm(Ecc))))
    if np.dot(R, V) < 0:
        TA = -TA

    # Semi-major axis
    a = 1 / ((2 / np.linalg.norm(R)) - (np.linalg.norm(V)**2 / mu))

    return a, ecc, i, RAAN, AOP, TA


def orbital_elements_to_inertial(a, e, i, RAAN, AOP, TA, mu=MU_EARTH):
    """
    Classical Orbital Elements -> inertial position/velocity.

    Inputs:
        a     : semi-major axis [km] (a<0 allowed for hyperbolic)
        e     : eccentricity (scalar,e!=1)
        i     : inclination [deg]
        RAAN  : right ascension of ascending node Ω [deg]
        AOP  : argument of perigee ω [deg]
        TA    : true anomaly f [deg]
        mu    : gravitational parameter [km^3/s^2]

    Returns:
        r : position vector in inertial frame [km]
        v : velocity vector in inertial frame [km/s]
    """

    # Convert angles from degrees to radians
    i = np.radians(i)
    RAAN = np.radians(RAAN)
    AOP = np.radians(AOP)
    TA = np.radians(TA)

    # semi-latus rectum (works for elliptical & hyperbolic; not defined for parabolic e=1)
    if np.isclose(e, 1.0, atol=1e-12):
        raise ValueError("Parabolic case (e close to 1) not supported.")
    p = a * (1.0 - e**2)

    cnu, snu = np.cos(TA), np.sin(TA)
    r_pf = (p / (1.0 + e * cnu)) * np.array([cnu, snu, 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-snu, e + cnu, 0.0])

    # Rotation PQW -> IJK
    cO, sO = np.cos(RAAN), np.sin(RAAN)
    ci, si = np.cos(i), np.sin(i)
    co, so = np.cos(AOP), np.sin(AOP)

    R3_O = np.array([[cO, -sO, 0.0], [sO, cO, 0.0], [0.0, 0.0, 1.0]])
    R1_i = np.array([[1.0, 0.0, 0.0], [0.0, ci, -si], [0.0, si, ci]])
    R3_o = np.array([[co, -so, 0.0], [so, co, 0.0], [0.0, 0.0, 1.0]])

    Q_pqw_to_ijk = R3_O @ R1_i @ R3_o

    r = Q_pqw_to_ijk @ r_pf
    v = Q_pqw_to_ijk @ v_pf
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
