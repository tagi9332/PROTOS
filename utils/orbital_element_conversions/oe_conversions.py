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


def _wrap01(x):
    # wrap to [0, 2π)
    return np.mod(x, 2*np.pi)

def ta_to_m(TA, e):
    """
    True anomaly -> Mean anomaly (elliptic 0<=e<1).
    Handles scalar or ndarray TA. Returns M in [0,2π).
    """
    TA = np.atleast_1d(TA).astype(float)
    # principal eccentric anomaly via atan2 form (robust)
    # tan(E/2) = sqrt((1-e)/(1+e)) * tan(TA/2)
    E = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(TA/2.0),
                         np.sqrt(1.0 + e) * np.cos(TA/2.0))
    E = _wrap01(E)
    M = E - e * np.sin(E)
    M = _wrap01(M)
    return M.squeeze()

def m_to_ta(M, e, tol=1e-12, max_iter=100):
    """
    Mean anomaly -> True anomaly (elliptic 0<=e<1) using Newton-Raphson.
    Returns TA in [0, 2π).
    TA and E match the shape of input M (scalar or array).
    """
    M_in = np.atleast_1d(M).astype(float)
    M_wrapped = _wrap01(M_in)

    # initial guess for E
    E = np.copy(M_wrapped)
    # better guess for larger e
    mask = (e >= 0.8)
    if np.isscalar(E):
        if e >= 0.8:
            E = np.pi
    else:
        E[mask] = np.pi

    # Newton-Raphson solve for E
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M_wrapped
        fprime = 1.0 - e * np.cos(E)
        dE = f / fprime
        E = E - dE
        if np.max(np.abs(dE)) < tol:
            break
    else:
        raise RuntimeError("Kepler solve did not converge")

    E = _wrap01(E)
    # convert E->TA robustly using atan2 form
    sinv = np.sqrt(1.0 + e) * np.sin(E/2.0)
    cosv = np.sqrt(1.0 - e) * np.cos(E/2.0)
    TA = 2.0 * np.arctan2(sinv, cosv)
    TA = _wrap01(TA)

    # return shapes consistent with scalar input
    if np.isscalar(M):
        return float(TA.squeeze())
    return TA.squeeze()





#################################################

def normalize_angle(angle):
    """Wrap angle to [-pi, pi)."""
    a = (angle + np.pi) % (2.0 * np.pi) - np.pi
    return a


def rv_to_coe(r, v, mu=MU_EARTH):
    """
    Inertial r,v -> classical orbital elements (a, e, i, RAAN, argp, nu, M, E)
    Angles in radians. Returns tuple (a,e,i,RAAN,argp,nu,M,E)
    """
    r = np.array(r)
    v = np.array(v)
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    h_vec = np.cross(r, v)
    h_norm = np.linalg.norm(h_vec)
    # node vector
    k_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_hat, h_vec)
    n_norm = np.linalg.norm(n_vec)

    # eccentricity vector
    e_vec = (1.0 / mu) * ((v_norm**2 - mu / r_norm) * r - np.dot(r, v) * v)
    e = np.linalg.norm(e_vec)

    # specific mechanical energy
    eps = v_norm**2 / 2.0 - mu / r_norm
    if abs(eps) < 1e-12:
        a = np.inf
    else:
        a = -mu / (2.0 * eps)

    # inclination
    i = np.arccos(np.clip(h_vec[2] / h_norm, -1.0, 1.0))

    # RAAN
    if n_norm != 0:
        RAAN = np.arctan2(n_vec[1], n_vec[0])
    else:
        RAAN = 0.0

    # argument of perigee
    if n_norm != 0 and e > 1e-12:
        argp = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n_norm * e), -1.0, 1.0))
        if e_vec[2] < 0:
            argp = -argp
    else:
        argp = 0.0

    # true anomaly
    if e > 1e-12:
        nu = np.arccos(np.clip(np.dot(e_vec, r) / (e * r_norm), -1.0, 1.0))
        if np.dot(r, v) < 0:
            nu = 2.0 * np.pi - nu
    else:
        # circular: use angle between n_vec and r
        if n_norm != 0:
            nu = np.arccos(np.clip(np.dot(n_vec, r) / (n_norm * r_norm), -1.0, 1.0))
            if r[2] < 0:
                nu = 2.0 * np.pi - nu
        else:
            nu = 0.0

    # eccentric anomaly E and mean anomaly M (for elliptical e<1)
    if e < 1.0:
        # compute E from true anomaly
        E = 2.0 * np.arctan2(np.tan(nu / 2.0) * np.sqrt((1.0 - e) / (1.0 + e)), 1.0)
        M = E - e * np.sin(E)
        M = normalize_angle(M)
    else:
        E = None
        M = None

    # normalize angles
    RAAN = normalize_angle(RAAN)
    argp = normalize_angle(argp)
    nu = normalize_angle(nu)

    return a, e, i, RAAN, argp, nu, M, E