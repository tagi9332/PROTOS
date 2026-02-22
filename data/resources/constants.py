from datetime import datetime

# ---------------------------------------------------------------------------
# PROTOS/data/resources/constants.py
# Vallado 2013, Fundamentals of Astrodynamics and Applications, 4th Edition
# ----------------------------------------------------------------------------

# Gravitational parameters [km^3/s^2]
MU_EARTH = 398600.4418
MU_MOON = 4902.800066
MU_SUN = 1.32712440018e11

# Other constants
R_EARTH = 6378.137  # km, Earth radius
R_MOON = 1737.4     # km, Moon radius

# Earth's rotation rate [rad/s]
OMEGA_EARTH = 7.2921159e-5

# J2 perturbation constant
J2 = 1.08262668e-3
J3 = -2.532153e-6
J4 = -1.61962159137e-6

# Solar radiation pressure at 1 AU [N/m^2]
P_SRP = 4.56e-6

# J2000 epoch in datetime format
J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)
