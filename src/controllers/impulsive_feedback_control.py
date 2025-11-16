import numpy as np

# -------------------------------
# Quiz 12
# -------------------------------

# Orbit and desired changes
# Chief orbit parameters
i_c_deg = 53          # chief inclination in degrees
Omega_c_deg = 55      # chief RAAN in degrees

# Deputy orbit parameters
i_d_deg = 56          # deputy inclination in degrees
Omega_d_deg = 51      # deputy RAAN in degrees
a = 8500e3            # semi-major axis in meters
e = 0.0               # circular orbit

# Desired relative differences
delta_i_desired_deg = 1.0      # desired delta inclination (deg)
delta_Omega_desired_deg = -1.0 # desired delta RAAN (deg)

# Convert degrees to radians
i_c = np.radians(i_c_deg)
i_d = np.radians(i_d_deg)
Omega_c = np.radians(Omega_c_deg)
Omega_d = np.radians(Omega_d_deg)

delta_i_desired = np.radians(delta_i_desired_deg)
delta_Omega_desired = np.radians(delta_Omega_desired_deg)

# 1. Inclination difference tracking error Δi
delta_i_current = i_d - i_c
tracking_error_i = delta_i_current - delta_i_desired
tracking_error_i_deg = np.degrees(tracking_error_i)

# 2. Deputy true latitude angle θ_d
# Formula: tan(theta_d) = (ΔΩ * sin(i)) / Δi
theta_d_rad = np.arctan((delta_Omega_desired * np.sin(i_c)) / delta_i_desired)
theta_d_deg = np.degrees(theta_d_rad)

# 3. Delta-v magnitude Δv_h
# For circular orbit: h = sqrt(mu * a)
# Δv_h = (h / r) * sqrt( (Δi)^2 + (ΔΩ)^2 * sin^2(i) )
mu = 398600.0e9  # m^3/s^2
r = a            # circular orbit
h = np.sqrt(mu * a)

delta_v_h = (h / r) * np.sqrt(delta_i_desired**2 + (delta_Omega_desired**2) * np.sin(i_c)**2)

# 4. Burn direction relative to along-track θ_c
theta_c_rad = np.arctan((delta_Omega_desired * np.sin(i_c)) / delta_i_desired)
theta_c_deg = np.degrees(theta_c_rad)

# Print results
print(f"1. Inclination tracking error Δi: {tracking_error_i_deg:.3f} deg")
print(f"2. Deputy true latitude angle θ_d: {theta_d_deg:.3f} deg")
print(f"3. Out-of-plane delta-v magnitude Δv_h: {delta_v_h:.6f} m/s")
print(f"4. Burn direction relative to along-track θ_c: {theta_c_deg:.3f} deg")

# -------------------------------
# Quiz 13
# -------------------------------

# Orbit and desired changes
a = 8500e3        # semi-major axis in meters
e = 0.1           # eccentricity
i_deg = 56        # inclination in degrees
i = np.radians(i_deg)

# Desired changes in orbital elements
delta_Omega_deg = 0.1    # degrees
delta_omega_deg = 0.25   # degrees
delta_M_deg = -0.1       # degrees
delta_a_km = 0.10        # km
delta_e = 0.002

# Convert angles to radians
delta_Omega = np.radians(delta_Omega_deg)
delta_omega = np.radians(delta_omega_deg)
delta_M = np.radians(delta_M_deg)

# Convert delta_a to meters
delta_a = delta_a_km * 1e3

# Gravitational parameter for Earth
mu = 398600.0e9  # m^3/s^2

# Helper quantities
eta = np.sqrt(1 - e**2)
n = np.sqrt(mu / a**3)  # mean motion (rad/s)

# Delta-v formulas
# Radial burns
delta_v_rp = - (n * a / 4) * ( ((1 + e)**2 / eta) * (delta_omega + delta_Omega * np.cos(i)) + delta_M )
delta_v_ra = - (n * a / 4) * ( ((1 - e)**2 / eta) * (delta_omega + delta_Omega * np.cos(i)) + delta_M )

# Tangential burns
delta_v_theta_p = (n * a * eta / 4) * ( delta_a / a + delta_e / (1 + e) )
delta_v_theta_a = (n * a * eta / 4) * ( delta_a / a - delta_e / (1 - e) )

# Print results in m/s
print(f"Delta-v radial at periapsis (Δv_rp): {delta_v_rp:.6f} m/s")
print(f"Delta-v radial at apoapsis (Δv_ra): {delta_v_ra:.6f} m/s")
print(f"Delta-v tangential at periapsis (Δv_theta_p): {delta_v_theta_p:.6f} m/s")
print(f"Delta-v tangential at apoapsis (Δv_theta_a): {delta_v_theta_a:.6f} m/s")