import numpy as np

def atmos_density_expm_model(h_km):
    """
    Returns temperature [K], pressure [Pa], and density [kg/m^3]
    for a given altitude h_km [km], using an SI conversion of the
    U.S. Standard Atmosphere (troposphere/stratosphere model).
    """

    # Convert altitude to feet for direct use of the given equations
    h_ft = h_km * 3280.84

    # Piecewise conditions
    if h_ft < 36152:  # Troposphere
        T_F = 59 - 0.00356 * h_ft
        p_lbft2 = 2116 * ((T_F + 459.7) / 518.6) ** 5.256

    elif h_ft < 82345:  # Lower Stratosphere
        T_F = -70
        p_lbft2 = 473.1 * np.exp(1.73 - 0.000048 * h_ft)

    else:  # Upper Stratosphere
        T_F = -205.05 + 0.00164 * h_ft
        p_lbft2 = 51.97 * ((T_F + 459.7) / 389.98) ** (-11.388)

    # Density in slugs/ft^3
    rho_slugft3 = p_lbft2 / (1718 * (T_F + 459.7))

    # Convert to SI units
    T_K = (T_F - 32) * 5/9 + 273.15
    p_Pa = p_lbft2 * 47.8803
    rho_kgm3 = rho_slugft3 * 515.379

    return T_K, p_Pa, rho_kgm3


# Example usage
if __name__ == "__main__":
    for h in [0, 10000, 20000, 50000, 80000, 100000]:  # altitudes in meters
        T, p, rho = atmos_density_expm_model(h)
        print(f"h = {h:6.0f} m → T = {T:7.2f} K, p = {p:9.2f} Pa, rho = {rho:10.5f} kg/m³")
