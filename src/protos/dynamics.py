import numpy as np
from scipy.integrate import solve_ivp

# Earth constants
MU_EARTH = 398600.4418  # km^3/s^2
R_EARTH = 6378.137      # km
J2 = 1.08263e-3

def propagate(config: dict):
    """
    Numerically propagate the two-satellite scenario including optional perturbations.
    
    config: dict from io_utils.parse_input
    """
    # Extract inputs
    chief_r = np.array(config["chief_r"])  # km
    chief_v = np.array(config["chief_v"])  # km/s
    deputy_r = np.array(config["deputy_r"])  # relative km
    deputy_v = np.array(config["deputy_v"])  # relative km/s
    sim = config.get("simulation", {})
    perturb = sim.get("perturbations", {})

    dt = sim.get("time_step", 10.0)
    duration = sim.get("duration", 3600)
    steps = int(duration/dt) + 1
    t_span = (0, duration)
    t_eval = np.linspace(0, duration, steps)

    # Initial state vector: [x, y, z, vx, vy, vz] in RIC
    y0 = np.hstack((deputy_r, deputy_v))

    # Mean motion of chief (assumes circular orbit)
    r_mag = np.linalg.norm(chief_r)
    n = np.sqrt(MU_EARTH / r_mag**3)

    def dynamics(t, y):
        """
        Linearized Hill-Clohessy-Wiltshire + optional perturbations
        y = [x, y, z, vx, vy, vz] relative to chief in RIC
        """
        x, y_pos, z, vx, vy, vz = y
        
        # Basic Hill-Clohessy accelerations
        ax = 3*n**2*x + 2*n*vy
        ay = -2*n*vx
        az = -n**2*z

        # Add J2 perturbation if enabled
        if perturb.get("J2", False):
            r_total = np.linalg.norm(chief_r + np.array([x, y_pos, z]))
            z2 = (chief_r[2] + z)**2
            factor = 1.5*J2*MU_EARTH*R_EARTH**2/r_total**5
            ax -= factor*(1 - 5*z2/r_total**2)*(chief_r[0] + x)
            ay -= factor*(1 - 5*z2/r_total**2)*(chief_r[1] + y_pos)
            az -= factor*(3 - 5*z2/r_total**2)*(chief_r[2] + z)

        # Drag and SRP placeholders (implement if you have area, Cd, rho, etc.)
        # if perturb.get("drag", False):
        #     ax, ay, az = add_drag(ax, ay, az, ...)

        # if perturb.get("solar_pressure", False):
        #     ax, ay, az = add_srp(ax, ay, az, ...)

        return [vx, vy, vz, ax, ay, az]

    # Integrate using RK45
    sol = solve_ivp(dynamics, t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-12)

    trajectory = {
        "time": sol.t.tolist(),
        "state": sol.y.T.tolist()  # transpose so each row is one timestep
    }

    return trajectory
