import numpy as np
from scipy.integrate import solve_ivp

# Earth constants
MU_EARTH = 398600.4418  # km^3/s^2
R_EARTH = 6378.137      # km
J2 = 1.08263e-3

def _relative_dynamics(t, y, chief_r, n, perturb):
    """
    Relative dynamics model: Hill-Clohessy-Wiltshire + perturbations.
    y = [x, y, z, vx, vy, vz] in RIC relative to chief
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
        factor = 1.5 * J2 * MU_EARTH * R_EARTH**2 / r_total**5
        ax -= factor * (1 - 5*z2/r_total**2) * (chief_r[0] + x)
        ay -= factor * (1 - 5*z2/r_total**2) * (chief_r[1] + y_pos)
        az -= factor * (3 - 5*z2/r_total**2) * (chief_r[2] + z)

    return [vx, vy, vz, ax, ay, az]


def step(state: dict, dt: float, config: dict) -> dict:
    """
    Advance the deputy's relative state by one time step.
    
    state: dict with chief_r, chief_v, deputy_r, deputy_v
    dt: timestep [s]
    config: dynamics config dict
    """
    chief_r = np.array(state["chief_r"])
    deputy_r = np.array(state["deputy_r"])
    deputy_v = np.array(state["deputy_v"])
    sim = config.get("simulation", {})
    perturb = sim.get("perturbations", {})

    # Mean motion of chief (circular assumption)
    r_mag = np.linalg.norm(chief_r)
    n = np.sqrt(MU_EARTH / r_mag**3)

    # Initial relative state
    y0 = np.hstack((deputy_r, deputy_v))

    # Integrate for one step [0, dt]
    sol = solve_ivp(
        _relative_dynamics,
        (0, dt),
        y0,
        t_eval=[dt],
        rtol=1e-9,
        atol=1e-12,
        args=(chief_r, n, perturb)
    )

    y_next = sol.y[:, -1]

    # Return updated state
    return {
        "chief_r": chief_r,        # keeping chief fixed here
        "chief_v": state["chief_v"],
        "deputy_r": y_next[0:3],
        "deputy_v": y_next[3:6],
    }
