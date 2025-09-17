
## Linearized relative motion propagation
import numpy as np
from utils.frame_convertions.rel_to_inertial_functions import LVLH_DCM, rel_vector_to_inertial, compute_omega
from .constants import MU_EARTH, R_EARTH, J2

def step_linearized_2body(state: dict, dt: float, config: dict):
    import numpy as np

    chief_r = state["chief_r"]
    chief_v = state["chief_v"]
    deputy_rho = state["deputy_rho"]
    deputy_rho_dot = state["deputy_rho_dot"]
    sim = config.get("simulation", {})
    perturb = sim.get("perturbations", {})
    mu = sim.get("mu", 398600.4418)  # Earth GM [km^3/s^2]

    # Compute chief orbit parameters
    r_c = np.linalg.norm(chief_r)
    v_c = np.linalg.norm(chief_v)
    f_dot = v_c / r_c  # orbital angular velocity (rad/s)
    r_d = np.linalg.norm(chief_r + deputy_rho)  # deputy distance from Earth

    def rhs(t, y):
        """Compute derivatives of y = [x, y, z, x_dot, y_dot, z_dot] in LVLH"""
        x, y_, z = y[:3]
        vx, vy, vz = y[3:]

        # Equations of motion in LVLH frame
        ax = 2 * f_dot * (vy - y_ * (0)) + x * (-f_dot**2) - mu / r_d**3 * (r_c + x) 
        ay = -2 * f_dot * (vx - x * (0)) + y_ * (-f_dot**2) - mu / r_d**3 * y_
        az = -mu / r_d**3 * z

        return np.array([vx, vy, vz, ax, ay, az])

    # Current state vector
    y0 = np.hstack((deputy_rho, deputy_rho_dot))

    # RK4 integration
    k1 = rhs(0, y0)
    k2 = rhs(dt/2, y0 + dt/2 * k1)
    k3 = rhs(dt/2, y0 + dt/2 * k2)
    k4 = rhs(dt, y0 + dt * k3)
    y_next = y0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    deputy_rho_next = y_next[:3]
    deputy_rho_dot_next = y_next[3:]

    # Propagate chief (still circular approx)
    theta = f_dot * dt
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    STM = np.array([[cos_theta, -sin_theta, 0],
                    [sin_theta,  cos_theta, 0],
                    [0,          0,         1]])
    chief_r_next = STM @ chief_r
    chief_v_next = STM @ chief_v

    # Compute deputy inertial state
    deputy_r_next, deputy_v_next = rel_vector_to_inertial(deputy_rho_next, deputy_rho_dot_next,
                                                          chief_r_next, chief_v_next)

    return {
        "chief_r": chief_r_next,
        "chief_v": chief_v_next,
        "deputy_r": deputy_r_next,
        "deputy_v": deputy_v_next,
        "deputy_rho": deputy_rho_next,
        "deputy_rho_dot": deputy_rho_dot_next
    }
