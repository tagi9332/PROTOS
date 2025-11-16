import numpy as np
from utils.frame_conversions.rel_to_inertial_functions import rel_vector_to_inertial
from data.resources.constants import MU_EARTH

def step_linearized_2body(state: dict, dt: float, config: dict):
    chief_r = state["chief_r"]
    chief_v = state["chief_v"]
    deputy_rho = state["deputy_rho"]
    deputy_rho_dot = state["deputy_rho_dot"]
    
    sim = config.get("simulation", {})
    perturb_config = sim.get("perturbations", {})
    mu = sim.get("mu", MU_EARTH)

    r_c = np.linalg.norm(chief_r)
    v_c = np.linalg.norm(chief_v)
    f_dot = v_c / r_c  # orbital angular velocity

    # Transform LVLH RHS to include perturbations
    def rhs(t, y):
        x, y_, z = y[:3]
        vx, vy, vz = y[3:]

        # Compute deputy inertial position
        deputy_r_inertial = chief_r + np.array([x, y_, z])
        
        # Perturbation acceleration in inertial frame [TODO]
        a_pert_lvlh = np.zeros(3) # Placeholder for perturbation acceleration in LVLH frame

        # Linearized LVLH equations with perturbation
        ax = 2 * f_dot * vy - (mu / r_c**3) * x #+ a_pert_lvlh[0]
        ay = -2 * f_dot * vx - (mu / r_c**3) * y_ #+ a_pert_lvlh[1]
        az = - (mu / r_c**3) * z #+ a_pert_lvlh[2]

        return np.array([vx, vy, vz, ax, ay, az])

    # Current LVLH state vector
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
    deputy_r_next, deputy_v_next = rel_vector_to_inertial(
        deputy_rho_next, deputy_rho_dot_next, chief_r_next, chief_v_next
    )

    return {
        "chief_r": chief_r_next,
        "chief_v": chief_v_next,
        "deputy_r": deputy_r_next,
        "deputy_v": deputy_v_next,
        "deputy_rho": deputy_rho_next,
        "deputy_rho_dot": deputy_rho_dot_next
    }


