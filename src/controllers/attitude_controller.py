def attitude_step(state: dict, config: dict) -> dict:
    """
    Placeholder 6DOF attitude controller.
    Currently returns zero torques for both Chief and Deputy.

    Parameters
    ----------
    state : dict
        Dictionary containing the current state of chief/deputy
    config : dict
        Dictionary containing GNC configuration

    Returns
    -------
    dict
        torque_chief : list[float] -- 3-element zero torque vector
        torque_deputy : list[float] -- 3-element zero torque vector
    """
    torque_chief = [0.0, 0.0, 0.0]
    torque_deputy = [0.0, 0.0, 0.0003]

    return {
        "torque_chief": torque_chief,
        "torque_deputy": torque_deputy
    }
