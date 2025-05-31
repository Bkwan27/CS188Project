import numpy as np

def compute_reward(env, info):
    """
    Args:
        env : the Robosuite environment instance
        info: dict returned by env.step(), includes object & gripper state
    Returns:
        reward : float
        done   : bool   # terminate episode?
    """
    # ---------- convenience handles ----------
    gripper_pos   = env._hand_pos                           # (x,y,z)
    cube_pos      = info["cube_pos"]                        # (x,y,z)
    gripper_open  = info["gripper_open"] > 0.04             # bool
    cube_grasped  = info["cube_grasped"]                    # bool (from env contact sensor)
    lift_height   = cube_pos[2] - env.table_height
    dt            = env.control_timestep

    # ---------- weights (tune!) ----------
    w_approach = 1.0
    w_align    = 5.0
    w_grasp    = 10.0
    w_lift     = 8.0
    w_time     = 0.01
    w_drop_pen = 20.0

    # ---------- shaping terms ----------
    # 1. Approach (dense, decays exponentially with distance)
    dist = np.linalg.norm(gripper_pos - cube_pos)
    r_approach = w_approach * np.exp(-3.0 * dist)

    # 2. Alignment bonus: only if within 2 cm & gripper is open
    aligned = (dist < 0.02) and gripper_open
    r_align  = w_align if aligned else 0.0

    # 3. Grasp reward (one-time)
    r_grasp = w_grasp if cube_grasped and not info.get("grasp_given", False) else 0.0

    # 4. Lift reward (dense height gain, only after grasp)
    target_height = 0.10   # 10 cm above table
    if cube_grasped:
        r_lift = w_lift * np.clip(lift_height / target_height, 0.0, 1.0)
    else:
        r_lift = 0.0

    # 5. Penalty for drops
    dropped = (not cube_grasped) and (lift_height > 0.05)
    r_drop  = -w_drop_pen if dropped else 0.0

    # 6. Small time penalty to encourage speed
    r_time = -w_time * dt

    reward = r_approach + r_align + r_grasp + r_lift + r_drop + r_time

    # ---------- termination criteria ----------
    success = cube_grasped and lift_height >= target_height
    done = success or dropped                     # end episode

    return reward, done
