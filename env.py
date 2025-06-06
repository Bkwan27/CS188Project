import gymnasium as gym
import robosuite as suite
from robosuite.wrappers import GymWrapper   
import numpy as np
import robosuite.utils.sim_utils as SU


import gymnasium as gym
import numpy as np

class RewardOverrideWrapper(gym.Wrapper):
    """
    Gym-level wrapper that replaces Robosuite's reward with a custom one.
    """

    # --------------------------------------------------------------
    def __init__(self, env, reward_scale=10, reward_shaping=True):
        super().__init__(env)
        raw = env
        while hasattr(raw, "env"):       # e.g. GymWrapper, some VecWrapper, etc.
            raw = raw.env
        # At this point, `raw` should be something like a robosuite.environments.lift.Lift instance
        self.lift_env = raw
        self.base = self.env.unwrapped
        # ----- cached MuJoCo handles -----
        self.sim   = self.base.sim
        self.cube_bid = self.base.cube_body_id              # cube body id
        # if hasattr(base, "cube_geom_id"):
        #     self.cube_gid = base.cube_geom_id
        # else:
        #     geom_adr  = self.sim.model.body_geomadr[self.cube_bid]
        #     geom_num  = self.sim.model.body_geomnum[self.cube_bid]
        #     self.cube_gid = int(geom_adr)       # pick the first geom              # cube geom id
        self.ee_sid   = self.base.robots[0].eef_site_id['right']     # end-effector site id
        # self.l_finger_geom_ids = base.l_finger_geom_ids
        # self.r_finger_geom_ids = base.r_finger_geom_ids

        # ----- reward settings -----
        self.reward_scale   = reward_scale
        self.reward_shaping = reward_shaping
        self.prev_info      = None           # will store previous step's info
        self._table_top_z   = None           # filled in at reset()

    # --------------------------------------------------------------
    # helper: EE is above cube and aligned
    def in_grasp_window(
        self, xy_thresh=0.005, z_low=0.01, z_high=0.05, ori_thresh=0.9
    ) -> bool:
        cube_pos = self.sim.data.body_xpos[self.cube_bid]
        ee_pos   = self.sim.data.site_xpos[self.ee_sid]

        xy_dist  = np.linalg.norm(ee_pos[:2] - cube_pos[:2])
        z_diff   = ee_pos[2] - cube_pos[2]
        ee_z_vec = self.sim.data.site_xmat[self.ee_sid].reshape(3, 3)[:, 2]
        ori_ok   = ee_z_vec @ np.array([0, 0, -1]) > ori_thresh

        return (xy_dist < xy_thresh) and (z_low < z_diff < z_high) and ori_ok

    # --------------------------------------------------------------
    def _check_success(self) -> bool:
        cube_height = float(self.sim.data.body_xpos[self.cube_bid][2])
        return cube_height > self._table_top_z + 0.04     # 4 cm above table

    # --------------------------------------------------------------
    def _custom_reward(self, obs, info, prev_info) -> float:
        self.base = self.env.unwrapped 
        reward = 0.0

        # 1) sparse success
        cube_z = obs["cube_pos"][2]
        if cube_z > self._table_top_z + 0.04:
            reward += 1.0

        if self.reward_shaping:
            # 2) distance shaping
            # cube_pos = self.sim.data.body_xpos[self.cube_bid]
            # ee_pos   = self.sim.data.site_xpos[self.ee_sid]
            ee_pos = obs['robot0_eef_pos']
            cube_pos = obs['cube_pos']
            dist = np.linalg.norm(ee_pos - cube_pos)
            #print(dist)
            if dist < 0.01:
                reward += 0.25
            
            reward += (1 - np.tanh(10.0 * dist))
            info['ee_dist'] = float(dist)
            # 3) gripper gating
            gripper_to_cube = obs["gripper_to_cube_pos"]
            in_window = np.linalg.norm(gripper_to_cube) < 0.01

            gripper_qpos = obs["robot0_gripper_qpos"]  
            grasping = (gripper_qpos[0] < 0.01) and (gripper_qpos[1] < 0.01)

            if in_window:
                reward += 0.5
            if in_window and grasping:
                reward += 0.75
                height = cube_z - self._table_top_z
                LIFT_SCALE = 10.0
                if height > 0.0:
                    reward += height * LIFT_SCALE
            if in_window and not grasping:
                reward -= 0.25
            # reward lifting up cube
        # reward -= 0.01
        return reward * self.reward_scale / 2.25

    # --------------------------------------------------------------
    # Gymnasium API
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # cube rests on table → capture table height
        self._table_top_z = float(self.lift_env._get_observations()["cube_pos"][2])
        self.prev_info = info
        return obs, info

    def step(self, action):
        obs, _orig_r, terminated, truncated, info = self.env.step(action)
        # custom reward
        reward = self._custom_reward(self.lift_env._get_observations(), info, self.prev_info)
        

        # (optional) expose cube height for logging/debug
        info["cube_height"] = float(self.sim.data.body_xpos[self.cube_bid][2]) - self._table_top_z
        self.prev_info = info
        return obs, reward, terminated, truncated, info
