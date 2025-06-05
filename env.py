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
        self.base = self.env.unwrapped                      # raw Lift env

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
        self, xy_thresh=0.02, z_low=0.01, z_high=0.05, ori_thresh=0.9
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
    def _custom_reward(self, info, prev_info) -> float:
        self.base = self.env.unwrapped 
        reward = 0.0

        # 1) sparse success
        if self._check_success():
            reward += 1.0

        if self.reward_shaping:
            # 2) distance shaping
            cube_pos = self.sim.data.body_xpos[self.cube_bid]
            ee_pos   = self.sim.data.site_xpos[self.ee_sid]
            dist = np.linalg.norm(ee_pos - cube_pos)
            #print(dist)
            if dist < 0.02:
                reward += 0.25
            
            reward += (1 - np.tanh(10.0 * dist))
            info['ee_dist'] = float(dist)
            # 3) gripper gating
            in_window  = self.in_grasp_window()
            # grip_ctrl   = self.sim.data.ctrl[self.gripper_act_ids]
            # closing_cmd = np.mean(grip_ctrl) < -0.01

            # if closing_cmd and not in_window:
            #     reward -= 0.10            # closing too early
            # if (not closing_cmd) and in_window:
            #     reward -= 0.05            # failing to close on target

            # 4) contact bonus
            robot = self.env.unwrapped.robots[0]
            gripper = robot.gripper
            if isinstance(gripper, dict):          # {'right': PandaGripper} (or left/right)
                gripper = next(iter(gripper.values()))
            g_geoms = [gripper.important_geoms["left_fingerpad"], gripper.important_geoms["right_fingerpad"]]
            grasping = True
            for g_group in g_geoms:
                if not SU.check_contact(sim=self.sim, geoms_1=g_group, geoms_2=self.base.cube):
                    grasping = False
            # left, right = False, False
            # for i in range(self.sim.data.ncon):
            #     c = self.sim.data.contact[i]
            #     g1, g2 = c.geom1, c.geom2
            #     if (g1 in self.l_finger_geom_ids and g2 == self.cube_gid) or \
            #        (g2 in self.l_finger_geom_ids and g1 == self.cube_gid):
            #         left = True
            #     if (g1 in self.r_finger_geom_ids and g2 == self.cube_gid) or \
            #        (g2 in self.r_finger_geom_ids and g1 == self.cube_gid):
            #         right = True
            #print(f'Reward: {reward}')
            if in_window:
                reward += 0.5
            if in_window and grasping:
                reward += 0.75
            if in_window and not grasping:
                reward -= 0.25

            # reward lifting up cube
            if grasping:
                cube_z   = float(self.sim.data.body_xpos[self.cube_bid][2])
                height   = cube_z - self._table_top_z
                prev_h   = prev_info.get("cube_height", 0.0) if prev_info else 0.0
                info['cube_height'] = height
                reward += 2 * max(0.0, height - prev_h)            # only reward upward motion
        # reward -= 0.01
        return reward * self.reward_scale / 2.25

    # --------------------------------------------------------------
    # Gymnasium API
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # cube rests on table → capture table height
        self._table_top_z = float(self.sim.data.body_xpos[self.cube_bid][2])

        self.prev_info = info
        return obs, info

    def step(self, action):
        obs, _orig_r, terminated, truncated, info = self.env.step(action)

        # custom reward
        reward = self._custom_reward(info, self.prev_info)
        

        # (optional) expose cube height for logging/debug
        info["cube_height"] = float(self.sim.data.body_xpos[self.cube_bid][2]) - self._table_top_z
        self.prev_info = info
        return obs, reward, terminated, truncated, info
