import gymnasium as gym
import robosuite as suite
from robosuite.environments.manipulation.lift import Lift
from robosuite.wrappers import GymWrapper   
import robosuite.utils.sim_utils as SU
from robosuite.utils.transform_utils import mat2quat
import numpy as np

import gymnasium as gym
import numpy as np

class RewardOverrideWrapper(Lift):
    """
    Lift wrapper that replaces Robosuite's Lift reward with a custom one.
    """
    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the cube is lifted

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward += 2.25

        # use a shaping reward
        elif self.reward_shaping:

            # reaching reward
            reach_dist = self._gripper_to_target(
                gripper=self.robots[0].gripper, target=self.cube.root_body, target_type="body", return_distance=True
            )
            cube_pos = self.sim.data.get_body_xpos(self.cube.root_body)
            eef_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id['right']]

            xy_dist = np.linalg.norm(eef_pos[:2] - cube_pos[:2])
            z_above = eef_pos[2] - cube_pos[2]
            dist = np.linalg.norm(eef_pos - cube_pos)

            # Add reward if gripper is well-aligned in XY and slightly above cube in Z
            if xy_dist < 0.03 and 0.03 < z_above < 0.1:
                reward += 0.3
            reward += 1 - np.tanh(10.0 * dist)

            # grasping reward
            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube):
                reward += 0.75
            reward += self.add_grasp_reward()

            # dense lifting reward
            cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
            table_height = self.model.mujoco_arena.table_offset[2]
            # 1: >= 4cm; normalized to [0, 1] w.r.t. 4cm lift height
            lift_progress = np.clip((cube_height - table_height) / 0.04, 0.0, 1.0)
            reward += 1.0 * lift_progress
            # time punishment for not lifting cube
            if reach_dist < 0.05 and lift_progress < 0.05:
                reward -= 0.01

            # discourage closing grip far from cube
            # print(self.robots[0].gripper)
            site_id = self.robots[0].eef_site_id['right']
            site_name = self.sim.model.site_id2name(site_id)
            grip_ctrl = self.sim.data.ctrl[self.robots[0].gripper_dof_idx]
            if np.mean(grip_ctrl) < -0.01 and reach_dist > 0.1:
                reward -= 0.1  # closing grip far from cube

            # achieve desired orientation
            # desired_downward_quat = np.array([0, 1, 0, 0])  # or tune for your environment
            # site_id = self.robots[0].eef_site_id['right']
            # site_name = self.sim.model.site_id2name(site_id)
            # current_mat = self.sim.data.get_site_xmat(site_name).flatten()
            # current_quat = mat2quat(current_mat)
            # dot = np.abs(np.dot(current_quat, desired_downward_quat))
            # alignment_reward = 1 - np.clip(1 - dot, 0.0, 1.0)
            # reward += 0.2 * alignment_reward



        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def _check_grasp(self, gripper, object_geoms):
        # Use Mujoco’s contact detection
        if isinstance(gripper, dict):
            gripper_geoms = []
            for g in gripper.values():
                gripper_geoms += [g.important_geoms["left_fingerpad"][0], g.important_geoms["right_fingerpad"][0]]
        else:
            gripper_geoms = [gripper.important_geoms["left_fingerpad"][0], gripper.important_geoms["right_fingerpad"][0]]

        for g in gripper_geoms:
            if not SU.check_contact(sim=self.sim, geoms_1=g, geoms_2=object_geoms):
                return False

        # Add a motion constraint: is the cube moving with the gripper?
        cube_vel = self.sim.data.get_body_xvelp(self.cube.root_body)
        # Get the name of the site from the ID
        site_id = self.robots[0].eef_site_id['right']
        site_name = self.sim.model.site_id2name(site_id)
        eef_vel = self.sim.data.get_site_xvelp(site_name)

        if np.linalg.norm(cube_vel - eef_vel) > 0.05:
            return False

        return True


    def add_grasp_reward(self):
        reward = 0.0
        gripper = self.robots[0].gripper

        # Sparse grasping reward
        if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube):
            reward += 0.75
            return reward

        # If not grasping, compute dense grasping reward
        if isinstance(gripper, dict):
            grippers = list(gripper.values())
        else:
            grippers = [gripper]

        for g in grippers:
            left_geom = g.important_geoms["left_fingerpad"][0]
            right_geom = g.important_geoms["right_fingerpad"][0]

            left_pos = self.sim.data.get_geom_xpos(left_geom)
            right_pos = self.sim.data.get_geom_xpos(right_geom)
            cube_pos = self.sim.data.get_body_xpos(self.cube.root_body)

            # Dense grasp reward: closer fingerpads = better
            left_dist = np.linalg.norm(left_pos - cube_pos)
            right_dist = np.linalg.norm(right_pos - cube_pos)
            proximity = 1 - np.tanh(10.0 * (left_dist + right_dist) / 2.0)
            reward += 0.5 * proximity

        return reward

