import gymnasium as gym
import robosuite as suite
from robosuite.environments.manipulation.lift import Lift
from robosuite.wrappers import GymWrapper   
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
            reward += 1 - np.tanh(10.0 * reach_dist)

            # grasping reward
            reward += self.add_grasp_reward()

            # dense lifting reward
            # cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
            # table_height = self.model.mujoco_arena.table_offset[2]
            # # 1: >= 4cm; normalized to [0, 1] w.r.t. 4cm lift height
            # lift_progress = np.clip((cube_height - table_height) / 0.04, 0.0, 1.0)
            # reward += 1.0 * lift_progress

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

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

