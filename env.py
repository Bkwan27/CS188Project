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
    def __init__(self, dense=False, **kwargs):
        self.dense = dense
        self.reward_components = {}
        self.debug_info = {}
        super().__init__(**kwargs)

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
        self.reward_components.clear()
        self.debug_info.clear()

        # sparse completion reward
        if self._check_success():
            reward += 2.25
            self.reward_components["success"] = 2.25

        # use a shaping reward
        elif self.reward_shaping:

            # reaching reward
            reach_dist = self._gripper_to_target(
                gripper=self.robots[0].gripper, 
                target=self.cube.root_body, 
                target_type="body", 
                return_distance=True
            )
            reach_reward = 1 - np.tanh(10.0 * reach_dist)
            reward += reach_reward
            self.reward_components["reach"] = reach_reward
            self.debug_info["reach_dist"] = reach_dist

            # grasping reward
            grasp_reward = self.add_grasp_reward()
            reward += grasp_reward
            self.reward_components["grasp"] = grasp_reward

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def add_grasp_reward(self):
        gripper = self.robots[0].gripper

        # Sparse grasping reward
        if self._check_grasp(gripper=gripper, object_geoms=self.cube):
            return 0.5 if self.dense else 0.25

        return 0.0

