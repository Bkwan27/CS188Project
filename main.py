import gymnasium as gym
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class RewardPrinter(BaseCallback):
    def _on_step(self) -> bool:
        # `infos` is a list (one per env in the VecEnv)
        for info in self.locals["infos"]:
            if "episode" in info:                 # episode just finished
                r = info["episode"]["r"]
                l = info["episode"]["l"]
                print(f"Episode done: return={r:.2f}, length={l} steps")
        return True                               # keep training

class RewardOverrideWrapper(gym.Wrapper):
    """
    Replace the robosuite reward with a custom one.

    Works on top of GymWrapper, so it obeys the 5-tuple Gymnasium API:
        obs, reward, terminated, truncated, info
    """
    def __init__(self, env):
        super().__init__(env)
        self.prev_info = None          # we’ll store the previous step’s info

        # --- one-time cache of useful IDs inside MuJoCo ---
        rs_env = self.unwrapped        # original robosuite env
        self.table_z = rs_env.table_offset[2]
        self.cube_bid = rs_env.cube_body_id
        self.ee_sid   = rs_env.robots[0].eef_site_id
        self.reward_scale = 10

        self.reward_shaping = True

    # ------------ your custom reward ------------
    def _custom_reward(self, info, prev_info) -> float:
        """
        Shape a dense reward for the Lift task.
        Change anything you like inside this function.
        """
        
        rs_env = self.unwrapped
        self.sim = rs_env.sim

        # # Current positions
        # cube_pos = sim.data.body_xpos[self.cube_bid]
        # ee_pos   = sim.data.site_xpos[self.ee_sid]

        # # 1) Approach shaping (smooth 1 − tanh)
        # dist = np.linalg.norm(ee_pos - cube_pos)
        # r_approach = 16*(0.5 - np.tanh(5*dist))

        # # 2) Grasp bonus when first getting three contacts
        # contacts      = info["gripper_contact_num"]
        # prev_contacts = prev_info["gripper_contact_num"] if prev_info else 0
        # grasping      = contacts >= 3
        # first_grasp   = grasping and prev_contacts < 3
        # r_grasp       = 10.0 if first_grasp else 0.0
        # r_hold        = 0.2  if grasping     else 0.0

        # # 3) Height shaping (reward *delta-z* each step)
        # h_now  = cube_pos[2] - self.table_z
        # h_prev = prev_info["cube_height"] if prev_info else 0.0
        # r_lift = 2.0 * max(0, h_now - h_prev)

        # # 4) Success / drop
        # success_height = 0.10          # 10 cm
        # r_success = 40.0 if (h_now > success_height and grasping) else 0.0
        # r_drop    = -40.0 if (prev_info and prev_info["cube_height"] > 0.02
        #                       and h_now < 0.02 and not grasping) else 0.0

        # # 5) Per-step time cost
        # r_time = -0.01

        # return r_approach + r_grasp + r_hold + r_lift + r_success + r_drop + r_time
        reward = 0.0
        # sparse completion reward
        if self._check_success():
            reward = 1.0
        # use a shaping reward
        if self.reward_shaping:
            # reaching reward
            cube_pos = self.sim.data.body_xpos[self.cube_body_id]
            gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            dist = np.linalg.norm(gripper_site_pos - cube_pos)
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward
            # grasping reward
            touch_left_finger = False
            touch_right_finger = False
            for i in range(self.sim.data.ncon):
                c = self.sim.data.contact[i]
                if c.geom1 in self.l_finger_geom_ids and c.geom2 == self.cube_geom_id:
                    touch_left_finger = True
                if c.geom1 == self.cube_geom_id and c.geom2 in self.l_finger_geom_ids:
                    touch_left_finger = True
                if c.geom1 in self.r_finger_geom_ids and c.geom2 == self.cube_geom_id:
                    touch_right_finger = True
                if c.geom1 == self.cube_geom_id and c.geom2 in self.r_finger_geom_ids:
                    touch_right_finger = True
            if touch_left_finger and touch_right_finger:
                reward += 0.25

        return reward * self.reward_scale / 2.25
    
    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.table_full_size[2]
        # cube is higher than the table top above a margin
        return cube_height > table_height + 0.04

    # ------------ standard Gymnasium methods ------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_info = info
        return obs, info

    def step(self, action):
        obs, _orig_reward, terminated, truncated, info = self.env.step(action)
        # inject additional fields for convenience
        info["cube_height"] = self.unwrapped.sim.data.body_xpos[self.cube_bid][2] - self.table_z
        # compute custom reward
        reward = self._custom_reward(info, self.prev_info)
        self.prev_info = info
        return obs, reward, terminated, truncated, info

def make_lift_env():
    env = suite.make(
        env_name="Lift", 
        robots="Panda",  
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        horizon=500,
    )
    #env = Monitor(env)          # <- adds episode reward/length to info dict
    gym_env = GymWrapper(env)
    return gym_env

num_env = 4
vec_env = DummyVecEnv([make_lift_env for _ in range(num_env)])
model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.1, tensorboard_log="./ppo_lift_tb/")

model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=3e-4, tensorboard_log="./ppo_lift_tb/")

callback = RewardPrinter()
print("Training PPO on Lift environment...")


model.learn(total_timesteps=500000, callback=callback)
model.save("ppo_lift")

print("Testing trained model...")
obs = vec_env.reset()
episode_rewards = [0.0] * num_env
episode_lengths = [0] * num_env
episode_counts = [0] * num_env

for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = vec_env.step(action)
    
    for env_idx in range(num_env):
        episode_rewards[env_idx] += rewards[env_idx]
        episode_lengths[env_idx] += 1
        
        if dones[env_idx]:
            episode_counts[env_idx] += 1
            print(f"Env {env_idx}: Episode {episode_counts[env_idx]} finished at step {i}")
            print(f"  - Episode length: {episode_lengths[env_idx]} steps")
            print(f"  - Total reward: {episode_rewards[env_idx]:.3f}")
            print(f"  - Average reward per step: {episode_rewards[env_idx]/episode_lengths[env_idx]:.4f}")
            
            episode_rewards[env_idx] = 0.0
            episode_lengths[env_idx] = 0
    
    if i % 50 == 0:
        print(f"\nStep {i} Status:")
        for env_idx in range(num_env):
            print(f"  Env {env_idx}: Current reward = {rewards[env_idx]:.4f}, "
                  f"Episode progress = {episode_lengths[env_idx]} steps, "
                  f"Cumulative reward = {episode_rewards[env_idx]:.3f}")
        print(f"  Average reward across envs: {np.mean(rewards):.4f}")
        print("-" * 50)

print("\nFinal Statistics:")
for env_idx in range(num_env):
    print(f"Environment {env_idx}: Completed {episode_counts[env_idx]} episodes")
