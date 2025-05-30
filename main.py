import gymnasium as gym
import robosuite as suite
from robosuite.wrappers import GymWrapper
import numpy as np
#from robosuite import load_controller_config

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

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

    # ------------ your custom reward ------------
    def _custom_reward(self, info, prev_info) -> float:
        """
        Shape a dense reward for the Lift task.
        Change anything you like inside this function.
        """
        rs_env = self.unwrapped
        sim    = rs_env.sim

        # Current positions
        cube_pos = sim.data.body_xpos[self.cube_bid]
        ee_pos   = sim.data.site_xpos[self.ee_sid]

        # 1) Approach shaping (smooth 1 − tanh)
        dist = np.linalg.norm(ee_pos - cube_pos)
        r_approach = 2.0 * (1 - np.tanh(5 * dist))

        # 2) Grasp bonus when first getting three contacts
        contacts      = info["gripper_contact_num"]
        prev_contacts = prev_info["gripper_contact_num"] if prev_info else 0
        grasping      = contacts >= 3
        first_grasp   = grasping and prev_contacts < 3
        r_grasp       = 10.0 if first_grasp else 0.0
        r_hold        = 0.2  if grasping     else 0.0

        # 3) Height shaping (reward *delta-z* each step)
        h_now  = cube_pos[2] - self.table_z
        h_prev = prev_info["cube_height"] if prev_info else 0.0
        r_lift = 2.0 * max(0, h_now - h_prev)

        # 4) Success / drop
        success_height = 0.10          # 10 cm
        r_success = 40.0 if (h_now > success_height and grasping) else 0.0
        r_drop    = -40.0 if (prev_info and prev_info["cube_height"] > 0.02
                              and h_now < 0.02 and not grasping) else 0.0

        # 5) Per-step time cost
        r_time = -0.01

        return r_approach + r_grasp + r_hold + r_lift + r_success + r_drop + r_time

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
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        horizon=500,
    )
    
    gym_env = GymWrapper(env)
    return gym_env

vec_env = DummyVecEnv([make_lift_env])

model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=3e-4)

print("Training PPO on Lift environment...")
model.learn(total_timesteps=50000)
model.save("ppo_lift")

print("Testing trained model...")
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    if dones[0]:
        print(f"Episode finished at step {i}")
        obs = vec_env.reset()
    if i % 10 == 0:
        print(f"Step {i}, Reward: {rewards[0]}")