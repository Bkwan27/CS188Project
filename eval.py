import os
os.environ["MUJOCO_GL"] = "egl"  # must be before importing mujoco/robosuite

import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from robosuite.wrappers import GymWrapper
from env import RewardOverrideWrapper  # Your custom reward Lift class

def make_lift_env():
    env = RewardOverrideWrapper(
        robots="Panda",  
        has_renderer=False,                # disable GUI viewer (GLFW)
        has_offscreen_renderer=False,       # enable EGL offscreen rendering
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        horizon=500,
    )
    env = GymWrapper(env)
    return env

# Instantiate environment and model
num_env = 1
vec_env = DummyVecEnv([make_lift_env for _ in range(num_env)])
model = PPO.load("ppo_lift.zip", env=vec_env)

obs = vec_env.reset()
episode, ep_return, ep_len = 1, 0.0, 0

print("Running saved policy in headless mode (EGL, no viewer)...")

try:
    while True:
        # Predict and step
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)

        ep_return += rewards[0]
        ep_len += 1

        # Optional: render RGB frame for debug (can record/save here)
        # frame = vec_env.envs[0].render(mode='rgb_array')

        if dones[0]:
            print(f"Episode {episode} | Return: {ep_return:.3f} | Length: {ep_len} steps")
            obs = vec_env.reset()
            episode += 1
            ep_return = 0.0
            ep_len = 0

        time.sleep(1 / 60)  # Simulate real-time speed (60 FPS)

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    vec_env.close()
