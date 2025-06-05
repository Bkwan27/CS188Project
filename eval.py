"""
Roll out a saved Stable-Baselines3 PPO model on Robosuite’s Lift task
with live rendering.

Usage
-----
$ conda activate lift_rl          # or whatever env you built
$ python run_saved_ppo.py         # make sure ppo_lift.zip is in CWD
"""

import os, time, numpy as np
from stable_baselines3 import PPO
import robosuite as suite
from env import RewardOverrideWrapper
from robosuite.wrappers import GymWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

import numpy as np, sys
sys.modules['numpy._core'] = np.core
sys.modules['numpy._core.numeric'] = np.core.numeric
# --------------------------------------------------------------------
# 0) (Windows) make sure MuJoCo picks a GUI-capable backend
#    "glfw" works on Linux/Mac too; use "wgl" if glfw gives a black window
os.environ.setdefault("MUJOCO_GL", "glfw")
# print("hello")
# --------------------------------------------------------------------
# 1) Build *one* Lift env exactly like the training one, but with a viewer
def make_lift_env(render=True):
    env = suite.make(
        env_name="Lift", 
        robots="Panda",  
        has_renderer=render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        horizon=500,
    )
    #env = Monitor(env)          # <- adds episode reward/length to info dict
    gym_env = GymWrapper(env)
    gym_env = RewardOverrideWrapper(gym_env)
    return gym_env

num_env=1
vec_env = DummyVecEnv([make_lift_env for _ in range(num_env)])

# --------------------------------------------------------------------
# 2) Load the policy (no need to pass env when we only want predict)
model = PPO.load("ppo_lift.zip", env=vec_env)  # env helps SB3 sanity-check spaces

obs = vec_env.reset()
episode, ep_return, ep_len = 1, 0.0, 0

print("Running saved policy — close the viewer window to stop.")
try:
    while True:
        # 3) Policy → action
        action, _ = model.predict(obs, deterministic=True)

        # 4) Step the simulator
        obs, rewards, dones, infos = vec_env.step(action)
        ep_return += rewards[0]
        ep_len += 1

        # 5) Render one frame (robosuite auto-renders when has_renderer==True,
        #    but calling explicitly lets you slow the loop if you like)
        vec_env.envs[0].render()

        if dones[0]:
            print(f"Episode {episode} | return = {ep_return:.3f} | length = {ep_len} steps")
            obs = vec_env.reset()
            episode  += 1
            ep_return = 0.0
            ep_len    = 0

        # Optional: cap FPS so it doesn’t run too fast
        time.sleep(1 / 60)          # 60 Hz viewer
except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    vec_env.close()
