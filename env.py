import gymnasium as gym
import robosuite as suite
from robosuite.wrappers import GymWrapper        # <-- new in robosuite 1.5
                                                #     works with Gym & Gymnasium
# 1) Build any robosuite task you like
rs_env = suite.make(
        env_name="Lift",        # e.g. Lift, PickPlace, Door, …
        robots="Panda",         # or "Sawyer", "UR5e", …
        has_renderer=True,      # MuJoCo viewer pops up (render_mode="human")
        use_camera_obs=False,   # turn on if you want pixel observations
        control_freq=20,        # Hz
)

# 2) Convert it to a modern Gymnasium env
env = GymWrapper(rs_env, gymnasium=True)         # gymnasium=True → returns
                                                 # (obs, info) from reset,
                                                 # (..., terminated, truncated, info)
                                                 # from step, etc.

obs, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:          # Gymnasium-way of saying “episode done”
        obs, info = env.reset()
env.close()
