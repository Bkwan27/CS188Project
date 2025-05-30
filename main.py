import gymnasium as gym
import robosuite as suite
from robosuite.wrappers import GymWrapper
#from robosuite import load_controller_config

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


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