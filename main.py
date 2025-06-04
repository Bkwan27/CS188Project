import gymnasium as gym
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

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
    
    gym_env = GymWrapper(env)
    return gym_env

num_env = 4
vec_env = DummyVecEnv([make_lift_env for _ in range(num_env)])
model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.1, tensorboard_log="./ppo_lift_tb/")

print("Training PPO on Lift environment...")
training_start = time.time()

model.learn(total_timesteps=1000)

training_time = time.time() - training_start
print(f"Training completed in {training_time:.2f} seconds")

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
