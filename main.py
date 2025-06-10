import gymnasium as gym
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite import load_composite_controller_config
import time
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from env import RewardOverrideWrapper
#from default_env import RewardOverrideWrapper

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from argparse import ArgumentParser

# 4yQo6f3Z

class RewardPrinter(BaseCallback):
    def _on_step(self) -> bool:
        # `infos` is a list (one per env in the VecEnv)
        for info in self.locals["infos"]:
            if "episode" in info:                 # episode just finished
                r = info["episode"]["r"]
                l = info["episode"]["l"]
                print(f"Episode done: return={r:.2f}, length={l} steps")
        return True                               # keep training

def make_lift_env():
    # config = load_composite_controller_config(controller="BASIC")
    env = RewardOverrideWrapper(
        robots="Panda",  
        # controller_configs=config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        horizon=500,
    )
    env = GymWrapper(env)
    env = Monitor(env)
    return env

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--continue_train', type=bool, default=False)
    parser.add_argument('--model', type=str, default='PPO')
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for the model")
    return parser.parse_args()

def main():
    args = parse_args()
    num_env = 1
    vec_env = DummyVecEnv([make_lift_env for _ in range(num_env)])

    if args.model == 'PPO':
        if args.continue_train:
            print("Loading existing PPO model...")
            model = PPO.load("PPO6_lift_500000_steps_0.01to0.02.zip", env=vec_env, verbose=1, learning_rate=args.lr, tensorboard_log="./ppo_lift_tb/")
        else:
            model = PPO("MlpPolicy", env=vec_env, verbose=1, learning_rate=args.lr, tensorboard_log="./ppo_lift_tb/")
    elif args.model == 'SAC':
        if args.continue_train:
            print("Loading existing SAC model...")
            model = SAC.load("SAC.zip", env=vec_env, verbose=1, learning_rate=3e-4, tensorboard_log="./sac_lift_tb/")
        else:
            model = SAC("MlpPolicy", env=vec_env, verbose=1, learning_rate=args.lr, tensorboard_log="./sac_lift_tb/")

    callback = RewardPrinter()
    print("Training PPO on Lift environment...")

    model.learn(total_timesteps=1000000, callback=callback)
    model_filename = f"{args.model}6_lift_{1000000}_steps_10highrew_lift_0.2.zip"
    model.save(model_filename)

    print("Testing trained model...")
    obs = vec_env.reset()
    episode_rewards = [0.0] * num_env
    episode_lengths = [0] * num_env
    episode_counts = [0] * num_env

    for i in range(101):
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
        
        if i % 10 == 0:
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

if __name__ == "__main__":
    main()