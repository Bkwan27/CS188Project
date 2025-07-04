import gymnasium as gym
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
import time
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from env import RewardOverrideWrapper

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
import numpy as np
from argparse import ArgumentParser

# 4yQo6f3Z

class RewardPrinter(BaseCallback):
    def _on_step(self) -> bool:
        # `infos` is a list (one per env in the VecEnv)
        for i, info in enumerate(self.locals["infos"]):
            if "episode" in info:
                r = info["episode"]["r"]
                l = info["episode"]["l"]
                print(f"Episode done: return={r:.2f}, length={l} steps")
            
                env = self.training_env.envs[i].unwrapped
                if hasattr(env, "debug_info"):
                    debug_info = env.debug_info
                    for key, value in debug_info.items():
                        if value is not None:
                            self.logger.record(f"debug/{key}", value)

                if hasattr(env, "reward_components"):
                    reward_info = env.reward_components
                    for key, value in reward_info.items():
                        if value is not None:
                            self.logger.record(f"reward_components/{key}", value)
        return True

def make_lift_env():
    args = parse_args()
    env = RewardOverrideWrapper(
        robots="Panda",  
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=args.reward_shaping,
        control_freq=20,
        horizon=500,
    )
    env = GymWrapper(env)
    env = Monitor(env)
    return env

def make_door_env():
    env = suite.make(
        env_name="Door", 
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        horizon=500,    
        reward_scale=1.0,
    )
    
    gym_env = GymWrapper(env)
    gym_env = Monitor(gym_env)
    return gym_env

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, choices=['lift', 'door'], default='lift', help="Task to train on")
    parser.add_argument('--continue_train', action='store_true', help="Continue training from saved model")
    parser.add_argument('--reward_shaping', action='store_true', help="Enable reward shaping")
    parser.add_argument('--no_reward_shaping', dest='reward_shaping', action='store_false', help="Disable reward shaping")
    parser.set_defaults(reward_shaping=True)
    parser.add_argument('--checkpoint', type=str, help="Checkpoint file to load for training")
    parser.add_argument('--timesteps', type=int, default=500_000, help="Number of training timesteps")

    return parser.parse_args()

def main():
    args = parse_args()
    num_env = 1
    if args.task == 'lift':
        vec_env = DummyVecEnv([make_lift_env for _ in range(num_env)])
    elif args.task == 'door':
        vec_env = DummyVecEnv([make_door_env for _ in range(num_env)])

    if args.continue_train:
        print("Loading existing SAC model...")
        model = SAC.load(
            args.checkpoint, 
            env=vec_env,
            verbose=1,
            tensorboard_log=f"./sac_{args.task}_tb/",
            seed=1,
        )
    else:
        model = SAC(
            "MlpPolicy",
            env=vec_env,
            learning_rate=5e-4,
            learning_starts=3300,
            batch_size=128,
            train_freq=2500,
            gradient_steps=1000,
            target_update_interval=5,
            policy_kwargs={"net_arch": [256, 256]},
            verbose=1,
            tensorboard_log=f"./sac_{args.task}_tb/",
            seed=1,
        )

    callback = RewardPrinter()
    print(f"Training SAC on {args.task} environment for {args.timesteps} timesteps...")
    print(f"Reward shaping: {args.reward_shaping}")

    model.learn(total_timesteps=args.timesteps, callback=callback)
    shaping = "dense" if args.reward_shaping else "sparse"
    if args.reward_shaping:
        model.save(f'SAC_{args.task}_{shaping}_{args.timesteps}')
    else:
        model.save(f'SAC_{args.task}_{shaping}_{args.timesteps}')

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