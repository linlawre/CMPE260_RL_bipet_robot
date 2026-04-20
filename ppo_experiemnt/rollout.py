import os
import time
from stable_baselines3 import PPO

from biped_env import BipedalBulletEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TASK = "walk"   # "stand" or "walk"

if TASK == "stand":
    MODEL_PATH = os.path.join(BASE_DIR, "models_ppo_stand", "ppo_stand_final")
else:
    MODEL_PATH = os.path.join(BASE_DIR, "models_ppo_walk", "ppo_walk_final")


def main():
    env = BipedalBulletEnv(
        task=TASK,
        render=True,
        time_step=1.0 / 240.0,
        frame_skip=4,
        max_episode_steps=1000,
    )

    model = PPO.load(MODEL_PATH)

    obs, info = env.reset()
    episode_reward = 0.0
    episode_idx = 0

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            time.sleep(1.0 / 240.0)

            if terminated or truncated:
                print(f"\nEpisode {episode_idx} finished")
                print(f"  task:         {TASK}")
                print(f"  total_reward: {episode_reward:.3f}")
                print(f"  height:       {info.get('height', float('nan')):.3f}")
                print(f"  roll:         {info.get('roll', float('nan')):.3f}")
                print(f"  pitch:        {info.get('pitch', float('nan')):.3f}")
                print(f"  forward_vel:  {info.get('forward_vel', float('nan')):.3f}")
                print(f"  contact_sum:  {info.get('contact_sum', float('nan')):.3f}")

                obs, info = env.reset()
                episode_reward = 0.0
                episode_idx += 1

    except KeyboardInterrupt:
        print("\nRollout stopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()