import os
import numpy as np
import random
import torch
from stable_baselines3 import SAC

from biped_env import BipedalStandArmsBulletEnv


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "models_sac_stand_arms", "sac_stand_arms_final")
# MODEL_PATH = os.path.join(BASE_DIR, "models_sac_stand_arms", "best_model")

TIME_STEP = 1.0 / 240.0
FRAME_SKIP = 4
# MAX_EPISODE_STEPS = 1000
MAX_EPISODE_STEPS = 1500
RENDER = False
SEED = 101

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = BipedalStandArmsBulletEnv(
        render=True,
        time_step=TIME_STEP,
        frame_skip=FRAME_SKIP,
        max_episode_steps=MAX_EPISODE_STEPS,
    )

    model = SAC.load(MODEL_PATH)

    obs, info = env.reset()
    episode_reward = 0.0
    episode_idx = 0

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                print(f"\nEpisode {episode_idx} finished")
                print(f"  total_reward: {episode_reward:.3f}")
                print(f"  height:       {info.get('height', float('nan')):.3f}")
                print(f"  roll:         {info.get('roll', float('nan')):.3f}")
                print(f"  pitch:        {info.get('pitch', float('nan')):.3f}")
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