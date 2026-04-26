import os
import numpy as np
import random
import torch
from stable_baselines3 import TD3

from biped_env import BipedalStandBulletEnv
import pybullet as p



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
MODEL_PATH = os.path.join(BASE_DIR, "models_td3_stand", "best_model")

TIME_STEP = 1.0 / 240.0
FRAME_SKIP = 4
# MAX_EPISODE_STEPS = 1000
MAX_EPISODE_STEPS = 2000
RENDER = False
SEED = 101

DISTANCE = 3.6
YAW = 30
PITCH = -20
Z_OFFSET = 0.4

def update_follow_camera(env, distance=3.0, yaw=50, pitch=-20, z_offset=0.4):
    robot_pos, _ = p.getBasePositionAndOrientation(env.robot_id)

    target = [
        robot_pos[0],
        robot_pos[1],
        robot_pos[2] + z_offset,
    ]

    p.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=target,
    )

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = BipedalStandBulletEnv(
        render=True,
        time_step=TIME_STEP,
        frame_skip=FRAME_SKIP,
        max_episode_steps=MAX_EPISODE_STEPS,
    )

    model = TD3.load(MODEL_PATH)

    obs, info = env.reset()
    episode_reward = 0.0
    episode_idx = 0

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            update_follow_camera(
                env,
                distance=DISTANCE,
                yaw=YAW,
                pitch=PITCH,
                z_offset=Z_OFFSET,
            )


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