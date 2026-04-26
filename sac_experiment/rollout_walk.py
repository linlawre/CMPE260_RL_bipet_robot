import os
import argparse
import random
import numpy as np
import torch
from stable_baselines3 import SAC

from biped_env import BipedalWalkBulletEnv
import pybullet as p

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models_sac_walk", "sac_stand_final.zip")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models_sac_walk", "best_model.zip")

TIME_STEP = 1.0 / 240.0
FRAME_SKIP = 4
MAX_EPISODE_STEPS = 2000
SEED = 101

CURRICULUM_STEPS = 500_000
FORWARD_REWARD_WEIGHT = 1.0
LATERAL_PENALTY_WEIGHT = 0.10

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

def parse_args():
    parser = argparse.ArgumentParser(description="Roll out a trained SAC walking policy.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the saved SAC model zip file.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions during rollout.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=0,
        help="Stop after this many episodes. Use 0 for infinite rollout.",
    )
    return parser.parse_args()



def main():
    args = parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    env = BipedalWalkBulletEnv(
        render=True,
        time_step=TIME_STEP,
        frame_skip=FRAME_SKIP,
        max_episode_steps=MAX_EPISODE_STEPS,
        curriculum_steps=CURRICULUM_STEPS,
        forward_reward_weight=FORWARD_REWARD_WEIGHT,
        lateral_penalty_weight=LATERAL_PENALTY_WEIGHT,
    )

    model = SAC.load(args.model_path)

    obs, info = env.reset()
    episode_reward = 0.0
    episode_steps = 0
    episode_idx = 0

    try:
        while True:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            update_follow_camera(
                env,
                distance=DISTANCE,
                yaw=YAW,
                pitch=PITCH,
                z_offset=Z_OFFSET,
            )

            episode_reward += reward
            episode_steps += 1

            if terminated or truncated:
                print(f"\nEpisode {episode_idx} finished")
                print(f"  steps:         {episode_steps}")
                print(f"  total_reward:  {episode_reward:.3f}")
                print(f"  height:        {info.get('height', float('nan')):.3f}")
                print(f"  roll:          {info.get('roll', float('nan')):.3f}")
                print(f"  pitch:         {info.get('pitch', float('nan')):.3f}")
                print(f"  vx:            {info.get('vx', float('nan')):.3f}")
                print(f"  vy:            {info.get('vy', float('nan')):.3f}")
                print(f"  contact_sum:   {info.get('contact_sum', float('nan')):.3f}")
                print(f"  alpha:         {info.get('alpha', float('nan')):.3f}")
                print(f"  r_stand:       {info.get('r_stand', float('nan')):.3f}")
                print(f"  r_walk:        {info.get('r_walk', float('nan')):.3f}")

                episode_idx += 1
                if args.max_episodes > 0 and episode_idx >= args.max_episodes:
                    break

                obs, info = env.reset()
                episode_reward = 0.0
                episode_steps = 0

    except KeyboardInterrupt:
        print("\nRollout stopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()