import os
import random
import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env

from biped_env import BipedalWalkArmsBulletEnv


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs_sac_walk_arms")
MODEL_DIR = os.path.join(BASE_DIR, "models_sac_walk_arms")

# STAND_MODEL_PATH = os.path.join(BASE_DIR, "models_sac_stand_arms", "sac_stand_arms_final.zip")
STAND_MODEL_PATH = os.path.join(BASE_DIR, "models_sac_stand_arms", "best_model.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 1_000_000
MAX_EPISODE_STEPS = 1500
BATCH_SIZE = 512

POLICY_KWARGS = dict(net_arch=[512, 256])
EVAL_FREQ = 25_000


TIME_STEP = 1.0 / 240.0
FRAME_SKIP = 4
RENDER = False
SEED = 101

# Walking / curriculum settings
CURRICULUM_STEPS = 500_000
FORWARD_REWARD_WEIGHT = 1.0
LATERAL_PENALTY_WEIGHT = 0.10

DEVICE = "cuda"   # change to "cpu" if needed


def make_env(render=False, *, time_step, frame_skip, max_episode_steps):
    env = BipedalWalkArmsBulletEnv(
        render=render,
        time_step=time_step,
        frame_skip=frame_skip,
        max_episode_steps=max_episode_steps,
        curriculum_steps=CURRICULUM_STEPS,
        forward_reward_weight=FORWARD_REWARD_WEIGHT,
        lateral_penalty_weight=LATERAL_PENALTY_WEIGHT,
    )
    env = Monitor(env)
    return env


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_env = make_env(
        render=RENDER,
        time_step=TIME_STEP,
        frame_skip=FRAME_SKIP,
        max_episode_steps=MAX_EPISODE_STEPS,
    )
    eval_env = make_env(
        render=RENDER,
        time_step=TIME_STEP,
        frame_skip=FRAME_SKIP,
        max_episode_steps=MAX_EPISODE_STEPS,
    )

    # Check the new walking env once before training
    check_env(train_env.env, warn=True)

    if os.path.exists(STAND_MODEL_PATH):
        print(f"Loading standing SAC checkpoint: {STAND_MODEL_PATH}")
        model = SAC.load(
            STAND_MODEL_PATH,
            env=train_env,
            device=DEVICE,
            seed=SEED,
        )
        model.tensorboard_log = LOG_DIR
        model.tensorboard_log = LOG_DIR
    else:
        print("Standing checkpoint not found. Starting SAC from scratch.")
        print("Standing checkpoint not found. Starting SAC from scratch.")
        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            buffer_size=200_000,
            learning_starts=5_000,
            batch_size=BATCH_SIZE,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=1,
            ent_coef="auto",
            policy_kwargs=POLICY_KWARGS,
            verbose=1,
            tensorboard_log=LOG_DIR,
            seed=SEED,
            device=DEVICE,
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
    )

    # checkpoint_callback = CheckpointCallback(
    #     save_freq=50_000,
    #     save_path=MODEL_DIR,
    #     name_prefix="sac_walk_ckpt",
    # )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        # callback=[eval_callback, checkpoint_callback],
        callback=[eval_callback],
        log_interval=10,
        progress_bar=True,
        reset_num_timesteps=False,   # important when continuing from stand model
    )

    model.save(os.path.join(MODEL_DIR, "sac_walk_arms_final"))

    train_env.close()
    eval_env.close()