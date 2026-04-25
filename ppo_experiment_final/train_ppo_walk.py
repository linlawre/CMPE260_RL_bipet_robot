import os
import random
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

from biped_env import BipedalWalkBulletEnv


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs_ppo_walk")
MODEL_DIR = os.path.join(BASE_DIR, "models_ppo_walk")

STAND_MODEL_PATH = os.path.join(BASE_DIR, "models_ppo_stand", "best_model.zip")
# Alternative:
# STAND_MODEL_PATH = os.path.join(BASE_DIR, "models_ppo_stand", "ppo_stand_final.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TIME_STEP = 1.0 / 240.0
FRAME_SKIP = 4
MAX_EPISODE_STEPS = 2000
RENDER = False
SEED = 101

CURRICULUM_STEPS = 500_000
TOTAL_TIMESTEPS = 1_000_000
DEVICE = "cpu"


def make_env(render=False):
    env = BipedalWalkBulletEnv(
        render=render,
        time_step=TIME_STEP,
        frame_skip=FRAME_SKIP,
        max_episode_steps=MAX_EPISODE_STEPS,
        curriculum_steps=CURRICULUM_STEPS,
    )
    return Monitor(env)


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_env = make_env(render=RENDER)
    eval_env = make_env(render=False)

    check_env(train_env.env, warn=True)

    if os.path.exists(STAND_MODEL_PATH):
        print(f"Loading PPO standing checkpoint: {STAND_MODEL_PATH}")
        model = PPO.load(
            STAND_MODEL_PATH,
            env=train_env,
            device=DEVICE,
            seed=SEED,
        )
        model.tensorboard_log = LOG_DIR
    else:
        print("Standing PPO checkpoint not found. Starting walking PPO from scratch.")
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            verbose=1,
            tensorboard_log=LOG_DIR,
            seed=SEED,
            device=DEVICE,
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        log_interval=10,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    model.save(os.path.join(MODEL_DIR, "ppo_walk_final"))

    train_env.close()
    eval_env.close()