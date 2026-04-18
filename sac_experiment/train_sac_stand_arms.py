import os
import numpy as np
import random
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

from biped_env import BipedalStandArmsBulletEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs_sac_stand_arms")
MODEL_DIR = os.path.join(BASE_DIR, "models_sac_stand_arms")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 1_000_000
MAX_EPISODE_STEPS = 1500
BATCH_SIZE = 512

POLICY_KWARGS = dict(net_arch=[512, 256])
EVAL_FREQ = 20_000

TIME_STEP = 1.0 / 240.0
FRAME_SKIP = 4
RENDER = False
SEED = 101


def make_env(render=False, *, time_step, frame_skip, max_episode_steps):
    env = BipedalStandArmsBulletEnv(
        render=render,
        time_step=time_step,
        frame_skip=frame_skip,
        max_episode_steps=max_episode_steps,
    )
    env = Monitor(env)
    return env


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_env = make_env(
        render=RENDER,
        time_step=TIME_STEP,
        frame_skip=FRAME_SKIP,
        max_episode_steps=MAX_EPISODE_STEPS
    )
    eval_env = make_env(
        render=RENDER,
        time_step=TIME_STEP,
        frame_skip=FRAME_SKIP,
        max_episode_steps=MAX_EPISODE_STEPS
    )

    # Check Gymnasium API compliance
    check_env(train_env, warn=True)

    n_actions = train_env.action_space.shape[-1]

    # Smaller exploration noise since actions are pose offsets around a nominal standing pose
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.05 * np.ones(n_actions),
    )

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
        action_noise=action_noise,
        policy_kwargs=POLICY_KWARGS,
        verbose=1,
        tensorboard_log=LOG_DIR,
        seed=SEED
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        log_interval=10,
        progress_bar=True,
    )

    model.save(os.path.join(MODEL_DIR, "sac_stand_arms_final"))

    train_env.close()
    eval_env.close()