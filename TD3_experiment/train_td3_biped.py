import os
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

from biped_env import BipedalStandBulletEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs_td3_stand")
MODEL_DIR = os.path.join(BASE_DIR, "models_td3_stand")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def make_env(render=False):
    env = BipedalStandBulletEnv(
        render=render,
        time_step=1.0 / 240.0,
        frame_skip=4,
        max_episode_steps=1000,
    )
    env = Monitor(env)
    return env


if __name__ == "__main__":
    train_env = make_env(render=False)
    eval_env = make_env(render=False)

    # Check Gymnasium API compliance
    check_env(train_env, warn=True)

    n_actions = train_env.action_space.shape[-1]

    # Smaller exploration noise since actions are pose offsets around a nominal standing pose
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.05 * np.ones(n_actions),
    )

    model = TD3(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=LOG_DIR,
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
        total_timesteps=1_000_000,
        callback=eval_callback,
        log_interval=10,
        progress_bar=True,
    )

    model.save(os.path.join(MODEL_DIR, "td3_stand_final"))

    train_env.close()
    eval_env.close()