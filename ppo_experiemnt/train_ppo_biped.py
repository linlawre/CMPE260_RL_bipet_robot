import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

from biped_env import BipedalStandBulletEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs_ppo_walk")
MODEL_DIR = os.path.join(BASE_DIR, "models_ppo_walk")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def make_env(render=False):
    env = BipedalStandBulletEnv(
        render=render,
        time_step=1.0 / 240.0,
        frame_skip=4,
        max_episode_steps=1000,
    )
    return Monitor(env)


if __name__ == "__main__":
    train_env = make_env(render=False)
    eval_env = make_env(render=False)

    check_env(train_env, warn=True)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,

        # ✅ CORRECT PPO PARAMS
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,

        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,

        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,

        policy_kwargs=dict(net_arch=[256, 256]),

        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cpu",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=20000,   # ~5 min test
        callback=eval_callback,
        log_interval=10,
        progress_bar=True,
    )

    save_path = os.path.join(MODEL_DIR, "ppo_walk_final")

    if os.path.exists(save_path + ".zip"):
        os.remove(save_path + ".zip")

    model.save(save_path)

    train_env.close()
    eval_env.close()