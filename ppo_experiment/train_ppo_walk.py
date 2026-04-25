import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

from biped_env import BipedalBulletEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STAND_MODEL_PATH = os.path.join(BASE_DIR, "models_ppo_stand", "ppo_stand_final.zip")

LOG_DIR = os.path.join(BASE_DIR, "logs_ppo_walk")
MODEL_DIR = os.path.join(BASE_DIR, "models_ppo_walk")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def make_env(render=False):
    env = BipedalBulletEnv(
        task="walk",
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

    if not os.path.exists(STAND_MODEL_PATH):
        raise FileNotFoundError(
            f"Standing model not found at: {STAND_MODEL_PATH}\n"
            "Train standing first with train_ppo_stand.py"
        )

    model = PPO.load(STAND_MODEL_PATH, env=train_env, device="cpu")

    # Optional: slightly more exploration during walking fine-tune
    model.ent_coef = 0.001

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=500_000,
        callback=eval_callback,
        log_interval=10,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    final_path = os.path.join(MODEL_DIR, "ppo_walk_final")
    if os.path.exists(final_path + ".zip"):
        os.remove(final_path + ".zip")
    model.save(final_path)

    train_env.close()
    eval_env.close()