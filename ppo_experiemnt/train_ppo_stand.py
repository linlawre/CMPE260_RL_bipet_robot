import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

from biped_env import BipedalBulletEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs_ppo_stand")
MODEL_DIR = os.path.join(BASE_DIR, "models_ppo_stand")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def make_env(render=False):
    env = BipedalBulletEnv(
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

    # model = PPO(
    #     policy="MlpPolicy",
    #     env=train_env,
    #     learning_rate=1e-4,          # slightly lower for more stable standing learning
    #     n_steps=4096,                # longer rollout helps value estimates for balance
    #     batch_size=128,              # cleaner gradient updates
    #     n_epochs=10,

    #     gamma=0.995,                 # slightly longer horizon for staying upright
    #     gae_lambda=0.95,
    #     clip_range=0.2,

    #     ent_coef=0.002,              # small exploration, better than 0.0 at start
    #     vf_coef=0.5,
    #     max_grad_norm=0.5,

    #     policy_kwargs=dict(net_arch=[256, 256]),

    #     verbose=1,
    #     tensorboard_log=LOG_DIR,
    #     device="cpu",
    # )
    
    model = PPO.load(
    os.path.join(MODEL_DIR, "best_model.zip"),
    env=train_env,
    device="cpu",
    )
    # # Increase exploration (MOST IMPORTANT for you right now)
    # model.ent_coef = 0.003

    # # Slightly stabilize updates
    # model.clip_range = lambda _: 0.15

    # # Keep learning stable (safe override)
    # model.lr_schedule = lambda _: 1e-4

    # # Optional: slightly reduce value loss weight if it's dominating
    # model.vf_coef = 0.4

    # # Optional: keep gradients under control
    # model.max_grad_norm = 0.5
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=20000,
        deterministic=True,
        render=False,
    )
    model.learn(
        total_timesteps=3_000_000,
        callback=eval_callback,
        log_interval=10,
        progress_bar=True,
    )

    save_path = os.path.join(MODEL_DIR, "ppo_stand_final")

    if os.path.exists(save_path + ".zip"):
        os.remove(save_path + ".zip")

    model.save(save_path)

    train_env.close()
    eval_env.close()