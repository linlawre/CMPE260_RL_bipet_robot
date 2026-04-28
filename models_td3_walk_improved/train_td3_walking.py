import os
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from biped_env import BipedalStandBulletEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs_td3_walk")
MODEL_DIR = os.path.join(BASE_DIR, "models_td3_walk")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

class StepCurriculumCallback(BaseCallback):
    def __init__(self, total_timesteps, update_freq=100000, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.update_freq = update_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.update_freq == 0:
            current_alpha = min(1.0, self.num_timesteps / self.total_timesteps)
            # Apply alpha to the training environment
            self.training_env.env_method("set_alpha", current_alpha)
            if self.verbose > 0:
                print(f"Step {self.num_timesteps}: Updated alpha to {current_alpha:.2f}")
        return True

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
    print("Initializing environments...")
    
    # Training environment (starts at alpha=0.0 and climbs)
    train_env = make_vec_env(lambda: make_env(render=False), n_envs=1)
    
    # Eval environment (always tests the final task at alpha=1.0)
    eval_env = make_vec_env(lambda: make_env(render=False), n_envs=1)
    eval_env.env_method("set_alpha", 1.0)

    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.05 * np.ones(n_actions),
    )

    print("Initializing fresh TD3 model...")
    model = TD3(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=1_000_000, 
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

    total_steps = 1_000_000 
    
    curriculum_cb = StepCurriculumCallback(total_timesteps=total_steps, update_freq=100_000, verbose=1)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=10_000,          
        deterministic=True,
        render=False
    )
    
    all_callbacks = CallbackList([curriculum_cb, eval_callback])

    print("Starting training...")
    model.learn(
        total_timesteps=total_steps, 
        callback=all_callbacks,
        log_interval=10,
        progress_bar=True
    )

    print("Training complete! Saving final model...")
    model.save(os.path.join(MODEL_DIR, "td3_walk_final"))