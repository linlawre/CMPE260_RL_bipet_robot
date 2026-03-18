from stable_baselines3 import PPO
from humanoid_stand_env import HumanoidStandEnv

env = HumanoidStandEnv(render=False)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu",          # ✅ force CPU to avoid CUDA NaN weirdness
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    clip_range=0.2,
)

model.learn(total_timesteps=300_000)
model.save("ppo_humanoid_stand")
env.close()
