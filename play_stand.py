import time
from stable_baselines3 import PPO
from humanoid_stand_env import HumanoidStandEnv

env = HumanoidStandEnv(render=True)
model = PPO.load("ppo_humanoid_stand")

obs, _ = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    time.sleep(1/240)
    if terminated or truncated:
        obs, _ = env.reset()
