from stable_baselines3 import SAC

from biped_env import BipedalStandArmsBulletEnv, BipedalWalkArmsBulletEnv
from stable_baselines3.common.env_checker import check_env


env = BipedalWalkArmsBulletEnv(render=False)
# env = BipedalStandArmsBulletEnv(render=False)

print("Action space:", env.action_space.shape)
print("Obs space:", env.observation_space.shape)

obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print("Reward:", reward)
print("Info keys:", info.keys())
env.close()


train_env = BipedalWalkArmsBulletEnv()
# train_env = BipedalStandArmsBulletEnv()
check_env(train_env, warn=True)

model = SAC(
    "MlpPolicy",
    train_env,
    learning_starts=1000,
    buffer_size=10000,
    batch_size=512,
    verbose=1,
)

model.learn(total_timesteps=20_000)

train_env.close()


eval_env = BipedalWalkArmsBulletEnv(render=True)
# eval_env = BipedalStandArmsBulletEnv(render=True)

obs, _ = eval_env.reset()

for _ in range(5000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)

    if terminated or truncated:
        obs, _ = eval_env.reset()

eval_env.close()