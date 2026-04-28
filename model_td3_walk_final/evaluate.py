import pybullet as p
import time
from stable_baselines3 import TD3
from biped_env import BipedalStandBulletEnv

if __name__ == "__main__":
    print("Starting simulation window...")
    # 1. Create the environment WITH rendering turned ON
    env = BipedalStandBulletEnv(render=True)
    
    # Force the environment to use the full walking reward/behavior
    env.alpha = 1.0 

    print("Loading the trained model...")
    # 2. Load your finished model 
    # (Change this to "best_model.zip" if you want to test the standing one!)
    #model = TD3.load("models_td3_stand/best_model.zip", env=env)
    #model = TD3.load("models_td3_walk/best_model.zip", env=env)
    model = TD3.load("models_td3_walk/td3_walk_final.zip", env=env)
    # 3. Run a test episode
    obs, info = env.reset()
    print("Running episode. Watch the PyBullet window!")
    
    for i in range(1000):
        # deterministic=True ensures the robot takes its "best" action, not a random exploratory one
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # The environment automatically sleeps to match 240fps, 
        # so it will play out in real-time on your screen.

        if terminated or truncated:
            print(f"Episode finished after {i} steps!")
            break

    # Leave the window open for a couple of seconds before closing
    time.sleep(2)
    env.close()