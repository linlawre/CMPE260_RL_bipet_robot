import os
import time
import pybullet as p
from stable_baselines3 import TD3
from biped_env import BipedalStandBulletEnv

if __name__ == "__main__":
    print("Loading environment with rendering enabled...")
    # Initialize environment with GUI enabled
    env = BipedalStandBulletEnv(render=True)
    
    # Set to full walking behavior (alpha = 1.0)
    env.set_alpha(1.0) 

    # Define the path to your best model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models_td3_walk", "best_model")

    print(f"Loading model from: {model_path}")
    try:
        model = TD3.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        exit()

    obs, info = env.reset()
    
    # --- UPDATED CAMERA SETTINGS ---
    # CAM_DIST: 5.5 (Maintaining your zoomed out preference)
    # CAM_YAW: 315 (The opposite 45-degree isometric angle)
    # CAM_PITCH: -25 (Slightly higher angle for better visibility)
    CAM_DIST = 5.5
    CAM_YAW = 45 
    CAM_PITCH = -25

    print("Starting evaluation. Press Ctrl+C to stop.")
    try:
        while True:
            # Predict the best action
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the simulation
            obs, reward, terminated, truncated, info = env.step(action)

            # Update camera to follow the robot from the opposite isometric side
            base_pos, _ = p.getBasePositionAndOrientation(env.robot_id)
            
            p.resetDebugVisualizerCamera(
                cameraDistance=CAM_DIST, 
                cameraYaw=CAM_YAW, 
                cameraPitch=CAM_PITCH, 
                cameraTargetPosition=[base_pos[0], base_pos[1], 0.8]
            )
            
            if terminated or truncated:
                print(f"Episode finished. Resetting...")
                time.sleep(0.5) 
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")
    finally:
        env.close()