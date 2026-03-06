import time
import pybullet as p
import pybullet_data

# Connect to GUI
cid = p.connect(p.GUI)
print("connect id:", cid)

# Tell PyBullet where default models are
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Reset and setup world
p.resetSimulation()
p.setGravity(0, 0, -9.81)
p.setTimeStep(1.0 / 240.0)

# Load ground
p.loadURDF("plane.urdf")

# Load humanoid (biped)
humanoid = p.loadURDF("humanoid/humanoid.urdf", [0, 0, 1.2])
print("humanoid id:", humanoid)

# Position camera
p.resetDebugVisualizerCamera(
    cameraDistance=3,
    cameraYaw=40,
    cameraPitch=-25,
    cameraTargetPosition=[0, 0, 1]
)

# Keep simulation running
while p.isConnected():
    p.stepSimulation()
    time.sleep(1.0 / 240.0)
