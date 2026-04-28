import pybullet as p
import pybullet_data

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot_id = p.loadMJCF("mjcf/humanoid_symmetric.xml")[0]

print("--- AVAILABLE JOINTS ---")
for j in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, j)
    joint_name = info[1].decode("utf-8")
    print(joint_name)