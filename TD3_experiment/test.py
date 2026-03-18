import pybullet as p
import pybullet_data
import time
import math

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.resetSimulation()
p.setGravity(0, 0, -9.81)
p.setTimeStep(1.0 / 240.0)
p.setRealTimeSimulation(0)

plane = p.loadURDF("plane.urdf")
objects = p.loadMJCF("mjcf/humanoid_symmetric.xml")
robot = objects[0]

p.resetBasePositionAndOrientation(
    robot,
    [0, 0, 1.4],
    p.getQuaternionFromEuler([0, 0, 0])
)

# Use only the most useful sagittal-plane joints first
target_joint_names = [
    "right_hip_y",
    "right_knee",
    "left_hip_y",
    "left_knee",
]

joint_name_to_id = {}
for j in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, j)
    joint_name = info[1].decode("utf-8")
    if joint_name in target_joint_names:
        joint_name_to_id[joint_name] = j

print("\nSelected joints:")
for name in target_joint_names:
    jid = joint_name_to_id[name]
    info = p.getJointInfo(robot, jid)
    print(
        f"{name:12s} -> id={jid}, "
        f"limits=({info[8]:.3f}, {info[9]:.3f}), axis={info[13]}"
    )

# Disable default motors first
for jid in joint_name_to_id.values():
    p.setJointMotorControl2(
        bodyUniqueId=robot,
        jointIndex=jid,
        controlMode=p.VELOCITY_CONTROL,
        force=0
    )

# Slightly bent initial pose
initial_pose = {
    "right_hip_y": -0.25,
    "right_knee": -0.75,
    "left_hip_y": -0.25,
    "left_knee": -0.75,
}

for name, angle in initial_pose.items():
    p.resetJointState(robot, joint_name_to_id[name], targetValue=angle, targetVelocity=0.0)

# Let it settle a tiny bit
for _ in range(60):
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

t = 0.0
step_idx = 0

while True:
    # Larger oscillation so it is visually obvious
    s = math.sin(1.5 * t)

    target_positions = {
        "right_hip_y": -0.25 + 0.35 * s,
        "left_hip_y":  -0.25 - 0.35 * s,
        "right_knee":  -0.90 - 0.30 * s,
        "left_knee":   -0.90 + 0.30 * s,
    }

    for name, target in target_positions.items():
        jid = joint_name_to_id[name]
        p.setJointMotorControl2(
            bodyUniqueId=robot,
            jointIndex=jid,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target,
            force=300.0,          # stronger than before
            positionGain=0.5,
            velocityGain=0.3,
        )

    p.stepSimulation()
    time.sleep(1.0 / 240.0)

    if step_idx % 60 == 0:
        print("\n--- debug ---")
        base_pos, _ = p.getBasePositionAndOrientation(robot)
        print(f"base z = {base_pos[2]:.3f}")

        for name in target_joint_names:
            jid = joint_name_to_id[name]
            joint_state = p.getJointState(robot, jid)
            actual_pos = joint_state[0]
            actual_vel = joint_state[1]
            print(
                f"{name:12s} target={target_positions[name]: .3f} "
                f"actual={actual_pos: .3f} vel={actual_vel: .3f}"
            )

    t += 1.0 / 240.0
    step_idx += 1