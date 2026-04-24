import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data


class BipedalBulletEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 240}

    def __init__(
        self,
        task="stand",                  # "stand" or "walk"
        render=False,
        time_step=1.0 / 240.0,
        frame_skip=4,
        max_episode_steps=1000,
    ):
        super().__init__()

        assert task in ["stand", "walk"], f"Unknown task: {task}"
        self.task = task

        self.render_mode = "human" if render else None
        self.time_step = time_step
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps

        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setTimeStep(self.time_step, physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)

        self.plane_id = None
        self.robot_id = None

        # Keep the same 4 joints as your original setup so PPO weights can transfer cleanly
        self.target_joint_names = [
            "right_hip_y",
            "right_knee",
            "left_hip_y",
            "left_knee",
        ]

        self.joint_ids = []
        self.joint_name_to_id = {}
        self.foot_link_ids = []

        # Nominal slightly bent standing pose
        self.nominal_pose = {
            "right_hip_y": -0.25,
            "right_knee": -0.75,
            "left_hip_y": -0.25,
            "left_knee": -0.75,
        }

        # Action ranges around nominal pose
        self.action_scales = {
            "right_hip_y": 0.20,
            "right_knee": 0.25,
            "left_hip_y": 0.20,
            "left_knee": 0.25,
        }

        self._build_world()

        self.num_actuated_joints = len(self.joint_ids)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actuated_joints,),
            dtype=np.float32,
        )

        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )

        self.step_count = 0

        # Reference standing height
        self.target_base_height = 1.05

        # Walking target speed (forward in +x)
        self.target_forward_speed = 0.6

    def _build_world(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.time_step, physicsClientId=self.client)

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)

        objects = p.loadMJCF("mjcf/humanoid_symmetric.xml", physicsClientId=self.client)
        self.robot_id = objects[0]

        self.joint_ids = []
        self.joint_name_to_id = {}
        self.foot_link_ids = []

        for j in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.client)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")

            if joint_name in self.target_joint_names:
                self.joint_name_to_id[joint_name] = j
                self.joint_ids.append(j)

            if "foot" in link_name.lower():
                self.foot_link_ids.append(j)

        self.joint_ids = [self.joint_name_to_id[name] for name in self.target_joint_names]

        # Disable default motors on controlled joints
        for j in self.joint_ids:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=j,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0.0,
                force=0.0,
                physicsClientId=self.client,
            )

    def _get_joint_states(self):
        states = p.getJointStates(self.robot_id, self.joint_ids, physicsClientId=self.client)
        q = np.array([s[0] for s in states], dtype=np.float32)
        qd = np.array([s[1] for s in states], dtype=np.float32)
        return q, qd

    def _get_foot_contacts(self):
        contacts = []
        for link_id in self.foot_link_ids:
            pts = p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=self.plane_id,
                linkIndexA=link_id,
                physicsClientId=self.client,
            )
            contacts.append(1.0 if len(pts) > 0 else 0.0)

        if len(contacts) == 0:
            contacts = [0.0, 0.0]

        return np.array(contacts, dtype=np.float32)

    def _get_obs(self):
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.client)
        roll, pitch, yaw = p.getEulerFromQuaternion(base_orn)

        q, qd = self._get_joint_states()
        foot_contacts = self._get_foot_contacts()

        obs = np.concatenate([
            np.array([base_pos[2]], dtype=np.float32),      # base height
            np.array([roll, pitch, yaw], dtype=np.float32),
            np.array(base_lin_vel, dtype=np.float32),
            np.array(base_ang_vel, dtype=np.float32),
            q,
            qd,
            foot_contacts,
        ]).astype(np.float32)

        return obs

    def _apply_action(self, action):
        action = np.clip(action, -1.0, 1.0)

        for idx, joint_name in enumerate(self.target_joint_names):
            jid = self.joint_ids[idx]
            target = self.nominal_pose[joint_name] + self.action_scales[joint_name] * float(action[idx])

            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=jid,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=300.0,
                positionGain=0.5,
                velocityGain=0.3,
                physicsClientId=self.client,
            )

    def _compute_stand_reward(self, action):
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.client)
        roll, pitch, yaw = p.getEulerFromQuaternion(base_orn)
        q, qd = self._get_joint_states()
        contacts = self._get_foot_contacts()

        height = base_pos[2]

        # Stronger incentive to keep the episode alive
        alive_bonus = 2.0
        survival_bonus = 0.10

        # Softer penalties than before
        tilt_penalty = 1.0 * (abs(roll) + abs(pitch))
        height_penalty = 1.5 * max(0.0, self.target_base_height - height)
        ang_vel_penalty = 0.03 * (abs(base_ang_vel[0]) + abs(base_ang_vel[1]))
        drift_penalty = 0.05 * abs(base_lin_vel[0]) + 0.05 * abs(base_lin_vel[1])

        action_penalty = 0.0005 * np.sum(np.square(action))
        joint_vel_penalty = 0.0003 * np.sum(np.square(qd))

        # Reward foot support a bit more
        contact_bonus = 0.08 * np.sum(contacts)

        reward = (
            alive_bonus
            + survival_bonus
            + contact_bonus
            - tilt_penalty
            - height_penalty
            - ang_vel_penalty
            - drift_penalty
            - action_penalty
            - joint_vel_penalty
        )

        info = {
            "task": self.task,
            "height": float(height),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "forward_vel": float(base_lin_vel[0]),
            "lateral_vel": float(base_lin_vel[1]),
            "contact_sum": float(np.sum(contacts)),
            "tilt_penalty": float(tilt_penalty),
            "height_penalty": float(height_penalty),
            "ang_vel_penalty": float(ang_vel_penalty),
            "drift_penalty": float(drift_penalty),
            "action_penalty": float(action_penalty),
            "joint_vel_penalty": float(joint_vel_penalty),
            "alive_bonus": float(alive_bonus),
            "survival_bonus": float(survival_bonus),
            "contact_bonus": float(contact_bonus),
        }

        return float(reward), info

    def _compute_walk_reward(self, action):
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.client)
        roll, pitch, yaw = p.getEulerFromQuaternion(base_orn)
        q, qd = self._get_joint_states()
        contacts = self._get_foot_contacts()

        height = base_pos[2]
        vx = base_lin_vel[0]
        vy = base_lin_vel[1]

        alive_bonus = 1.0

        # Keep the robot upright while walking
        height_penalty = 2.0 * max(0.0, self.target_base_height - height)
        tilt_penalty = 0.75 * (abs(roll) + abs(pitch))

        # Reward forward movement
        forward_reward = 1.5 * max(0.0, vx)

        # Optional speed tracking to avoid frantic lunges
        speed_tracking_penalty = 0.5 * abs(self.target_forward_speed - max(0.0, vx))

        # Discourage sideways drift and too much torso spinning
        lateral_penalty = 0.10 * abs(vy)
        yaw_penalty = 0.05 * abs(yaw)
        ang_vel_penalty = 0.02 * (abs(base_ang_vel[0]) + abs(base_ang_vel[1]))

        action_penalty = 0.001 * np.sum(np.square(action))
        joint_vel_penalty = 0.0005 * np.sum(np.square(qd))

        # Mild contact bonus; keep it small so dragging feet is not over-rewarded
        contact_bonus = 0.03 * np.sum(contacts)

        reward = (
            alive_bonus
            + forward_reward
            + contact_bonus
            - speed_tracking_penalty
            - height_penalty
            - tilt_penalty
            - lateral_penalty
            - yaw_penalty
            - ang_vel_penalty
            - action_penalty
            - joint_vel_penalty
        )

        info = {
            "task": self.task,
            "height": float(height),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "forward_vel": float(vx),
            "lateral_vel": float(vy),
            "contact_sum": float(np.sum(contacts)),
            "forward_reward": float(forward_reward),
            "speed_tracking_penalty": float(speed_tracking_penalty),
            "height_penalty": float(height_penalty),
            "tilt_penalty": float(tilt_penalty),
            "lateral_penalty": float(lateral_penalty),
            "yaw_penalty": float(yaw_penalty),
        }

        return float(reward), info

    def _compute_reward(self, action):
        if self.task == "stand":
            return self._compute_stand_reward(action)
        elif self.task == "walk":
            return self._compute_walk_reward(action)
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def _is_fallen(self):
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        roll, pitch, _ = p.getEulerFromQuaternion(base_orn)

        if base_pos[2] < 0.55:
            return True
        if abs(roll) > 1.0 or abs(pitch) > 1.0:
            return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._build_world()

        base_x = self.np_random.uniform(-0.01, 0.01)
        base_y = self.np_random.uniform(-0.01, 0.01)
        base_yaw = self.np_random.uniform(-0.03, 0.03)

        p.resetBasePositionAndOrientation(
            self.robot_id,
            [base_x, base_y, 1.12],
            p.getQuaternionFromEuler([0.0, 0.0, base_yaw]),
            physicsClientId=self.client,
        )

        for joint_name in self.target_joint_names:
            jid = self.joint_name_to_id[joint_name]
            init_q = self.nominal_pose[joint_name] + self.np_random.uniform(-0.03, 0.03)
            init_qd = self.np_random.uniform(-0.02, 0.02)

            p.resetJointState(
                bodyUniqueId=self.robot_id,
                jointIndex=jid,
                targetValue=init_q,
                targetVelocity=init_qd,
                physicsClientId=self.client,
            )

        # Let the robot settle
        zero_action = np.zeros(self.num_actuated_joints, dtype=np.float32)
        for _ in range(20):
            self._apply_action(zero_action)
            p.stepSimulation(physicsClientId=self.client)
            if self.render_mode == "human":
                time.sleep(self.time_step)

        self.step_count = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self._apply_action(action)

        for _ in range(self.frame_skip):
            p.stepSimulation(physicsClientId=self.client)
            if self.render_mode == "human":
                time.sleep(self.time_step)

        self.step_count += 1

        obs = self._get_obs()
        reward, reward_info = self._compute_reward(action)

        terminated = self._is_fallen()
        truncated = self.step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, reward_info

    def render(self):
        pass

    def close(self):
        if self.client is not None and p.isConnected(self.client):
            p.disconnect(self.client)
            self.client = None