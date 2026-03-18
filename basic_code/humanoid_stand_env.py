import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces


class HumanoidStandEnv(gym.Env):
    def __init__(self, render: bool = False):
        super().__init__()
        self.render = render
        self.cid = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.dt = 1.0 / 240.0
        self.substeps = 8               # smoother than 4
        self.max_steps = 1500
        self.step_count = 0

        # Humanoid joints: 0=root, 1..14 are controllable joints
        self.joint_ids = list(range(1, 15))
        self.nj = len(self.joint_ids)

        # Action = target joint positions (normalized)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.nj,), dtype=np.float32
        )

        # Obs: base z, roll, pitch, base lin vel(3), base ang vel(3), joint pos(nj), joint vel(nj)
        obs_dim = 1 + 2 + 3 + 3 + self.nj + self.nj
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._reset_sim()

    def _sanitize(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(x, -10.0, 10.0).astype(np.float32)

    def _reset_sim(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)

        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("humanoid/humanoid.urdf", [0, 0, 1.2])

        if self.render:
            p.resetDebugVisualizerCamera(
                cameraDistance=3,
                cameraYaw=40,
                cameraPitch=-25,
                cameraTargetPosition=[0, 0, 1]
            )

        # Disable default motors
        for j in range(p.getNumJoints(self.robot)):
            p.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL, force=0)

        # Stable-ish starting pose (slight knee bend)
        init = {9: 0.0, 10: 0.3, 11: -0.2, 12: 0.0, 13: 0.3, 14: -0.2}
        for jid, tgt in init.items():
            p.setJointMotorControl2(
                self.robot, jid,
                p.POSITION_CONTROL,
                targetPosition=tgt,
                force=120
            )

        # Let it settle
        for _ in range(80):
            p.stepSimulation()

    def _obs(self) -> np.ndarray:
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        lin, ang = p.getBaseVelocity(self.robot)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)

        q, qd = [], []
        for jid in self.joint_ids:
            js = p.getJointState(self.robot, jid)
            q.append(js[0])
            qd.append(js[1])

        obs = np.array(
            [pos[2], roll, pitch] + list(lin) + list(ang) + q + qd,
            dtype=np.float32
        )
        return self._sanitize(obs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._reset_sim()
        return self._obs(), {}

    def step(self, action):
        self.step_count += 1

        a = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Gentler targets to avoid sim blow-ups
        targets = 0.35 * a

        # Lower force for stability
        for i, jid in enumerate(self.joint_ids):
            p.setJointMotorControl2(
                self.robot, jid,
                p.POSITION_CONTROL,
                targetPosition=float(targets[i]),
                force=80
            )

        for _ in range(self.substeps):
            p.stepSimulation()

        obs = self._obs()

        # Hard NaN safety (should not trigger now)
        if not np.isfinite(obs).all():
            return self._sanitize(obs), -10.0, True, False, {}

        z, roll, pitch = float(obs[0]), float(obs[1]), float(obs[2])

        # Reward: upright + height bonus - action penalty
        upright = float(np.exp(-3.0 * (roll * roll + pitch * pitch)))
        height_bonus = max(0.0, z - 0.9)
        act_pen = 0.01 * float(np.sum(a * a))
        reward = 1.0 * upright + 0.5 * height_bonus - act_pen

        terminated = (z < 0.7) or (abs(roll) > 1.0) or (abs(pitch) > 1.0)
        truncated = (self.step_count >= self.max_steps)

        return obs, reward, terminated, truncated, {}

    def close(self):
        if p.isConnected():
            p.disconnect()
