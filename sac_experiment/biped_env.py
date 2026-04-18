import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data


class BipedalStandBulletEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 240}

    def __init__(
        self,
        render=False,
        time_step=1.0 / 240.0,
        frame_skip=4,
        max_episode_steps=1000,
        *args,
        **kwargs
    ):
        super().__init__()

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

        self.target_joint_names, self.nominal_pose, self.action_scales = self._build_joints()

        self.joint_ids = []
        self.joint_name_to_id = {}
        self.foot_link_ids = []

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

        # Measured standing reference height after reset pose
        self.target_base_height = 1.05
    
    def _build_joints(self):
        # Only use the 4 sagittal-plane leg joints for the standing task
        target_joint_names = [
            "right_hip_y",
            "right_knee",
            "left_hip_y",
            "left_knee",
        ]
        # Nominal slightly bent standing pose
        nominal_pose = {
            "right_hip_y": -0.25,
            "right_knee": -0.75,
            "left_hip_y": -0.25,
            "left_knee": -0.75,
        }

        # Small action ranges around nominal pose
        action_scales = {
            "right_hip_y": 0.20,
            "right_knee": 0.25,
            "left_hip_y": 0.20,
            "left_knee": 0.25,
        }
        return target_joint_names, nominal_pose, action_scales

    def _build_world(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.time_step, physicsClientId=self.client)

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)

        # Load Bullet's built-in humanoid MJCF
        objects = p.loadMJCF("mjcf/humanoid_symmetric.xml", physicsClientId=self.client)
        self.robot_id = objects[0]

        self.joint_ids = []
        self.joint_name_to_id = {}
        self.foot_link_ids = []

        # Gather joints
        for j in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.client)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")

            if joint_name in self.target_joint_names:
                self.joint_name_to_id[joint_name] = j
                self.joint_ids.append(j)

            if "foot" in link_name.lower():
                self.foot_link_ids.append(j)

        # Keep deterministic ordering matching action layout
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

        # Simpler observation for standing task
        obs = np.concatenate([
            np.array([base_pos[2]], dtype=np.float32),          # base height
            np.array([roll, pitch, yaw], dtype=np.float32),     # orientation
            np.array(base_lin_vel, dtype=np.float32),           # linear velocity
            np.array(base_ang_vel, dtype=np.float32),           # angular velocity
            q,                                                  # joint positions
            qd,                                                 # joint velocities
            foot_contacts,                                      # foot contacts
        ]).astype(np.float32)

        return obs

    def _apply_action(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Action is offset around nominal standing pose
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

    def _get_reward_state(self):
        base_pos, base_orn = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.client
        )
        base_lin_vel, base_ang_vel = p.getBaseVelocity(
            self.robot_id, physicsClientId=self.client
        )
        roll, pitch, yaw = p.getEulerFromQuaternion(base_orn)
        q, qd = self._get_joint_states()
        contacts = self._get_foot_contacts()

        return {
            "base_pos": base_pos,
            "base_orn": base_orn,
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "q": q,
            "qd": qd,
            "contacts": contacts,
            "height": base_pos[2],
            "vx": base_lin_vel[0],
            "vy": base_lin_vel[1],
        }

    def _compute_stand_reward_terms(self, state, action):
        height = state["height"]
        roll = state["roll"]
        pitch = state["pitch"]
        qd = state["qd"]
        contacts = state["contacts"]
        vx = state["vx"]
        vy = state["vy"]

        alive_bonus = 1.0
        height_penalty = 2.0 * max(0.0, self.target_base_height - height)
        tilt_penalty = 0.75 * (abs(roll) + abs(pitch))
        # action_penalty = 0.001 * np.sum(np.square(action))

        # adjust it as the arms have been added to give agent more freeddom to use them
        action_penalty = 0.0005 * np.sum(np.square(action))
        # action_penalty = 0.0006 * np.sum(np.square(action))

        joint_vel_penalty = 0.0005 * np.sum(np.square(qd))
        contact_bonus = 0.05 * np.sum(contacts)
        drift_penalty = 0.05 * abs(vx) + 0.05 * abs(vy)

        terms = {
            "alive_bonus": float(alive_bonus),
            "height_penalty": float(height_penalty),
            "tilt_penalty": float(tilt_penalty),
            "action_penalty": float(action_penalty),
            "joint_vel_penalty": float(joint_vel_penalty),
            "contact_bonus": float(contact_bonus),
            "drift_penalty": float(drift_penalty),
        }
        return terms
    
    def _sum_stand_reward_terms(self, terms):
        reward = (
            terms["alive_bonus"]
            + terms["contact_bonus"]
            - terms["height_penalty"]
            - terms["tilt_penalty"]
            - terms["action_penalty"]
            - terms["joint_vel_penalty"]
            - terms["drift_penalty"]
        )
        return float(reward)

    def _compute_reward(self, action):
        state = self._get_reward_state()
        terms = self._compute_stand_reward_terms(state, action)
        reward = self._sum_stand_reward_terms(terms)

        info = {
            "height": float(state["height"]),
            "roll": float(state["roll"]),
            "pitch": float(state["pitch"]),
            "yaw": float(state["yaw"]),
            "vx": float(state["vx"]),
            "vy": float(state["vy"]),
            "contact_sum": float(np.sum(state["contacts"])),
            **terms,
            "reward_total": float(reward),
        }
        return reward, info

    def _is_fallen(self):
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        roll, pitch, _ = p.getEulerFromQuaternion(base_orn)

        # Terminate if clearly collapsed
        if base_pos[2] < 0.55:
            return True
        if abs(roll) > 1.0 or abs(pitch) > 1.0:
            return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._build_world()

        # Small randomization around nominal standing pose
        base_x = self.np_random.uniform(-0.01, 0.01)
        base_y = self.np_random.uniform(-0.01, 0.01)
        base_yaw = self.np_random.uniform(-0.03, 0.03)

        p.resetBasePositionAndOrientation(
            self.robot_id,
            [base_x, base_y, 1.12],
            p.getQuaternionFromEuler([0.0, 0.0, base_yaw]),
            physicsClientId=self.client,
        )

        # Initialize controlled joints near nominal standing pose
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

        # Let the robot settle briefly under the position targets
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

class BipedalWalkBulletEnv(BipedalStandBulletEnv):

    def __init__(self, *args, curriculum_steps=500_000, **kwargs):
        super().__init__(*args, **kwargs)

        self.curriculum_steps = curriculum_steps
        self.global_step_count = 0  # IMPORTANT for curriculum

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.global_step_count += 1
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action):
        state = self._get_reward_state()
        terms = self._compute_stand_reward_terms(state, action)

        # Standing reward from parent logic
        r_stand = self._sum_stand_reward_terms(terms)

        # Walking modifications
        vx = state["vx"]
        vy = state["vy"]

        # Walking reward
        w_f = 1.0
        w_y = 0.1

        forward_reward = w_f * max(0.0, vx)
        lateral_penalty = w_y * abs(vy)

        # Build walking reward by reusing parent terms but replacing drift logic
        r_walk = (
            terms["alive_bonus"]
            + terms["contact_bonus"]
            + forward_reward
            - terms["height_penalty"]
            - terms["tilt_penalty"]
            - terms["action_penalty"]
            - terms["joint_vel_penalty"]
            - lateral_penalty
        )

        alpha = min(1.0, self.global_step_count / self.curriculum_steps)
        reward = (1.0 - alpha) * r_stand + alpha * r_walk

        info = {
            "height": float(state["height"]),
            "roll": float(state["roll"]),
            "pitch": float(state["pitch"]),
            "yaw": float(state["yaw"]),
            "vx": float(vx),
            "vy": float(vy),
            "contact_sum": float(np.sum(state["contacts"])),
            **terms,
            "forward_reward": float(forward_reward),
            "lateral_penalty": float(lateral_penalty),
            "r_stand": float(r_stand),
            "r_walk": float(r_walk),
            "alpha": float(alpha),
            "reward_total": float(reward),
        }
        return float(reward), info

class BipedalStandArmsBulletEnv(BipedalStandBulletEnv):
    def __init__(self, render=False, time_step=1 / 240, frame_skip=4, max_episode_steps=1000, *args, **kwargs):
        super().__init__(render, time_step, frame_skip, max_episode_steps, *args, **kwargs)
    
    def _build_joints(self):
        target_joint_names, nominal_pose, action_scales = super()._build_joints()

        # avoid mutating the parent-returned objects in place
        target_joint_names = list(target_joint_names)
        nominal_pose = dict(nominal_pose)
        action_scales = dict(action_scales)
        
        target_joint_names += [
            "right_shoulder1",
            "left_shoulder1",

            "right_shoulder2",
            "left_shoulder2",
            ]
        

        nominal_pose |= {
            "right_shoulder1": 0.10,
            "left_shoulder1": -0.10,

            "right_shoulder2": 0.0,
            "left_shoulder2": 0.0,
        }
        action_scales |= {
            "right_shoulder1": 0.10,
            "left_shoulder1": 0.10,

            "right_shoulder2": 0.05,
            "left_shoulder2": 0.05,
        }
        return target_joint_names, nominal_pose, action_scales

class BipedalWalkArmsBulletEnv(BipedalStandArmsBulletEnv):
    def __init__(
        self,
        render=False,
        time_step=1.0 / 240.0,
        frame_skip=4,
        max_episode_steps=1000,
        curriculum_steps=1_000_000,
        forward_reward_weight=0.5,
        lateral_penalty_weight=0.10,
    ):
        self.curriculum_steps = curriculum_steps
        self.forward_reward_weight = forward_reward_weight
        self.lateral_penalty_weight = lateral_penalty_weight

        self.global_step_count = 0

        super().__init__(
            render=render,
            time_step=time_step,
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.global_step_count += 1
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action):
        """
        If want different standing terms rewards override: _compute_stand_reward_terms
        """
        state = self._get_reward_state()
        terms = self._compute_stand_reward_terms(state, action)

        # Standing reward from parent logic
        r_stand = self._sum_stand_reward_terms(terms)

        # Walking modifications
        vx = state["vx"]
        vy = state["vy"]

        forward_reward = self.forward_reward_weight * max(0.0, vx)
        lateral_penalty = self.lateral_penalty_weight * abs(vy)

        # Build walking reward by reusing parent terms but replacing drift logic
        r_walk = (
            terms["alive_bonus"]
            + terms["contact_bonus"]
            + forward_reward
            - terms["height_penalty"]
            - terms["tilt_penalty"]
            - terms["action_penalty"]
            - terms["joint_vel_penalty"]
            - lateral_penalty
        )

        # Curriculum
        alpha = min(1.0, self.global_step_count / self.curriculum_steps)
        reward = (1.0 - alpha) * r_stand + alpha * r_walk

        info = {
            "height": float(state["height"]),
            "roll": float(state["roll"]),
            "pitch": float(state["pitch"]),
            "yaw": float(state["yaw"]),
            "vx": float(vx),
            "vy": float(vy),
            "contact_sum": float(np.sum(state["contacts"])),
            **terms,
            "forward_reward": float(forward_reward),
            "lateral_penalty": float(lateral_penalty),
            "r_stand": float(r_stand),
            "r_walk": float(r_walk),
            "alpha": float(alpha),
            "reward_total": float(reward),
        }
        return float(reward), info
