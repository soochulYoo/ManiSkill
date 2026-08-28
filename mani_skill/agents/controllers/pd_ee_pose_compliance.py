from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import numpy as np
import torch
from gymnasium import spaces

from mani_skill.agents.controllers.utils.kinematics import Kinematics
from mani_skill.utils import gym_utils, sapien_utils
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_axis_angle,
    standardize_quaternion,
)
from mani_skill.utils.structs import Link, Pose

from .base_controller import BaseController, ControllerConfig


class PDEEPoseComplianceController(BaseController):
    """A Cartesian impedance ("compliance") controller for the end-effector pose.

    Unlike the other PD end-effector controllers (e.g. :class:`PDEEPoseController`), which
    convert an end-effector target into a joint-space target via IK and let PhysX's implicit
    joint drive rigidly track it, this controller directly computes a 6D task-space PD wrench
    from the end-effector pose/velocity error and maps it to joint torques via ``qf = J^T F``.
    This makes the end-effector's stiffness/damping against external forces (e.g. contact with
    the environment) an explicit, tunable property instead of being rigidly enforced, which is
    the point of "compliance control".

    The action space is identical to :class:`PDEEPoseController` (6D delta or absolute pose in
    the ``root_translation:root_aligned_body_rotation`` frame) so it is a drop-in replacement
    action-space-wise.

    NOTE that on the GPU it is assumed the controlled robot is not a merged articulation and is
    the same across every sub-scene.
    """

    config: "PDEEPoseComplianceControllerConfig"
    _target_pose = None
    _nullspace_target_qpos = None
    root_link: Link
    sets_target_qpos = False
    sets_target_qvel = False

    def _check_gpu_sim_works(self):
        assert (
            self.config.frame == "root_translation:root_aligned_body_rotation"
        ), "currently only translation and rotation in the root frame is supported in GPU sim"

    def _initialize_joints(self):
        super()._initialize_joints()
        if self.scene.gpu_sim_enabled:
            self._check_gpu_sim_works()
        self.kinematics = Kinematics(
            self.config.urdf_path,
            self.config.ee_link,
            self.articulation,
            self.active_joint_indices,
        )
        self.ee_link = self.kinematics.end_link

        if self.config.root_link_name is not None:
            root_link = sapien_utils.get_obj_by_name(
                self.articulation.get_links(), self.config.root_link_name
            )
            assert root_link is not None and isinstance(
                root_link, Link
            ), f"Root link {self.config.root_link_name} matches more than one link or was not found"
            self.root_link = root_link
        else:
            self.root_link = self.articulation.root

        n = len(self.joints)
        self._pos_stiffness = torch.tensor(
            np.broadcast_to(self.config.pos_stiffness, 3),
            device=self.device,
            dtype=torch.float32,
        )
        self._pos_damping = torch.tensor(
            np.broadcast_to(self.config.pos_damping, 3),
            device=self.device,
            dtype=torch.float32,
        )
        self._rot_stiffness = torch.tensor(
            np.broadcast_to(self.config.rot_stiffness, 3),
            device=self.device,
            dtype=torch.float32,
        )
        self._rot_damping = torch.tensor(
            np.broadcast_to(self.config.rot_damping, 3),
            device=self.device,
            dtype=torch.float32,
        )
        self._joint_damping = torch.tensor(
            np.broadcast_to(self.config.joint_damping, n),
            device=self.device,
            dtype=torch.float32,
        )
        self._force_limit = torch.tensor(
            np.broadcast_to(self.config.force_limit, n),
            device=self.device,
            dtype=torch.float32,
        )

    def _initialize_action_space(self):
        low = np.float32(
            np.hstack(
                [
                    np.broadcast_to(self.config.pos_lower, 3),
                    np.broadcast_to(self.config.rot_lower, 3),
                ]
            )
        )
        high = np.float32(
            np.hstack(
                [
                    np.broadcast_to(self.config.pos_upper, 3),
                    np.broadcast_to(self.config.rot_upper, 3),
                ]
            )
        )
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    def _clip_and_scale_action(self, action):
        # NOTE(xiqiang): rotation should be clipped by norm.
        pos_action = gym_utils.clip_and_scale_action(
            action[:, :3], self.action_space_low[:3], self.action_space_high[:3]
        )
        rot_action = action[:, 3:].clone()
        rot_norm = torch.linalg.norm(rot_action, axis=1)
        rot_action[rot_norm > 1] = torch.mul(rot_action, 1 / rot_norm[:, None])[
            rot_norm > 1
        ]
        rot_action = rot_action * self.config.rot_lower
        return torch.hstack([pos_action, rot_action])

    def set_drive_property(self):
        # torques are applied directly via qf every substep, so the underlying PhysX implicit
        # joint drive is disabled (stiffness/damping=0) to avoid it fighting the explicit torque.
        n = len(self.joints)
        force_limit = np.broadcast_to(self.config.force_limit, n)
        friction = np.broadcast_to(self.config.friction, n)
        for i, joint in enumerate(self.joints):
            joint.set_drive_properties(0, 0, force_limit=force_limit[i], mode="force")
            joint.set_friction(friction[i])

    @property
    def ee_pose(self):
        return self.ee_link.pose

    @property
    def ee_pose_at_base(self):
        to_base = self.root_link.pose.inv()
        return to_base * self.ee_pose

    def reset(self):
        super().reset()
        mask = self.scene._reset_mask
        if self._target_pose is None:
            self._target_pose = self.ee_pose_at_base
            self._nullspace_target_qpos = self.qpos.clone()
        else:
            assert self._target_pose is not None and self._nullspace_target_qpos is not None
            # TODO (stao): this is a strange way to mask setting individual batched pose parts
            self._target_pose.raw_pose[mask] = self.ee_pose_at_base.raw_pose[mask]
            self._nullspace_target_qpos[mask] = self.qpos[mask].clone()

    def compute_target_pose(self, prev_ee_pose_at_base: Pose, action: torch.Tensor) -> Pose:
        if self.config.use_delta:
            delta_pos, delta_rot = action[:, 0:3], action[:, 3:6]
            delta_quat = matrix_to_quaternion(euler_angles_to_matrix(delta_rot, "XYZ"))
            q = quaternion_multiply(delta_quat, prev_ee_pose_at_base.q)
            p = prev_ee_pose_at_base.p + delta_pos
        else:
            p, target_rot = action[:, 0:3], action[:, 3:6]
            q = matrix_to_quaternion(euler_angles_to_matrix(target_rot, "XYZ"))
        return Pose.create_from_pq(p, q)

    def set_action(self, action: torch.Tensor):
        action = self._preprocess_action(action)
        self._step = 0
        assert self._target_pose is not None, "reset() must be called before set_action()"
        self._target_pose = self.compute_target_pose(self._target_pose, action)
        self._apply_compliance_torque()

    def before_simulation_step(self):
        self._step += 1
        self._apply_compliance_torque()

    def _apply_compliance_torque(self):
        assert self._target_pose is not None and self._nullspace_target_qpos is not None
        cur_pose = self.ee_pose_at_base
        root_quat = self.root_link.pose.q

        qpos_full = self.articulation.get_qpos()
        jacobian = self.kinematics.compute_jacobian(qpos_full, ee_rot_quat=cur_pose.q)

        pos_err = self._target_pose.p - cur_pose.p
        q_err = standardize_quaternion(
            quaternion_multiply(self._target_pose.q, quaternion_invert(cur_pose.q))
        )
        rot_err = quaternion_to_axis_angle(q_err)

        lin_vel = quaternion_apply(
            quaternion_invert(root_quat), self.ee_link.linear_velocity
        )
        ang_vel = quaternion_apply(
            quaternion_invert(root_quat), self.ee_link.angular_velocity
        )

        force = self._pos_stiffness * pos_err - self._pos_damping * lin_vel
        torque = self._rot_stiffness * rot_err - self._rot_damping * ang_vel
        wrench = torch.hstack([force, torque]).unsqueeze(-1)  # (B, 6, 1)

        tau = (jacobian.transpose(1, 2) @ wrench).squeeze(-1)  # (B, n_active_joints)

        if self.config.nullspace_stiffness > 0:
            # a secondary joint-space PD task (pull towards a nominal posture + damp joint
            # velocity), projected into the nullspace of J so it does not disturb the EE pose task.
            # Uses a damped pseudo-inverse (as compute_ik does) instead of torch.linalg.pinv: near a
            # kinematic singularity the true pseudo-inverse blows up, and the resulting garbage
            # nullspace projection was empirically found to destabilize the controller.
            lambd = 1e-4
            task_dim = jacobian.shape[-2]
            jjt_reg = jacobian @ jacobian.transpose(1, 2) + lambd * torch.eye(
                task_dim, device=self.device
            )
            jacobian_pinv = jacobian.transpose(1, 2) @ torch.linalg.inv(jjt_reg)  # (B, n, 6)
            nullspace_proj = torch.eye(
                jacobian.shape[-1], device=self.device
            ) - jacobian_pinv @ jacobian
            qpos_err = self._nullspace_target_qpos - self.qpos
            tau_null = (
                self.config.nullspace_stiffness * qpos_err - self._joint_damping * self.qvel
            )
            tau = tau + (nullspace_proj @ tau_null.unsqueeze(-1)).squeeze(-1)
        else:
            tau = tau - self._joint_damping * self.qvel

        tau = torch.clamp(tau, -self._force_limit, self._force_limit)

        qf = self.articulation.qf.clone()
        qf[:, self.active_joint_indices] = tau
        self.articulation.set_qf(qf)

    def get_state(self) -> dict:
        assert self._target_pose is not None, "Target pose is not set"
        return {"target_pose": self._target_pose.raw_pose}

    def set_state(self, state: dict):
        target_pose = state["target_pose"]
        self._target_pose = Pose.create_from_pq(target_pose[:, :3], target_pose[:, 3:])

    def __repr__(self):
        return f"{self.__class__.__name__}(dof={self.single_action_space.shape[0]}, active_joints={len(self.joints)}, end_link={self.config.ee_link}, joints=({', '.join([x.name for x in self.joints])}))"


@dataclass
class PDEEPoseComplianceControllerConfig(ControllerConfig):
    pos_lower: Union[float, Sequence[float], np.ndarray]
    """Lower bound for position control. If a single float then X, Y, and Z rotations are bounded by this value. Otherwise can be three floats to specify each dimensions bounds"""
    pos_upper: Union[float, Sequence[float], np.ndarray]
    """Upper bound for position control. If a single float then X, Y, and Z rotations are bounded by this value. Otherwise can be three floats to specify each dimensions bounds"""
    rot_lower: Union[float, Sequence[float]]
    """Lower bound for rotation control (radians, applied to euler-angle delta/target rotation)."""
    rot_upper: Union[float, Sequence[float]]
    """Upper bound for rotation control (radians, applied to euler-angle delta/target rotation)."""

    ee_link: str
    """The name of the end-effector link to control."""
    urdf_path: str
    """Path to the URDF file defining the robot to control."""

    pos_stiffness: Union[float, Sequence[float], np.ndarray] = 1000.0
    """Cartesian translational stiffness (N/m). Scalar or per-axis (x, y, z)."""
    pos_damping: Union[float, Sequence[float], np.ndarray] = 100.0
    """Cartesian translational damping (N*s/m). Scalar or per-axis (x, y, z)."""
    rot_stiffness: Union[float, Sequence[float], np.ndarray] = 50.0
    """Cartesian rotational stiffness (N*m/rad). Scalar or per-axis (x, y, z)."""
    rot_damping: Union[float, Sequence[float], np.ndarray] = 5.0
    """Cartesian rotational damping (N*m*s/rad). Scalar or per-axis (x, y, z)."""

    nullspace_stiffness: float = 10.0
    """Stiffness (N*m/rad) of the secondary joint-space task pulling redundant joint motion
    towards the posture held at the last reset/set_action, projected into the nullspace of the
    end-effector Jacobian so it does not affect end-effector pose tracking. Set to 0 to disable."""
    joint_damping: Union[float, Sequence[float], np.ndarray] = 1.0
    """Joint-space damping (N*m*s/rad) applied to every controlled joint, either as part of the
    nullspace task (if nullspace_stiffness > 0) or directly (otherwise), for stability."""

    force_limit: Union[float, Sequence[float], np.ndarray] = 100.0
    """Per-joint torque limit (N*m) applied to the final computed torque, and also to the
    (disabled) underlying PhysX joint drive as a safety bound."""
    friction: Union[float, Sequence[float], np.ndarray] = 0.0

    root_link_name: Optional[str] = None
    """Optionally set different root link for root translation control (e.g. if root is different than base)"""
    frame: Literal["root_translation:root_aligned_body_rotation",] = (
        "root_translation:root_aligned_body_rotation"
    )
    use_delta: bool = True
    """Whether to use delta-action control. If true then actions indicate the delta/change in position and rotation. If
    false, actions indicate the absolute target pose (in the root frame) for the end-effector."""
    normalize_action: bool = True
    """Whether to normalize each action dimension into a range of [-1, 1]. Normally for most machine learning workflows this is recommended to be kept true."""

    controller_cls = PDEEPoseComplianceController
