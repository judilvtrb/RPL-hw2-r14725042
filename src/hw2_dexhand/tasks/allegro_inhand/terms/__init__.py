"""Manager terms for the Allegro in-hand tasks, referenced by dotted path from YAML."""

from hw2_dexhand.tasks.allegro_inhand.terms.actions import (
    IntegratedJointPositionAction,
    IntegratedJointPositionActionCfg,
)
from hw2_dexhand.tasks.allegro_inhand.terms.commands import (
    GoalOrientationCommand,
    GoalOrientationCommandCfg,
)
from hw2_dexhand.tasks.allegro_inhand.terms.events import (
    load_grasp_cache,
    reset_from_grasp_cache,
)
from hw2_dexhand.tasks.allegro_inhand.terms.grasp import (
    fingertip_pos_w,
    hand_joint_pos_raw,
    reset_joint_noise,
)
from hw2_dexhand.tasks.allegro_inhand.terms.multi_object import (
    active_object_dropped,
    active_object_ids,
    active_object_pos_w,
    active_object_quat_w,
    place_active_object,
    reset_from_grasp_caches,
)
from hw2_dexhand.tasks.allegro_inhand.terms.observations import (
    goal_relative_quat,
    hand_joint_pos_normalized,
    joint_position_target,
    object_ang_vel_w,
    object_lin_vel_w,
    object_pos_w,
    object_quat_canonical_w,
    object_quat_w,
)
from hw2_dexhand.tasks.allegro_inhand.terms.randomization import (
    HAND_GAIN_PRIV_DIM,
    PRIV_DIM,
    hand_gains_priv,
    object_physics_priv,
    randomize_hand_gains,
    randomize_object_physics,
)
from hw2_dexhand.tasks.allegro_inhand.terms.rewards import (
    goal_orientation_reward,
    goal_success,
    hand_pd_torque_l2,
    hand_pd_work_sq,
    hand_pose_diff_from_reset,
    object_angvel_about_axis,
    object_lin_vel_l1,
)
from hw2_dexhand.tasks.allegro_inhand.terms.terminations import (
    goal_count_reached,
    object_dropped,
)

__all__ = [
    "HAND_GAIN_PRIV_DIM",
    "PRIV_DIM",
    "hand_gains_priv",
    "object_physics_priv",
    "randomize_hand_gains",
    "randomize_object_physics",
    "GoalOrientationCommand",
    "GoalOrientationCommandCfg",
    "IntegratedJointPositionAction",
    "IntegratedJointPositionActionCfg",
    "goal_count_reached",
    "goal_orientation_reward",
    "goal_relative_quat",
    "goal_success",
    "object_quat_canonical_w",
    "active_object_dropped",
    "active_object_ids",
    "active_object_pos_w",
    "active_object_quat_w",
    "fingertip_pos_w",
    "hand_joint_pos_normalized",
    "hand_joint_pos_raw",
    "reset_joint_noise",
    "hand_pd_torque_l2",
    "hand_pd_work_sq",
    "hand_pose_diff_from_reset",
    "joint_position_target",
    "load_grasp_cache",
    "object_ang_vel_w",
    "object_angvel_about_axis",
    "object_dropped",
    "object_lin_vel_l1",
    "object_lin_vel_w",
    "object_pos_w",
    "object_quat_w",
    "place_active_object",
    "reset_from_grasp_caches",
    "reset_from_grasp_cache",
]
