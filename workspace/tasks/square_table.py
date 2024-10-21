import torch
import numpy as np
import furniture_bench.controllers.control_utils as C

from tasks.utils import *

# Pose Operations
def find_leg_pose_x_look_front(env, leg_pose):
    best_leg_pose = leg_pose.clone()
    tmp_leg_pose = leg_pose
    rot = rot_mat_tensor(0, -np.pi / 2, 0, env.device)
    for i in range(3):
        tmp_leg_pose = tmp_leg_pose @ rot
        if best_leg_pose[0, 0] < tmp_leg_pose[0, 0]:
            best_leg_pose = tmp_leg_pose
    return best_leg_pose

def gripper_less(gripper_width, target_width):
    if gripper_width <= target_width:
        return True
    
    return False

# Path Planning
def pre_grasp(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    # start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    leg_pose = env_states[1]
    target_ori_1 = rot_mat_tensor(np.pi, 0, 0, env.device)[:3, :3]
    target_pos_1 = leg_pose[:3,3]
    target_pos_1[2] += 0.02
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]#, (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None,None),(None,None)]
    return target_ee_states, thresholds, False

def screw(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    # start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    target_ori_1 = rot_mat_tensor(np.pi, 0, -np.pi / 2 - np.pi / 36, env.device)[
        :3, :3
    ]
    target_pos_1 = (start_pos_1)[:3]
    target_pos_1[2] -= 0.005
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]#, (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None, 0.3),(None,None)]
    return target_ee_states, thresholds, False

def release_gripper(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    # start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    gripper_action = -1
    gripper = torch.tensor([gripper_action], device=env.device)
    target_ee_states = [(start_pos_1, start_quat_1, gripper)]#, (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None,None),(None,None)]
    return target_ee_states, thresholds, True

def close_gripper(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    # start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    gripper_action = 1
    gripper = torch.tensor([gripper_action], device=env.device)
    target_ee_states = [(start_pos_1, start_quat_1, gripper)]#, (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None,None),(None,None)]
    return target_ee_states, thresholds, True

def pre_grasp_xy(env, start_ee_states, env_states):
    # start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    leg_pose = env_states[1]
    target_ori_1 = rot_mat_tensor(np.pi, 0, 0, env.device)[:3, :3]
    target_pos_1 = leg_pose[:3,3]
    target_pos_1[2] += 0.1
    target_quat_1 = C.mat2quat(target_ori_1)
    gripper_action = -1
    gripper = torch.tensor([gripper_action], device=env.device)
    target_ee_states = [(target_pos_1, target_quat_1, gripper)]#, (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None,None),(None,None)]
    return target_ee_states, thresholds, False


func_map = {
    "close_gripper": close_gripper,
    "release_gripper": release_gripper,
    "pre_grasp": pre_grasp,
    "screw": screw,
    "pre_grasp_xy": pre_grasp_xy,
}
def prepare(env, prev_target_ee_states):
    target_ee_states, result = act_phase(env, "pre_grasp_xy", func_map, prev_target_ee_states)
    return target_ee_states, result

def perform(env, prev_target_ee_states):
    target_ee_states = prev_target_ee_states
    for i in range(2):
        target_ee_states, result = act_phase(env, "release_gripper", func_map, target_ee_states)
        target_ee_states, result = act_phase(env, "pre_grasp_xy", func_map, target_ee_states)
        if not result:
            print("Gripper Collision at Pre Grasp XY")
            return False
        target_ee_states, result = act_phase(env, "pre_grasp", func_map, target_ee_states)
        if not result:
            print("Gripper Collision at Pre Grasp")
            return False
        target_ee_states, result = act_phase(env, "close_gripper", func_map, target_ee_states)
        target_ee_states, result = act_phase(env, "screw", func_map, target_ee_states)

    return True