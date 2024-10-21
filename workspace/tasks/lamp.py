import torch
import numpy as np
import furniture_bench.controllers.control_utils as C

from tasks.utils import *
from tasks.task_config import task_config

furniture_name = "lamp"
part1_name = task_config[furniture_name]["part_names"][0]
part2_name = task_config[furniture_name]["part_names"][1]

# Path Planning
def pre_grasp_z(env, start_ee_states, env_states):
    # start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    leg_pose = env_states[1]
    target_ori_1 = rot_mat_tensor(np.pi, 0, 0, env.device)[:3, :3]
    target_pos_1 = leg_pose[:3,3]
    target_pos_1[2] += 0.08
    target_quat_1 = C.mat2quat(target_ori_1)
    gripper_action = -1
    gripper = torch.tensor([gripper_action], device=env.device)
    target_ee_states = [(target_pos_1, target_quat_1, gripper)]#, (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None,None),(None,None)]
    return target_ee_states, thresholds, False

def pre_grasp_xy(env, start_ee_states, env_states):
    # start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    leg_pose = env_states[1]
    target_ori_1 = rot_mat_tensor(np.pi, 0, 0, env.device)[:3, :3]
    target_pos_1 = leg_pose[:3,3]
    target_pos_1[2] += 0.12
    target_quat_1 = C.mat2quat(target_ori_1)
    gripper_action = -1
    gripper = torch.tensor([gripper_action], device=env.device)
    target_ee_states = [(target_pos_1, target_quat_1, gripper)]#, (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None,None),(None,None)]
    return target_ee_states, thresholds, False

def pre_grasp(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    # start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    leg_pose = env_states[1]
    target_ori_1 = rot_mat_tensor(np.pi, 0, 0, env.device)[:3, :3]
    target_pos_1 = leg_pose[:3,3]
    target_pos_1[2] += 0.07
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]#, (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None,None),(None,None)]
    return target_ee_states, thresholds, False

def screw(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    target_ori_1 = rot_mat_tensor(np.pi, 0, -np.pi / 2 - np.pi / 36, env.device)[
        :3, :3
    ]
    target_pos_1 = (start_pos_1)[:3]
    target_pos_1[2] -= 0.005
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
    thresholds = [(None, 0.3),(None,None)]
    return target_ee_states, thresholds, False

def rev_screw(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    target_ori_1 = rot_mat_tensor(np.pi, 0, np.pi / 2 + np.pi / 36, env.device)[
        :3, :3
    ]
    target_pos_1 = (start_pos_1)[:3]
    target_pos_1[2] -= 0.005
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]
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


func_map = {
    "close_gripper": close_gripper,
    "release_gripper": release_gripper,
    "pre_grasp": pre_grasp,
    "screw": screw,
    "pre_grasp_xy": pre_grasp_xy,
    "pre_grasp_z": pre_grasp_z,
    "rev_screw": rev_screw,
}

def act_phase(env, phase, func_map, last_target_ee_states=None):
    rb_states = env.rb_states
    part_idxs = env.part_idxs

    part2_pose = C.to_homogeneous(
            rb_states[part_idxs[part2_name]][0][:3],
            C.quat2mat(rb_states[part_idxs[part2_name]][0][3:7]),
        )

    part1_pose = C.to_homogeneous(
        rb_states[part_idxs[part1_name]][0][:3],
        C.quat2mat(rb_states[part_idxs[part1_name]][0][3:7]),
    )

    if last_target_ee_states is not None:
        ee_pos_1, ee_quat_1, gripper_1 = last_target_ee_states[0]
    else:
        ee_pos_1, ee_quat_1 = env.get_ee_pose_world()
        gripper_1 = env.last_grasp_1

    ee_pos_1, ee_quat_1 = ee_pos_1.squeeze(), ee_quat_1.squeeze()
    gripper_1 = gripper_1.squeeze()


    start_ee_states = [(ee_pos_1, ee_quat_1, gripper_1)]
    env_states = [part1_pose, part2_pose]

    target_ee_states, thresholds, is_gripper = func_map[phase](env, start_ee_states, env_states)
    result = reach_target(env, target_ee_states, thresholds, is_gripper)

    return target_ee_states, result

def prepare(env, prev_target_ee_states):
    target_ee_states, result = act_phase(env, "pre_grasp_xy", func_map, prev_target_ee_states)
    return target_ee_states, result

def perform(env, prev_target_ee_states):

    target_ee_states = prev_target_ee_states
    for i in range(2):
        # zhp: Efficiency? seems like release_gripper takes a while
        target_ee_states, result = act_phase(env, "release_gripper", func_map, target_ee_states)
        target_ee_states, result = act_phase(env, "pre_grasp", func_map, target_ee_states)
        if not result:
            print("Gripper Collision at Pre Grasp")
            return False
        target_ee_states, result = act_phase(env, "close_gripper", func_map, target_ee_states)
        target_ee_states, result = act_phase(env, "screw", func_map, target_ee_states)
        
    return True

def perform_disassemble(env, prev_target_ee_states):

    target_ee_states = prev_target_ee_states
    for i in range(5):
        # zhp: Efficiency? seems like release_gripper takes a while
        target_ee_states, result = act_phase(env, "release_gripper", func_map, target_ee_states)
        target_ee_states, result = act_phase(env, "pre_grasp", func_map, target_ee_states)
        if not result:
            print("Gripper Collision at Pre Grasp")
            return False
        target_ee_states, result = act_phase(env, "close_gripper", func_map, target_ee_states)
        target_ee_states, result = act_phase(env, "rev_screw", func_map, target_ee_states)
        
    return True