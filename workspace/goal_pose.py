import furniture_bench
import gym
import torch
import numpy as np
import furniture_bench.controllers.control_utils as C
from furniture_bench.config import ROBOT_HEIGHT, config
from furniture_bench.utils.pose import *
def rot_mat(angles, hom: bool = False):
    """Given @angles (x, y, z), compute rotation matrix
    Args:
        angles: (x, y, z) rotation angles.
        hom: whether to return a homogeneous matrix.
    """
    x, y, z = angles
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx
    if hom:
        M = np.zeros((4, 4), dtype=np.float32)
        M[:3, :3] = R
        M[3, 3] = 1.0
        return M
    return R


def rot_mat_tensor(x, y, z, device):
    return torch.tensor(rot_mat([x, y, z], hom=True), device=device).float()

def rel_rot_mat(s, t):
    s_inv = torch.linalg.inv(s)
    return t @ s_inv

def find_leg_pose_x_look_front(leg_pose):
    best_leg_pose = leg_pose.clone()
    tmp_leg_pose = leg_pose
    rot = rot_mat_tensor(0, -np.pi / 2, 0, env.device)
    for i in range(3):
        tmp_leg_pose = tmp_leg_pose @ rot
        if best_leg_pose[0, 0] < tmp_leg_pose[0, 0]:
            best_leg_pose = tmp_leg_pose
    return best_leg_pose

def satisfy(
    current,
    target,
    pos_error_threshold=None,
    ori_error_threshold=None,
    spend_time=0,
    max_len=25,
) -> bool:
    default_pos_error_threshold = 0.01
    default_ori_error_threshold = 0.2

    if pos_error_threshold is None:
        pos_error_threshold = default_pos_error_threshold
    if ori_error_threshold is None:
        ori_error_threshold = default_ori_error_threshold

    if ((current[:3, 3] - target[:3, 3]).abs().sum() < pos_error_threshold) and (
        (target[:3, :3] - current[:3, :3]).abs().sum() < ori_error_threshold
    ):
        return True
    if spend_time >= max_len:
        return True
    return False

def gripper_less(gripper_width, target_width, spend_time, cnt_max=10):
    if gripper_width <= target_width:
        return True
    if spend_time >= cnt_max:
        return True
    return False

env = gym.make(
    "test-v0",
    furniture='one_leg',
    num_envs=1,
    # record=True,
    resize_img=False,
)
env.reset()

# All the coordinate are in world frame.
def reach_leg_floor_xy(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    leg_pose = env_states[1]
    target_pos_1 = leg_pose[:3, 3]
    target_quat_1 = start_quat_1
    target_pos_1[2] = start_pos_1[2]
    target_pos_1[1] += 0.01
    target_pos_1[0] += 0

    target_ee_states = [(target_pos_1, target_quat_1, gripper_1), (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(0.02,None),(None,None)]

    return target_ee_states, thresholds, False

def reach_leg_ori(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]

    margin = rot_mat_tensor(0, -np.pi / 5, 0, env.device)
    rot = rot_mat_tensor(np.pi, np.pi/2 , 0, env.device)
    target_ori_1 = (margin @ rot)[:3, :3]
    target_quat_1 = C.mat2quat(target_ori_1)

    target_ee_states = [(start_pos_1, target_quat_1, gripper_1), (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(0.015,None),(None,None)]
    return target_ee_states, thresholds, False

def reach_leg_floor_z(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    leg_pose = env_states[1]

    target_pos_1 = start_pos_1
    target_pos_1[2] = leg_pose[2, 3]
    target_quat_1 = start_quat_1

    target_ee_states = [(target_pos_1, target_quat_1, gripper_1), (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(0.007,None),(None,None)]
    return target_ee_states, thresholds, False

def close_gripper(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    gripper_action = 1
    gripper = torch.tensor([gripper_action], device=env.device)
    target_ee_states = [(start_pos_1, start_quat_1, gripper), (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None,None),(None,None)]
    return target_ee_states, thresholds, True

def lift_up(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]

    target_pos_1 = start_pos_1 + torch.tensor(
        [0, 0, 0.15], device=env.device
    )
    target_ee_states = [(target_pos_1, start_quat_1, gripper_1), (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(0.02,0.3),(None,None)]
    return target_ee_states, thresholds, False

def move_center(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]

    target_pos_1 = torch.tensor([0.1, 0.2, 0.6], device=env.device)
    target_ee_states = [(target_pos_1, start_quat_1, gripper_1), (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(0.02,0.3),(None,None)]
    return target_ee_states, thresholds, False

def match_leg_ori(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]

    margin = rot_mat_tensor(0, -np.pi / 5, 0, env.device)
    target_ori = (margin @ rot_mat_tensor(np.pi, 0, 0, env.device))[:3, :3]
    target_pos_1 = torch.tensor([0.1, 0.2, 0.55], device=env.device)
    target_quat_1 = C.mat2quat(target_ori)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1), (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(0.02,0.3),(None,None)]
    return target_ee_states, thresholds, False

def reach_table_top_xy(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    

    default_assembled_pose = config["furniture"]["square_table"]["square_table_leg4"]["default_assembled_pose"]
    table_pose, leg_pose = env_states[0], env_states[1]
    scaled_default_assembled_pose = default_assembled_pose.clone()
    scaled_default_assembled_pose[:3, 3] *= env.furniture_scale_factor
    # leg_pose = find_leg_pose_x_look_front(leg_pose)
    table_hole_pose = (
        table_pose
        @ torch.tensor(
            get_mat(scaled_default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
            device=env.device,
        )
    )
    # table_hole_pose_robot[0,3] +=0.005
    # table_hole_pose_robot[1,3] -=0.005
    target_leg_pose = torch.tensor(
        [
            [1.0, 0.0, 0.0, table_hole_pose[0, 3]],
            [0.0, 0.0, -1.0, table_hole_pose[1, 3]],
            [0.0, 1.0, 0.0, table_hole_pose[2, 3] + 0.2],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=env.device,
    )
    
    rel = rel_rot_mat(leg_pose, target_leg_pose)
    start_pose_1 = C.to_homogeneous(start_pos_1, C.quat2mat(start_quat_1))
    target_1 = rel @ start_pose_1
    target_pos_1 = target_1[:3, 3]
    target_quat_1 = C.mat2quat(target_1[:3, :3])
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1), (start_pos_2, start_quat_2, gripper_2)]
    print("start_pos_1: ", start_pos_1)
    print("table_hole_pose: ", table_hole_pose[2, 3] + 0.2)
    print("target_pos_1: ", target_pos_1)
    thresholds = [(0.005,0.3),(None,None)]
    return target_ee_states, thresholds, False

def reach_table_top_z(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    table_pose, leg_pose = env_states[0], env_states[1]
    default_assembled_pose = config["furniture"]["square_table"]["square_table_leg4"]["default_assembled_pose"]
    scaled_default_assembled_pose = default_assembled_pose.clone()
    scaled_default_assembled_pose[:3, 3] *= env.furniture_scale_factor
    # leg_pose = find_leg_pose_x_look_front(leg_pose)
    table_hole_pose = (
        table_pose
        @ torch.tensor(
            get_mat(scaled_default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
            device=env.device,
        )
    )
    # table_hole_pose_robot[0,3] +=0.005
    # table_hole_pose_robot[1,3] -=0.005
    target_leg_pose = torch.tensor(
        [
            [1.0, 0.0, 0.0, table_hole_pose[0, 3]],
            [0.0, 0.0, -1.0, table_hole_pose[1, 3]],
            [0.0, 1.0, 0.0, table_hole_pose[2, 3] + 0.03],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=env.device,
    )
    rel = rel_rot_mat(leg_pose, target_leg_pose)
    start_pose_1 = C.to_homogeneous(start_pos_1, C.quat2mat(start_quat_1))
    target_1 = rel @ start_pose_1
    target_pos_1 = target_1[:3, 3]
    target_quat_1 = C.mat2quat(target_1[:3, :3])
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1), (start_pos_2, start_quat_2, gripper_2)]   
    thresholds = [(0.007,0.15),(None,None)]
    print("start_pos_1: ", start_pos_1)
    print("table_hole_pose: ", table_hole_pose[2, 3] + 0.2)
    print("target_pos_1: ", target_pos_1)
    return target_ee_states, thresholds, False

def insert_table_wait(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    table_pose, leg_pose = env_states[0], env_states[1]
    default_assembled_pose = config["furniture"]["square_table"]["square_table_leg4"]["default_assembled_pose"]
    scaled_default_assembled_pose = default_assembled_pose.clone()
    scaled_default_assembled_pose[:3, 3] *= env.furniture_scale_factor

    # leg_pose = find_leg_pose_x_look_front(leg_pose)
    table_hole_pose = (
        table_pose
        @ torch.tensor(
            get_mat(scaled_default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]),
            device=env.device,
        )
    )
    # table_hole_pose_robot[0,3] +=0.005
    # table_hole_pose_robot[1,3] -=0.005
    target_leg_pose = torch.tensor(
        [
            [1.0, 0.0, 0.0, table_hole_pose[0, 3]],
            [0.0, 0.0, -1.0, table_hole_pose[1, 3]],
            [0.0, 1.0, 0.0, table_hole_pose[2, 3] + 0.016],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=env.device,
    )
    rel = rel_rot_mat(leg_pose, target_leg_pose)
    start_pose_1 = C.to_homogeneous(start_pos_1, C.quat2mat(start_quat_1))
    target_1 = rel @ start_pose_1
    target_pos_1 = target_1[:3, 3]
    target_quat_1 = C.mat2quat(target_1[:3, :3])
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1), (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(0,0),(None,None)]
    print("start_pos_1: ", start_pos_1)
    print("table_hole_pose: ", table_hole_pose[2, 3] + 0.2)
    print("target_pos_1: ", target_pos_1)
    return target_ee_states, thresholds, False

def release_gripper(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    gripper_action = -1
    gripper = torch.tensor([gripper_action], device=env.device)
    target_ee_states = [(start_pos_1, start_quat_1, gripper), (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None,None),(None,None)]
    return target_ee_states, thresholds, True

def pre_grasp_xy(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    leg_pose = env_states[1]
    target_ori_1 = rot_mat_tensor(np.pi, 0, 0, env.device)[:3, :3]
    target_pos_1 = leg_pose[:3,3]
    target_pos_1[2] += 0.1
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1), (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None,None),(None,None)]
    return target_ee_states, thresholds, False

def pre_grasp(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    leg_pose = env_states[1]
    target_ori_1 = rot_mat_tensor(np.pi, 0, 0, env.device)[:3, :3]
    target_pos_1 = leg_pose[:3,3]
    target_pos_1[2] += 0.02
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1), (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None,None),(None,None)]
    return target_ee_states, thresholds, False

def screw(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    target_ori_1 = rot_mat_tensor(np.pi, 0, -np.pi / 2 - np.pi / 36, env.device)[
        :3, :3
    ]
    target_pos_1 = (start_pos_1)[:3]
    target_pos_1[2] -= 0.005
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1), (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None, 0.3),(None,None)]
    return target_ee_states, thresholds, False

def hold_table_xy(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    table_pose = env_states[0]

    pos = table_pose[:4, 3]
    target_pos_2 = (pos)[:3]
    target_ori_2 = rot_mat_tensor(np.pi, 0, 0, env.device)[:3, :3]
    target_quat_2 = C.mat2quat(target_ori_2)
    target_pos_2[0] -= 0.15
    target_pos_2[1] -= 0.09
    target_pos_2[2] += 0.1

    target_ee_states = [(start_pos_1, start_quat_1, gripper_1), (target_pos_2, target_quat_2, gripper_2)]
    thresholds = [(0.02,None),(None,None)]
    return target_ee_states, thresholds, False

def hold_table_z(env, start_ee_states, env_states):
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    table_pose = env_states[0]

    target_pos_2 = start_pos_2
    pos = table_pose[:4, 3]
    target_pos_2[2] = pos[2]+0.01
    target_ee_states = [(start_pos_1, start_quat_1, gripper_1), (target_pos_2, start_quat_2, gripper_2)]
    thresholds = [(0.02,None),(None,None)]
    return target_ee_states, thresholds, True

def get_action(start_pos, start_quat, target_pos, target_quat, gripper):
    delta_pos = target_pos - start_pos

    # Scale translational action.
    delta_pos_sign = delta_pos.sign()
    delta_pos = torch.abs(delta_pos) * 2
    for i in range(3):
        if delta_pos[i] > 0.03:
            delta_pos[i] = 0.03 + (delta_pos[i] - 0.03) * np.random.normal(1.5, 0.1)
    delta_pos = delta_pos * delta_pos_sign

    # Clamp too large action.
    max_delta_pos = 0.11 + 0.01 * torch.rand(3, device=env.device)
    max_delta_pos[2] -= 0.04
    delta_pos = torch.clamp(delta_pos, min=-max_delta_pos, max=max_delta_pos)

    delta_quat = C.quat_mul(C.quat_conjugate(start_quat), target_quat)

    gripper = torch.tensor([gripper], device=env.device)
    action = torch.concat([delta_pos, delta_quat, gripper]).unsqueeze(0)
    return action

def reach_target(target_ee_states, thresholds, is_gripper):
    target_pos_1, target_quat_1, gripper_1 = target_ee_states[0]
    target_pos_2, target_quat_2, gripper_2 = target_ee_states[1]
    pos_err_1, ori_err_1 = thresholds[0]
    pos_err_2, ori_err_2 = thresholds[1]   
    spend_time = 0
    while True:
        ee_pos_1, ee_quat_1, ee_pos_2, ee_quat_2 = env.get_ee_pose_world()
        ee_pos_1, ee_quat_1 = ee_pos_1.squeeze(), ee_quat_1.squeeze()
        ee_pos_2, ee_quat_2 = ee_pos_2.squeeze(), ee_quat_2.squeeze()
        action_1 = get_action(ee_pos_1, ee_quat_1, target_pos_1, target_quat_1, gripper_1)
        action_2 = get_action(ee_pos_2, ee_quat_2, target_pos_2, target_quat_2, gripper_2)
        action = torch.cat((action_1, action_2), dim=1)

        ee_pose_1 = C.to_homogeneous(ee_pos_1, C.quat2mat(ee_quat_1))
        ee_pose_2 = C.to_homogeneous(ee_pos_2, C.quat2mat(ee_quat_2))
        target_pose_1 = C.to_homogeneous(target_pos_1, C.quat2mat(target_quat_1))
        target_pose_2 = C.to_homogeneous(target_pos_2, C.quat2mat(target_quat_2))


        # zhp: need to change
        # gripper_width = env.dof_pos[:, 7:8] + env.dof_pos[:, 8:9]
        # half_width = 0.015
        gripper_width = 1
        half_width = 0

        if is_gripper:
            if gripper_less(gripper_width, 2 * half_width + 0.001, spend_time=spend_time):
                return
        else:
            if satisfy(ee_pose_1, target_pose_1, pos_err_1, ori_err_1, spend_time=spend_time) and satisfy(ee_pose_2, target_pose_2, pos_err_2, ori_err_2, spend_time=spend_time):
                return
        env.step(action)
        spend_time += 1



def act_phase(env, phase, last_target_ee_states=None):
    rb_states = env.rb_states
    part_idxs = env.part_idxs

    leg_pose = C.to_homogeneous(
            rb_states[part_idxs["square_table_leg4"]][0][:3],
            C.quat2mat(rb_states[part_idxs["square_table_leg4"]][0][3:7]),
        )

    table_pose = C.to_homogeneous(
        rb_states[part_idxs["square_table_top"]][0][:3],
        C.quat2mat(rb_states[part_idxs["square_table_top"]][0][3:7]),
    )

    # print(leg_pose)
    # input()

    func_map = {
        "reach_leg_floor_xy": reach_leg_floor_xy,
        "reach_leg_ori": reach_leg_ori,
        "reach_leg_floor_z": reach_leg_floor_z,
        "close_gripper": close_gripper,
        "lift_up": lift_up,
        "move_center": move_center,
        "match_leg_ori": match_leg_ori,
        "reach_table_top_xy": reach_table_top_xy,
        "reach_table_top_z": reach_table_top_z,
        "insert_table_wait": insert_table_wait,
        "release_gripper": release_gripper,
        "pre_grasp": pre_grasp,
        "screw": screw,
        "hold_table_xy": hold_table_xy,
        "hold_table_z": hold_table_z,
        "pre_grasp_xy": pre_grasp_xy,
    }

    if last_target_ee_states is not None:
        ee_pos_1, ee_quat_1, gripper_1 = last_target_ee_states[0]
        ee_pos_2, ee_quat_2, gripper_2 = last_target_ee_states[1]
    else:
        ee_pos_1, ee_quat_1, ee_pos_2, ee_quat_2 = env.get_ee_pose_world()
        gripper_1, gripper_2 = env.last_grasp_1, env.last_grasp_2

    ee_pos_1, ee_quat_1 = ee_pos_1.squeeze(), ee_quat_1.squeeze()
    ee_pos_2, ee_quat_2 = ee_pos_2.squeeze(), ee_quat_2.squeeze()
    gripper_1, gripper_2 = gripper_1.squeeze(), gripper_2.squeeze()


    start_ee_states = [(ee_pos_1, ee_quat_1, gripper_1), (ee_pos_2, ee_quat_2, gripper_2)]
    env_states = [table_pose, leg_pose]

    target_ee_states, thresholds, is_gripper = func_map[phase](env, start_ee_states, env_states)
    print("Start phase: ", phase)
    reach_target(target_ee_states, thresholds, is_gripper)
    print("End phase: ", phase)
    return target_ee_states

def wait():
    for i in range(500):
        action = torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=env.device).unsqueeze(0)
        new_action = torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=env.device).unsqueeze(0)
        action = torch.cat((action, new_action), dim=1)

        env.step(action)

def act(env):
    # target_ee_states = act_phase(env, "hold_table_xy")
    # target_ee_states = act_phase(env, "hold_table_z", target_ee_states)

    # target_ee_states = act_phase(env, "reach_leg_floor_xy", target_ee_states)
    # target_ee_states = act_phase(env, "reach_leg_ori", target_ee_states)
    # target_ee_states = act_phase(env, "reach_leg_floor_z", target_ee_states)
    # target_ee_states = act_phase(env, "close_gripper", target_ee_states)
    # target_ee_states = act_phase(env, "lift_up", target_ee_states)
    # target_ee_states = act_phase(env, "move_center", target_ee_states)
    # target_ee_states = act_phase(env, "match_leg_ori", target_ee_states)
    # target_ee_states = act_phase(env, "reach_table_top_xy", target_ee_states)
    # target_ee_states = act_phase(env, "reach_table_top_z", target_ee_states)
    # target_ee_states = act_phase(env, "insert_table_wait", target_ee_states)
    # target_ee_states = act_phase(env, "screw", target_ee_states)
    # target_ee_states = None
    # target_ee_states = act_phase(env, "pre_grasp_xy", target_ee_states)
    # for i in range(10):
    #     target_ee_states = act_phase(env, "release_gripper", target_ee_states)
    #     target_ee_states = act_phase(env, "pre_grasp", target_ee_states)
    #     target_ee_states = act_phase(env, "close_gripper", target_ee_states)
    #     target_ee_states = act_phase(env, "screw", target_ee_states)
    wait()


    print("Done!")

if __name__ == "__main__":
    act(env)
