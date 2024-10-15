import furniture_bench
import gym
import torch
import numpy as np
import furniture_bench.controllers.control_utils as C
from furniture_bench.config import ROBOT_HEIGHT, config
from furniture_bench.utils.pose import *
import isaacgym
from isaacgym import gymapi, gymtorch

import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import trange


# Matrix Operations
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

# Pose Operations
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

# Point Cloud Operations
def depth_image_to_point_cloud_GPU(depth_image, camera_view_matrix, camera_proj_matrix, width:float, height:float, depth_bar:float=None, device:torch.device='cuda:0'):
    vinv = torch.inverse(camera_view_matrix)
    proj = camera_proj_matrix
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    centerU = width/2
    centerV = height/2

    u = torch.linspace(0, width - 1, width, device=device)
    v = torch.linspace(0, height - 1, height, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    Z = depth_image
    x_para = -Z*fu/width
    y_para = Z*fv/height
    X = (u-centerU) * x_para
    Y = (v-centerV) * y_para

    # valid = Z > -0.8
    position = torch.stack([X, Y, Z, torch.ones_like(X)], dim=-1)
    position = position.view(-1, 4)
    position = position@vinv
    points = position[:, :3]

    # normal_image = normal_image.permute(1, 2, 0).view(-1, 3)  # 将法线图像转换为 (N, 3) 形状
    # normal_image = normal_image @ vinv[:3, :3]

    x_dz = (depth_image[1:height-1, 2:width] - depth_image[1:height-1, 0:width-2])*0.5
    y_dz = (depth_image[2:height, 1:width-1] - depth_image[0:height-2, 1:width-1])*0.5
    dx = x_para
    dy = y_para
    dx = dx[1:height-1, 1:width-1]
    dy = dy[1:height-1, 1:width-1]

    normal_x = -x_dz/dx
    normal_y = -y_dz/dy
    normal_z = torch.ones((height-2, width-2), device=device)

    normal_l = torch.sqrt(normal_x*normal_x + normal_y*normal_y + normal_z*normal_z)
    normal_x = normal_x/normal_l
    normal_y = normal_y/normal_l
    normal_z = normal_z/normal_l

    normal_map = torch.stack([normal_x, normal_y, normal_z], dim=-1)
    normal_map_full = torch.zeros((height, width,3)).to(depth_image.device)
    normal_map_full[1:height-1, 1:width-1, :] = normal_map
    normals = normal_map_full.view(-1, 3)
    normals = normals @ vinv[:3, :3]

    return points, normals


def capture_pc(env, camera_handle):
    # camera_handle = env.camera_handles["front"][0]
    cam_proj = torch.tensor(env.isaac_gym.get_camera_proj_matrix(env.sim, env.envs[0], camera_handle)).to(env.device)
    cam_view = torch.tensor(env.isaac_gym.get_camera_view_matrix(env.sim, env.envs[0], camera_handle)).to(env.device)

    depth_render_type = gymapi.IMAGE_DEPTH
    depth_image = gymtorch.wrap_tensor(
                        env.isaac_gym.get_camera_image_gpu_tensor(
                            env.sim, env.envs[0], camera_handle, depth_render_type
                        )
                    )
    
    points, normals = depth_image_to_point_cloud_GPU(
        depth_image,
        cam_view,
        cam_proj,
        width=env.img_size[0],
        height=env.img_size[1],

    )

    seg_render_type = gymapi.IMAGE_SEGMENTATION
    seg_image = gymtorch.wrap_tensor(
                        env.isaac_gym.get_camera_image_gpu_tensor(
                            env.sim, env.envs[0], camera_handle, seg_render_type
                        )
                    )
    segments = seg_image.flatten()

    return points, normals, segments



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
    start_pos_1,start_quat_1, gripper_1 = start_ee_states[0]
    # start_pos_2,start_quat_2, gripper_2 = start_ee_states[1]
    leg_pose = env_states[1]
    target_ori_1 = rot_mat_tensor(np.pi, 0, 0, env.device)[:3, :3]
    target_pos_1 = leg_pose[:3,3]
    target_pos_1[2] += 0.1
    target_quat_1 = C.mat2quat(target_ori_1)
    target_ee_states = [(target_pos_1, target_quat_1, gripper_1)]#, (start_pos_2, start_quat_2, gripper_2)]
    thresholds = [(None,None),(None,None)]
    return target_ee_states, thresholds, False

def wait():
    for i in range(50):
        action = torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=env.device).unsqueeze(0)
        # new_action = torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=env.device).unsqueeze(0)
        # action = torch.cat((action, new_action), dim=1)

        env.step(action)

# Control Operations
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
    # target_pos_2, target_quat_2, gripper_2 = target_ee_states[1]
    pos_err_1, ori_err_1 = thresholds[0]
    pos_err_2, ori_err_2 = thresholds[1]   
    spend_time = 0
    while True:
        # ee_pos_1, ee_quat_1, ee_pos_2, ee_quat_2 = env.get_ee_pose_world()
        ee_pos_1, ee_quat_1 = env.get_ee_pose_world()
        ee_pos_1, ee_quat_1 = ee_pos_1.squeeze(), ee_quat_1.squeeze()
        # ee_pos_2, ee_quat_2 = ee_pos_2.squeeze(), ee_quat_2.squeeze()
        action_1 = get_action(ee_pos_1, ee_quat_1, target_pos_1, target_quat_1, gripper_1)
        # action_2 = get_action(ee_pos_2, ee_quat_2, target_pos_2, target_quat_2, gripper_2)
        # action = torch.cat((action_1, action_2), dim=1)
        action = action_1

        ee_pose_1 = C.to_homogeneous(ee_pos_1, C.quat2mat(ee_quat_1))
        # ee_pose_2 = C.to_homogeneous(ee_pos_2, C.quat2mat(ee_quat_2))
        target_pose_1 = C.to_homogeneous(target_pos_1, C.quat2mat(target_quat_1))
        # target_pose_2 = C.to_homogeneous(target_pos_2, C.quat2mat(target_quat_2))


        # zhp: need to change
        # gripper_width = env.dof_pos[:, 7:8] + env.dof_pos[:, 8:9]
        # half_width = 0.015
        gripper_width = 1
        half_width = 0

        if is_gripper:
            if gripper_less(gripper_width, 2 * half_width + 0.001, spend_time=spend_time):
                return
        else:
            if satisfy(ee_pose_1, target_pose_1, pos_err_1, ori_err_1, spend_time=spend_time): # and satisfy(ee_pose_2, target_pose_2, pos_err_2, ori_err_2, spend_time=spend_time):
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
        "close_gripper": close_gripper,
        "release_gripper": release_gripper,
        "pre_grasp": pre_grasp,
        "screw": screw,
        # "reach_leg_floor_xy": reach_leg_floor_xy,
        # "reach_leg_ori": reach_leg_ori,
        # "reach_leg_floor_z": reach_leg_floor_z,
        # "lift_up": lift_up,
        # "move_center": move_center,
        # "match_leg_ori": match_leg_ori,
        # "reach_table_top_xy": reach_table_top_xy,
        # "reach_table_top_z": reach_table_top_z,
        # "insert_table_wait": insert_table_wait,
        # "hold_table_xy": hold_table_xy,
        # "hold_table_z": hold_table_z,
        "pre_grasp_xy": pre_grasp_xy,
    }

    if last_target_ee_states is not None:
        ee_pos_1, ee_quat_1, gripper_1 = last_target_ee_states[0]
        # ee_pos_2, ee_quat_2, gripper_2 = last_target_ee_states[1]
    else:
        # ee_pos_1, ee_quat_1, ee_pos_2, ee_quat_2 = env.get_ee_pose_world()
        # gripper_1, gripper_2 = env.last_grasp_1, env.last_grasp_2
        ee_pos_1, ee_quat_1 = env.get_ee_pose_world()
        gripper_1 = env.last_grasp_1

    ee_pos_1, ee_quat_1 = ee_pos_1.squeeze(), ee_quat_1.squeeze()
    # ee_pos_2, ee_quat_2 = ee_pos_2.squeeze(), ee_quat_2.squeeze()
    # gripper_1, gripper_2 = gripper_1.squeeze(), gripper_2.squeeze()
    gripper_1 = gripper_1.squeeze()


    start_ee_states = [(ee_pos_1, ee_quat_1, gripper_1)]#, (ee_pos_2, ee_quat_2, gripper_2)]
    env_states = [table_pose, leg_pose]

    target_ee_states, thresholds, is_gripper = func_map[phase](env, start_ee_states, env_states)
    print("Start phase: ", phase)
    reach_target(target_ee_states, thresholds, is_gripper)
    print("End phase: ", phase)
    return target_ee_states

if __name__ == "__main__":
    # env = gym.make(
    # "dual-franka-hand-v0",
    # furniture='one_leg',
    # num_envs=1,
    # # record=True,
    # resize_img=False,
    # )
    # env.reset()
    # env.isaac_gym.start_access_image_tensors(env.sim)
    
    # camera_names = ["front", "back", "left", "right"]
    # points = None
    # normals = None
    # for camera_name in camera_names:
    #     camera_handle = env.camera_handles[camera_name][0]
    #     points_, normals_, segments = capture_pc(env, camera_handle)
    #     mask = (segments == 1) | (segments == 2)
    #     points_ = points_[mask]
    #     normals_ = normals_[mask]
    #     # points_ = torch.concatenate([points_[segments==1],points_[segments==2]],dim=0)
    #     if points is None:
    #         points = points_
    #         normals = normals_
    #     else:
    #         points = torch.cat((points, points_), dim=0)
    #         normals = torch.cat((normals, normals_), dim=0)
    # points = points.cpu().numpy()
    # normals = normals.cpu().numpy()
    points_path = "data/collect_data_7/points.pt"
    points = torch.load(points_path)
    points = points.numpy()

    normals_path = "data/collect_data_7/normals.pt"
    normals = torch.load(normals_path)
    normals = normals.numpy()

    leg_points_path = "data/collect_data_7/leg_points.pt"
    leg_points = torch.load(leg_points_path)
    leg_points = leg_points.numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    leg_pcd = o3d.geometry.PointCloud()
    leg_pcd.points = o3d.utility.Vector3dVector(leg_points)

    lines = []
    line_points = []
    line_count = 0

    for i in range(0, points.shape[0], 10):
        start_point = points[i]
        end_point = points[i] + 0.01 * normals[i]
        line_points.append(start_point)
        line_points.append(end_point)
        lines.append([2 * line_count, 2 * line_count + 1])
        line_count += 1

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(line_points))
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Visualize the point cloud and the normals
    o3d.visualization.draw_geometries([pcd, line_set,leg_pcd])

    # env.isaac_gym.end_access_image_tensors(env.sim)

    # wait()
    # env.set_hand_transform()
    # # wait()
    # target_ee_states = None
    # target_ee_states = act_phase(env, "pre_grasp_xy", target_ee_states)
    # for i in range(10):
    #     target_ee_states = act_phase(env, "release_gripper", target_ee_states)
    #     target_ee_states = act_phase(env, "pre_grasp", target_ee_states)
    #     target_ee_states = act_phase(env, "close_gripper", target_ee_states)
    #     target_ee_states = act_phase(env, "screw", target_ee_states)


