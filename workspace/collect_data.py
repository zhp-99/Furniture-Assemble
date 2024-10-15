import os
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

    return False

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

def small_wait():
    for i in range(10):
        action = torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=env.device).unsqueeze(0)
        # new_action = torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=env.device).unsqueeze(0)
        # action = torch.cat((action, new_action), dim=1)

        env.step(action)

def wait():
    for i in range(30):
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
        ee_pos_1, ee_quat_1 = env.get_ee_pose_world()
        ee_pos_1, ee_quat_1 = ee_pos_1.squeeze(), ee_quat_1.squeeze()
        action_1 = get_action(ee_pos_1, ee_quat_1, target_pos_1, target_quat_1, gripper_1)
        action = action_1

        ee_pose_1 = C.to_homogeneous(ee_pos_1, C.quat2mat(ee_quat_1))
        target_pose_1 = C.to_homogeneous(target_pos_1, C.quat2mat(target_quat_1))

        gripper_width = 1
        half_width = 0

        if is_gripper:
            if gripper_less(gripper_width, 2 * half_width + 0.001) or spend_time > 10:
                return True
        else:
            if satisfy(ee_pose_1, target_pose_1, pos_err_1, ori_err_1): # and satisfy(ee_pose_2, target_pose_2, pos_err_2, ori_err_2, spend_time=spend_time):
                return True
            if spend_time > 30:
                return False
            
        
        
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
    # print("Start phase: ", phase)
    result = reach_target(target_ee_states, thresholds, is_gripper)
    # print("End phase: ", phase)
    return target_ee_states, result

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


def farthest_point_sample_GPU(points, npoint): 

    """
    Input:
        points: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint]
    """

    B, N, C = points.shape
    centroids = torch.zeros((B, npoint), dtype=torch.long, device=points.device)
    distance = torch.ones((B, N), device=points.device) * 1e10

    batch_indices = torch.arange(B, device=points.device)
    
    barycenter = torch.sum(points, dim=1) / N  
    barycenter = barycenter.view(B, 1, C)

    dist = torch.sum((points - barycenter) ** 2, dim=-1)  # (B,N)
    farthest = torch.argmax(dist, dim=1)  # (B)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((points - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask] 

        farthest = torch.argmax(distance, dim=1)
    # sampled_points = points[batch_indices, centroids, :]

    return centroids

def normal_to_direction(normal):

    z_axis = torch.tensor([0, 0, 1], device=normal.device, dtype=normal.dtype)
    cos_theta = torch.dot(normal, z_axis) / (torch.norm(normal) * torch.norm(z_axis))
    theta = torch.acos(cos_theta) 


    angle_with_xy_plane = np.pi / 2 - theta


    normal_xy_projection = normal.clone()
    normal_xy_projection[2] = 0


    normal_xy_projection = normal.clone()
    normal_xy_projection[2] = 0

    x_axis = torch.tensor([1, 0, 0], device=normal.device, dtype=normal.dtype)
    y_axis = torch.tensor([0, 1, 0], device=normal.device, dtype=normal.dtype)

    # 使用 atan2 计算角度
    phi = torch.atan2(torch.dot(normal_xy_projection, y_axis), torch.dot(normal_xy_projection, x_axis))


    return angle_with_xy_plane.item(), phi.item()



if __name__ == "__main__":
    env = gym.make(
    "dual-franka-hand-v0",
    furniture='one_leg',
    num_envs=1,
    # record=True,
    resize_img=False,
    )

    terminated_count = 15531
    finished_count = 8738
    for collect_count in range(10000000):
        env.reset()
        rb_states = env.rb_states
        part_idxs = env.part_idxs

        # leg_pose = C.to_homogeneous(
        #     rb_states[part_idxs["square_table_leg4"]][0][:3],
        #     C.quat2mat(rb_states[part_idxs["square_table_leg4"]][0][3:7]),
        # )
        ori_table_pose = C.to_homogeneous(
            rb_states[part_idxs["square_table_top"]][0][:3],
            C.quat2mat(rb_states[part_idxs["square_table_top"]][0][3:7]),
        )
        env.isaac_gym.start_access_image_tensors(env.sim)
        camera_names = ["front", "back", "left", "right"]
        points = None
        normals = None
        leg_points = None
        for camera_name in camera_names:
            camera_handle = env.camera_handles[camera_name][0]
            points_, normals_, segments = capture_pc(env, camera_handle)
            leg_points_ = points_[segments==2]

            mask = (segments == 1) # | (segments == 2)
            points_ = points_[mask]
            normals_ = normals_[mask]

            

            if points is None:
                points = points_
                normals = normals_
                leg_points = leg_points_
            else:
                points = torch.cat((points, points_), dim=0)
                normals = torch.cat((normals, normals_), dim=0)
                leg_points = torch.cat((leg_points, leg_points_), dim=0)

        points = points.unsqueeze(0)
        fps_indexes = farthest_point_sample_GPU(points, 2048).squeeze(0) #(2048)
        points = points.squeeze(0)

        points = points[fps_indexes, :]
        normals = normals[fps_indexes, :]

        leg_points = leg_points.unsqueeze(0)
        leg_fps_indexes = farthest_point_sample_GPU(leg_points, 2048).squeeze(0)
        leg_points = leg_points.squeeze(0)
        leg_points = leg_points[leg_fps_indexes, :]

        random_index = torch.randint(0, points.shape[0], (1,), device=points.device)
        sampled_point = points[random_index].squeeze(0) #(3)
        sampled_normal = normals[random_index].squeeze(0) #(3)

        angle_with_xy_plane, phi = normal_to_direction(sampled_normal)

        sampled_angle_with_xy_plane = -1
        num_trials = 0
        while sampled_angle_with_xy_plane < 0:
            random_angle = np.random.uniform(-np.pi/3, np.pi/3)
            sampled_angle_with_xy_plane = angle_with_xy_plane + random_angle
            num_trials += 1
            if num_trials > 1000:
                break
        if num_trials > 1000:
            continue

        hand_ori = rot_mat([0, np.pi/2+sampled_angle_with_xy_plane, 0], hom=True)
        hand_ori = rot_mat([0, 0, phi-np.pi], hom=True) @ hand_ori

        normal_xy_projection = torch.sqrt(sampled_normal[0]**2 + sampled_normal[1]**2)
        sampled_angle_with_xy_plane = torch.tensor(sampled_angle_with_xy_plane, device=sampled_normal.device)
        action_z = normal_xy_projection * torch.abs(torch.sin(sampled_angle_with_xy_plane)/torch.cos(sampled_angle_with_xy_plane))

        
        action_direct = sampled_normal.clone()
        if sampled_angle_with_xy_plane > np.pi/2:
            action_direct = -action_direct
        action_direct[2] = action_z
        action_direct = action_direct / torch.norm(action_direct)

        dev = 0.0045*np.sin(random_angle)

        hand_pos = sampled_point + action_direct * (0.12+dev)
        hand_pos = hand_pos.cpu().numpy()
        env.isaac_gym.end_access_image_tensors(env.sim)

        # Start simulation
        # check cotact
        # For unknown reason, the contact force tensor is not updated in the first step
        env.refresh()
        env.isaac_gym.refresh_net_contact_force_tensor(env.sim)
        _net_cf = env.isaac_gym.acquire_net_contact_force_tensor(env.sim)
     
        env.set_hand_transform(hand_pos, hand_ori)
        contact_flag = False
        fake_contact_flag = False
        for i in range(10):
            env.refresh()
            env.isaac_gym.refresh_net_contact_force_tensor(env.sim)
            _net_cf = env.isaac_gym.acquire_net_contact_force_tensor(env.sim)
            net_cf = gymtorch.wrap_tensor(_net_cf)
            table_top_cf = net_cf[part_idxs["square_table_top"]]
            hand_cf = net_cf[env.franka_hand_rigid_index]
            base_table_cf = net_cf[env.base_table_rigid_index]

            if torch.any(torch.abs(base_table_cf[:2]) > 50):
                contact_flag = True
                break
                
            if torch.any(torch.abs(table_top_cf[:2]) > 50):
                contact_flag = True
                break
        
        
        
        if not contact_flag:
            all_finished = True
            target_ee_states = None
            target_ee_states, result = act_phase(env, "pre_grasp_xy", target_ee_states)

            # check if the table is moved
            start_table_pose = C.to_homogeneous(
                rb_states[part_idxs["square_table_top"]][0][:3],
                C.quat2mat(rb_states[part_idxs["square_table_top"]][0][3:7]),
            )
            if not satisfy(ori_table_pose, start_table_pose):
                print("Table Moved")
                all_finished = False

            if not result:
                print("Gripper Collision at Pre Grasp XY")
                all_finished = False

            # Start screw
            if all_finished:
                for i in range(2):
                    target_ee_states, result = act_phase(env, "release_gripper", target_ee_states)
                    target_ee_states, result = act_phase(env, "pre_grasp_xy", target_ee_states)
                    if not result:
                        print("Gripper Collision at Pre Grasp XY")
                        all_finished = False
                        break
                    target_ee_states, result = act_phase(env, "pre_grasp", target_ee_states)
                    if not result:
                        print("Gripper Collision at Pre Grasp")
                        all_finished = False
                        break
                    target_ee_states, result = act_phase(env, "close_gripper", target_ee_states)
                    target_ee_states, result = act_phase(env, "screw", target_ee_states)

            if all_finished:
                save_path = "data/finished/data_%d.pt" % finished_count
                finished_count += 1
                final_table_pose = C.to_homogeneous(
                    rb_states[part_idxs["square_table_top"]][0][:3],
                    C.quat2mat(rb_states[part_idxs["square_table_top"]][0][3:7]),
                )
                distance = (final_table_pose[:3,3]-ori_table_pose[:3,3]).abs().sum()
                print("Distance: ", distance)
                # save data
                # torch.save({
                #     "all_finished": all_finished,
                #     "table_points": points.cpu(),
                #     "table_normals": normals.cpu(),
                #     "leg_points": leg_points.cpu(),
                #     "ori_table_pose": ori_table_pose.cpu(),
                #     "start_table_pose": start_table_pose.cpu(), 
                #     "final_table_pose": final_table_pose.cpu(),
                #     "sampled_point": sampled_point.cpu(),
                #     "sampled_normal": sampled_normal.cpu(),
                #     "action_direct": action_direct.cpu(),
                #     }, save_path)
            else:
                save_path = "data/terminated/data_%d.pt" % terminated_count
                terminated_count += 1
                # torch.save({
                #     "all_finished": all_finished,
                #     "table_points": points.cpu(),
                #     "table_normals": normals.cpu(),
                #     "leg_points": leg_points.cpu(),
                #     "ori_table_pose": ori_table_pose.cpu(), 
                #     "sampled_point": sampled_point.cpu(),
                #     "sampled_normal": sampled_normal.cpu(),
                #     "action_direct": action_direct.cpu(),
                #     }, save_path)


        else:
            save_path = "data/terminated/data_%d.pt" % terminated_count
            terminated_count += 1
            # torch.save({
            #     "all_finished": False,
            #     "table_points": points.cpu(),
            #     "table_normals": normals.cpu(),
            #     "leg_points": leg_points.cpu(),
            #     "ori_table_pose": ori_table_pose.cpu(), 
            #     "sampled_point": sampled_point.cpu(),
            #     "sampled_normal": sampled_normal.cpu(),
            #     "action_direct": action_direct.cpu(),
            #     }, save_path)

        env.refresh()
        env.isaac_gym.refresh_net_contact_force_tensor(env.sim)
        _net_cf = env.isaac_gym.acquire_net_contact_force_tensor(env.sim)

        if finished_count >= 40000:
            break


