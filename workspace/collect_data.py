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
import importlib

from tasks.utils import *
from tasks.task_config import task_config

def collect_data(furniture_name, part1_name, part2_name):
    env = gym.make(
        "dual-franka-hand-v0",
        furniture=furniture_name,
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

        ori_part1_pose = C.to_homogeneous(
            rb_states[part_idxs[part1_name]][0][:3],
            C.quat2mat(rb_states[part_idxs[part1_name]][0][3:7]),
        )
        env.isaac_gym.start_access_image_tensors(env.sim)
        camera_names = ["front", "back", "left", "right"]
        points = None
        normals = None
        part2_points = None
        for camera_name in camera_names:
            camera_handle = env.camera_handles[camera_name][0]
            points_, normals_, segments = capture_pc(env, camera_handle)
            part2_points_ = points_[segments==2]

            mask = (segments == 1)
            points_ = points_[mask]
            normals_ = normals_[mask]

            if points is None:
                points = points_
                normals = normals_
                part2_points = part2_points_
            else:
                points = torch.cat((points, points_), dim=0)
                normals = torch.cat((normals, normals_), dim=0)
                part2_points = torch.cat((part2_points, part2_points_), dim=0)

        points = points.unsqueeze(0)
        fps_indexes = farthest_point_sample_GPU(points, 2048).squeeze(0) #(2048)
        points = points.squeeze(0)

        points = points[fps_indexes, :]
        normals = normals[fps_indexes, :]

        part2_points = part2_points.unsqueeze(0)
        part2_fps_indexes = farthest_point_sample_GPU(part2_points, 2048).squeeze(0)
        part2_points = part2_points.squeeze(0)
        part2_points = part2_points[part2_fps_indexes, :]

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
            part1_top_cf = net_cf[part_idxs[part1_name]]
            hand_cf = net_cf[env.franka_hand_rigid_index]
            base_table_cf = net_cf[env.base_table_rigid_index]

            if torch.any(torch.abs(base_table_cf[:2]) > 50):
                contact_flag = True
                break
                
            if torch.any(torch.abs(part1_top_cf[:2]) > 50):
                contact_flag = True
                break
        
        # Start to perform the task
        task_module = importlib.import_module(f'tasks.{furniture_name}')
        prepare = getattr(task_module, 'prepare')
        perform = getattr(task_module, 'perform')
        if not contact_flag:
            all_finished = True
            target_ee_states = None
            target_ee_states, result = prepare(env, target_ee_states)
            # check if the part1 is moved
            start_part1_pose = C.to_homogeneous(
                rb_states[part_idxs[part1_name]][0][:3],
                C.quat2mat(rb_states[part_idxs[part1_name]][0][3:7]),
            )
            if not satisfy(ori_part1_pose, start_part1_pose):
                print("Part1 Moved")
                all_finished = False

            if not result:
                print("Gripper Collision at Pre Grasp XY")
                all_finished = False

            # Start screw
            if all_finished:
                all_finished = perform(env, target_ee_states)
            if all_finished:
                save_path = "data/finished/data_%d.pt" % finished_count
                finished_count += 1
                final_part1_pose = C.to_homogeneous(
                    rb_states[part_idxs[part1_name]][0][:3],
                    C.quat2mat(rb_states[part_idxs[part1_name]][0][3:7]),
                )
                distance = (final_part1_pose[:3,3]-ori_part1_pose[:3,3]).abs().sum()
                print("Distance: ", distance)
                # save data
                # torch.save({
                #     "all_finished": all_finished,
                #     "part1_points": points.cpu(),
                #     "part1_normals": normals.cpu(),
                #     "part2_points": part2_points.cpu(),
                #     "ori_part1_pose": ori_part1_pose.cpu(),
                #     "start_part1_pose": start_part1_pose.cpu(), 
                #     "final_part1_pose": final_part1_pose.cpu(),
                #     "sampled_point": sampled_point.cpu(),
                #     "sampled_normal": sampled_normal.cpu(),
                #     "action_direct": action_direct.cpu(),
                #     }, save_path)
            else:
                save_path = "data/terminated/data_%d.pt" % terminated_count
                terminated_count += 1
                # torch.save({
                #     "all_finished": all_finished,
                #     "part1_points": points.cpu(),
                #     "part1_normals": normals.cpu(),
                #     "part2"dual-franka-hand-v0"_points": part2_points.cpu(),
                #     "ori_part1_pose": ori_part1_pose.cpu(), 
                #     "sampled_point": sampled_point.cpu(),
                #     "sampled_normal": sampled_normal.cpu(),
                #     "action_direct": action_direct.cpu(),
                #     }, save_path)


        else:
            save_path = "data/terminated/data_%d.pt" % terminated_count
            terminated_count += 1
            # torch.save({
            #     "all_finished": False,
            #     "part1_points": points.cpu(),
            #     "part1_normals": normals.cpu(),
            #     "part2_points": part2_points.cpu(),
            #     "ori_part1_pose": ori_part1_pose.cpu(), 
            #     "sampled_point": sampled_point.cpu(),
            #     "sampled_normal": sampled_normal.cpu(),
            #     "action_direct": action_direct.cpu(),
            #     }, save_path)

        env.refresh()
        env.isaac_gym.refresh_net_contact_force_tensor(env.sim)
        _net_cf = env.isaac_gym.acquire_net_contact_force_tensor(env.sim)

        if finished_count >= 40000:
            break

def collect_other():
    env = gym.make(
        "dual-franka-hand-v0",
        furniture='lamp',
        num_envs=1,
        # record=True,
        resize_img=False,
    )
    env.reset()
    env.refresh()
    for i in range(10):
        wait(env)

if __name__ == "__main__":
    furniture_name = "square_table"
    collect_data(furniture_name, task_config[furniture_name]["part_names"][0], task_config[furniture_name]["part_names"][1])
    

