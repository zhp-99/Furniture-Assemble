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

def disassemble(furniture_name, part1_name, part2_name):
    env = gym.make(
        "dual-franka-hand-v0",
        furniture=furniture_name,
        num_envs=1,
        # record=True,
        resize_img=False,
        assembled=True,
        set_friction=False,
    )

    env.reset()
    env.refresh()

    # Start to perform the task
    rb_states = env.rb_states
    part_idxs = env.part_idxs
    task_module = importlib.import_module(f'tasks.{furniture_name}')
    prepare = getattr(task_module, 'prepare')
    perform_disassemble = getattr(task_module, 'perform_disassemble')

    target_ee_states = None
    target_ee_states, result = prepare(env, target_ee_states)

    part2_pose = C.to_homogeneous(
            rb_states[part_idxs[part2_name]][0][:3],
            C.quat2mat(rb_states[part_idxs[part2_name]][0][3:7]),
    )

    all_finished = perform_disassemble(env, target_ee_states)

    part1_pose = C.to_homogeneous(
            rb_states[part_idxs[part1_name]][0][:3],
            C.quat2mat(rb_states[part_idxs[part1_name]][0][3:7]),
    )
    part2_pose = C.to_homogeneous(
            rb_states[part_idxs[part2_name]][0][:3],
            C.quat2mat(rb_states[part_idxs[part2_name]][0][3:7]),
    )
    # Compute relative pose
    # Use torch to compute relative pose
    relative_pose = torch.linalg.inv(part1_pose) @ part2_pose
    relative_pose = relative_pose.cpu().numpy()
    # Save relative_pose
    relative_pose_path = os.path.join("tasks", "relative_poses", f"{furniture_name}_{part1_name}_{part2_name}.npy")
    np.save(relative_pose_path, relative_pose)

    env.refresh()

if __name__ == "__main__":
    # furniture_name = "square_table"
    furniture_name = "desk"
    part1_name = task_config[furniture_name]["part_names"][0]
    part2_name = task_config[furniture_name]["part_names"][1]
    disassemble(furniture_name, part1_name, part2_name)
    

