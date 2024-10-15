import furniture_bench
import gym
import torch
import numpy as np
import furniture_bench.controllers.control_utils as C
from furniture_bench.config import ROBOT_HEIGHT, config
from furniture_bench.utils.pose import *

####################################################################################
##                           FurnitureSim Configuration                           ##
####################################################################################

# env = gym.make(
#   "test-v0",
#   furniture = "lamp",                # Specifies the type of furniture [lamp | square_table | desk | drawer | cabinet | round_table | stool | chair | one_leg].
#   num_envs=1,               # Number of parallel environments.
#   obs_keys=None,            # List of observations.
#   concat_robot_state=False, # Whether to return robot_state in a vector or dictionary.
#   resize_img=True,          # If true, images are resized to 224 x 224.
#   headless=False,           # If true, simulation runs without GUI.
#   compute_device_id=0,      # GPU device ID for simulation.
#   graphics_device_id=0,     # GPU device ID for rendering.
#   init_assembled=False,     # If true, the environment is initialized with assembled furniture.
#   np_step_out=False,        # If true, env.step() returns Numpy arrays.
#   channel_first=False,      # If true, images are returned in channel first format.
#   randomness="low",         # Level of randomness in the environment [low | med | high].
#   high_random_idx=-1,       # Index of the high randomness level (range: [0-2]). Default -1 will randomly select the index within the range.
#   save_camera_input=False,  # If true, the initial camera inputs are saved.
#   record=False,             # If true, videos of the wrist and front cameras' RGB inputs are recorded.
#   max_env_steps=3000,       # Maximum number of steps per episode.
#   act_rot_repr='quat'       # Representation of rotation for action space. Options are 'quat' and 'axis'.
# )
# env.reset()
# input("Press Enter to continue...")

####################################################################################
##                           FurnitureSim env.step                                ##
####################################################################################

# """
# # Input
# action: torch.Tensor or np.ndarray (shape: [num_envs, action_dim]) # Action space is 8-dimensional (3D EE delta position, 4D EE delta rotation (quaternion), and 1D gripper.Range to [-1, 1].

# # Output
# obs: Dictionary of observations. The keys are specified in obs_keys. The default keys are: ['color_image1', 'color_image2', 'robot_state'].
# reward: torch.Tensor or np.ndarray (shape: [num_envs, 1])
# done: torch.Tensor or np.ndarray (shape: [num_envs, 1])
# info: Dictionary of additional information.
# """
# env = gym.make(
#     "test-v0",
#     furniture='one_leg',
#     num_envs=1,
# )
# env.reset()
# for i in range(100):
#     ac = torch.tensor(env.action_space.sample()).float().to('cuda') # (1, 8) torch.Tensor
#     ob, rew, done, _ = env.step(ac)
#     # env.render()

# print(ob.keys())                # ['color_image1', 'color_image2', 'robot_state']
# print(ob['robot_state'].keys()) # ['ee_pos', 'ee_quat', 'ee_pos_vel', 'ee_ori_vel', 'gripper_width']
# print(ob['color_image1'].shape) # Wrist camera of shape (1, 224, 224, 3)
# print(ob['color_image2'].shape) # Front camera os shape (1, 224, 224, 3)
# print(rew.shape)                # (1, 1)
# print(done.shape)               # (1, 1)
# input("Press Enter to continue...")
# env.close()
# def action_tensor(ac):
#     if isinstance(ac, (list, np.ndarray)):
#         return torch.tensor(ac).float().to(env.device)

#     ac = ac.clone()
#     if len(ac.shape) == 1:
#         ac = ac[None]
#     return ac.tile(1, 1).float().to(env.device)

# done = False
# while not done:
#     action, skill_complete = env.get_assembly_action()
#     action = action_tensor(action)
#     new_action = torch.tensor([0, 0, 0, 0, 0, 0, 1, -1]).float().unsqueeze(0).to(env.device)
#     action = torch.cat((action, new_action), dim=1)
#     ee_pos,ee_quat = env.get_ee_pose()
#     ob, rew, done, _ = env.step(action)
# for i in range(100):
#     action = torch.tensor([0, 0, 0, 0, 0, 0, 1, -1]).float().unsqueeze(0).to(env.device)
#     new_action = torch.tensor([0, 0, 0, 0, 0, 0, 1, -1]).float().unsqueeze(0).to(env.device)
#     action = torch.cat((action, new_action), dim=1)
#     ob, rew, done, _ = env.step(action)
#     ee_pose, _ = env.get_ee_pose()
#     input()

# for i in range(100):

#     table_top_pos = env.rb_states

#     ee_pos_1, ee_quat_1,ee_pos_2,ee_quat_2 = env.get_ee_pose()
#     ee_pos_1, ee_quat_1 = ee_pos_1.squeeze(), ee_quat_1.squeeze()
#     ee_pos_2, ee_quat_2 = ee_pos_2.squeeze(), ee_quat_2.squeeze()

#     gripper = torch.tensor([-1], dtype=torch.float32, device=env.device)
#     goal_pos = torch.tensor(
#         [ee_pos_1[0], ee_pos_1[1], 0.3], device=env.device
#     )
#     delta_pos = goal_pos - ee_pos_1
#     delta_quat = torch.tensor([0, 0, 0, 1], device=env.device)
#     action = torch.concat([delta_pos, delta_quat, gripper]).unsqueeze(0)

#     gripper = torch.tensor([-1], dtype=torch.float32, device=env.device)
#     goal_pos = torch.tensor(
#         [ee_pos_2[0], ee_pos_2[1], 0.3], device=env.device
#     )
#     delta_pos = goal_pos - ee_pos_2
#     delta_quat = torch.tensor([0, 0, 0, 1], device=env.device)
#     new_action = torch.concat([delta_pos, delta_quat, gripper]).unsqueeze(0)
#     action = torch.cat((action, new_action), dim=1)
#     ob, rew, done, _ = env.step(action)


# input("Press Enter to continue...")

env = gym.make(
    "test-v0",
    furniture='one_leg',
    num_envs=1,
)
env.reset()

gripper_width = env.dof_pos[:, 7:8] + env.dof_pos[:, 8:9]
half_width = 0.015

base_tag_from_robot_mat = config["robot"]["tag_base_from_robot_base"]
april_to_robot = torch.tensor(base_tag_from_robot_mat, device=env.device)

# franka_from_origin_mat_1 = get_mat(
#             [env.franka_pose_1.p.x, env.franka_pose_1.p.y, env.franka_pose_1.p.z],
#             [0, 0, 0],
#         )
# sim_to_april_mat_1 = torch.tensor(
#             np.linalg.inv(base_tag_from_robot_mat)
#             @ np.linalg.inv(franka_from_origin_mat_1),
#             device=env.device,
#         )

# env.close()
# print(base_tag_from_robot_mat)
# print(np.linalg.inv(franka_from_origin_mat_1))
print([env.franka_pose_1.p.x, env.franka_pose_1.p.y, env.franka_pose_1.p.z])