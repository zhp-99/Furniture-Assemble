import torch
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

data_folder = "data/finished"


sampled_points_list = []
distance_list = []
for i in range(17000):
    data_path = os.path.join(data_folder, "data_" + str(i) + ".pt")
    data_dict = torch.load(data_path)
    sampled_point = data_dict["sampled_point"]
    start_table_pose = data_dict["start_table_pose"]
    final_table_pose = data_dict["final_table_pose"]
    distance = (final_table_pose[:3,3]-start_table_pose[:3,3]).abs().sum()

    sampled_points_list.append(sampled_point)
    distance_list.append(distance)

leg_points = data_dict["leg_points"].cpu().numpy()

data_folder = "data/terminated"
terminated_sampled_points_list = []
for i in range(1):
    data_path = os.path.join(data_folder, "data_" + str(i) + ".pt")
    data_dict = torch.load(data_path)
    sampled_point = data_dict["sampled_point"]
    terminated_sampled_points_list.append(sampled_point)


sampled_points_tensor = torch.stack(sampled_points_list)
distance_tensor = torch.stack(distance_list)
terminated_sampled_points_tensor = torch.stack(terminated_sampled_points_list)

sampled_points_np = sampled_points_tensor.cpu().numpy()
terminated_sampled_points_np = terminated_sampled_points_tensor.cpu().numpy()
distance_np = distance_tensor.cpu().numpy()
distance_np = np.clip(distance_np, 0, 0.1)

distance_np = 0.1 - distance_np

# sorted_indices = np.argsort(distance_np)
# sorted_distance_np = distance_np[sorted_indices]

# plt.figure(figsize=(10, 6))
# plt.scatter(range(len(sorted_distance_np)), sorted_distance_np, c='blue', alpha=0.7)
# plt.title('Sorted distance_np Scatter Plot')
# plt.xlabel('Index')
# plt.ylabel('Distance')
# plt.grid(True)
# plt.show()



distance_normalized = (distance_np - distance_np.min()) / (distance_np.max() - distance_np.min())


all_points = np.concatenate([sampled_points_np, terminated_sampled_points_np, leg_points], axis=0)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points)


colors = plt.get_cmap("coolwarm")(distance_normalized)[:, :3]
# colors = np.zeros((distance_normalized.shape[0], 3))
colors[:, 0] = distance_normalized  # 将 distance_np 作为每一行的第一个值
terminated_color = np.zeros((terminated_sampled_points_np.shape[0], 3))

leg_colors = np.zeros((leg_points.shape[0], 3))
all_colors = np.concatenate([colors, terminated_color, leg_colors], axis=0)

pcd.colors = o3d.utility.Vector3dVector(all_colors)

o3d.visualization.draw_geometries([pcd])