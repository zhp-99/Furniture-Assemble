import os
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from models.model_assemble import Network as CriticNetwork
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, table_points, table_normals, sampled_points, sampled_normals, action_directs):
        self.table_points = table_points
        self.table_normals = table_normals
        self.sampled_points = sampled_points
        self.sampled_normals = sampled_normals
        self.action_directs = action_directs
    def __len__(self):
        return len(self.table_points)
    def __getitem__(self, idx):
        return self.table_points[idx], self.table_normals[idx], self.sampled_points[idx], self.sampled_normals[idx], self.action_directs[idx]
if __name__ == "__main__":
    checkpoint_path = "logs/ckpts/500-network.pth"
    finised_folder = "data/finished"

    network = CriticNetwork(128, 32, 32)
    network.load_state_dict(torch.load(checkpoint_path,weights_only=True))

    table_points = []
    table_normals = []
    sampled_points = []
    sampled_normals = []
    action_directs = []
    points_num = 4096
    start_num = 10000
    for i in range(start_num,start_num+points_num):
        finished_data = torch.load(os.path.join(finised_folder, "data_" + str(i) + ".pt"), weights_only=True)
        table_points.append(finished_data["table_points"])
        table_normals.append(finished_data["table_normals"])
        sampled_points.append(finished_data["sampled_point"])
        sampled_normals.append(finished_data["sampled_normal"])
        action_directs.append(finished_data["action_direct"])
    
    leg_points = finished_data["leg_points"].cpu().numpy()

    dataset = SimpleDataset(table_points, table_normals, sampled_points, sampled_normals, action_directs)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    


    batch_size = 8
    num_batches = points_num // batch_size

    network = network.cuda()
    network.eval()
    pred_scores = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_table_points, batch_table_normals, batch_sampled_points, batch_sampled_normals, batch_action_directs = batch
            batch_table_points = batch_table_points.cuda()
            batch_sampled_points = batch_sampled_points.cuda()
            batch_action_directs = batch_action_directs.cuda()
            batch_sampled_normals = batch_sampled_normals.cuda()
            pred_score = network.forward(batch_table_points, batch_sampled_points, batch_sampled_normals)
            pred_scores.append(pred_score)
    
    pred_scores = torch.cat(pred_scores, dim=0)

    score_np = pred_scores.cpu().numpy()
    # score_np = np.clip(score_np, 0, 0.1)

    score_np = 0.1 - score_np

    score_normalized = (score_np - score_np.min()) / (score_np.max() - score_np.min())


    sampled_points_np = torch.stack(sampled_points).cpu().numpy()
    all_points = np.concatenate([sampled_points_np, leg_points], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)


    colors = plt.get_cmap("coolwarm")(score_normalized)[:, :3]
    colors[:, 0] = score_normalized  # 将 score_np 作为每一行的第一个值


    leg_colors = np.zeros((leg_points.shape[0], 3))
    all_colors = np.concatenate([colors, leg_colors], axis=0)

    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    o3d.visualization.draw_geometries([pcd])