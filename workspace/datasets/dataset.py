import torch
import os
from torch.utils.data import Dataset

class AssembleDataset(Dataset):
    def __init__(self, split):
        self.data = self.make_dataset(split)

    def __len__(self):
        """
        Returns:
            int: The total number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # "all_finished": all_finished,
        # "table_points": points.cpu(),
        # "table_normals": normals.cpu(),
        # "leg_points": leg_points.cpu(),
        # "ori_table_pose": ori_table_pose.cpu(),
        # "start_table_pose": start_table_pose.cpu(), 
        # "final_table_pose": final_table_pose.cpu(),
        # "sampled_point": sampled_point.cpu(),
        # "sampled_normal": sampled_normal.cpu(),
        # "action_direct": action_direct.cpu(),

        all_finished = sample["all_finished"]
        if all_finished:
            reward = (sample["final_table_pose"][:3,3] - sample["start_table_pose"][:3,3]).abs().sum()
        else:
            reward = torch.tensor(1)
        
        return sample["table_points"],  sample["leg_points"], sample["sampled_point"], sample["sampled_normal"], sample["action_direct"], reward
    
    def make_dataset(self, split):
        finised_folder = "data/finished"
        terminated_folder = "data/terminated"

        dataset = []
        if split == "train":
            for i in range(10000):
                finished_data = torch.load(os.path.join(finised_folder, "data_" + str(i) + ".pt"))
                terminated_data = torch.load(os.path.join(terminated_folder, "data_" + str(i) + ".pt"))
                dataset.append(finished_data)
                dataset.append(terminated_data)
        else:
            for i in range(10000, 13000):
                finished_data = torch.load(os.path.join(finised_folder, "data_" + str(i) + ".pt"))
                terminated_data = torch.load(os.path.join(terminated_folder, "data_" + str(i) + ".pt"))
                dataset.append(finished_data)
                dataset.append(terminated_data)
        
        return dataset