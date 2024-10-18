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

        all_finished = sample["all_finished"]
        if all_finished:
            reward = (sample["final_table_pose"][:3,3] - sample["start_table_pose"][:3,3]).abs().sum()
        else:
            raise ValueError("This should not happen")
            reward = torch.tensor(0.5)
        
        # clamp to 0-0.1
        reward = torch.clamp(reward, 0, 0.1)
        
        return sample["table_points"],  sample["leg_points"], sample["sampled_point"], sample["sampled_normal"], sample["action_direct"], reward
    
    def make_dataset(self, split):
        finised_folder = "data/finished"
        terminated_folder = "data/terminated"

        dataset = []
        train_num = 10000
        val_num = 3000
        if split == "train":
            # check cache
            if os.path.exists(os.path.join("data", split + "_" + str(train_num) + ".pt")):
                print("loading from cache")
                return torch.load(os.path.join("data", split + "_" + str(train_num) + ".pt"), weights_only=True)
        
            for i in range(train_num):
                finished_data = torch.load(os.path.join(finised_folder, "data_" + str(i) + ".pt"), weights_only=True)
                # terminated_data = torch.load(os.path.join(terminated_folder, "data_" + str(i) + ".pt"), weights_only=True)
                dataset.append(finished_data)
                # dataset.append(terminated_data)
        else:
            # check cache
            if os.path.exists(os.path.join("data", split + "_" + str(val_num) + ".pt")):
                print("loading from cache")
                return torch.load(os.path.join("data", split + "_" + str(val_num) + ".pt"), weights_only=True)
            
            for i in range(train_num, train_num + val_num):
                finished_data = torch.load(os.path.join(finised_folder, "data_" + str(i) + ".pt"), weights_only=True)
                # terminated_data = torch.load(os.path.join(terminated_folder, "data_" + str(i) + ".pt"), weights_only=True)
                dataset.append(finished_data)
                # dataset.append(terminated_data)
        
        # cache the dataset
        if split == "train":
            torch.save(dataset, os.path.join("data", split + "_" + str(train_num) + ".pt"))
        else:
            torch.save(dataset, os.path.join("data", split + "_" + str(val_num) + ".pt"))
        
        return dataset