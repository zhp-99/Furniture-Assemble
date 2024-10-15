import torch
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import trange

data_folder = "data/terminated"


sampled_points_list = []
distance_list = []
for i in trange(8000, 18001):
    data_path = os.path.join(data_folder, "data_" + str(i) + ".pt")
    data_dict = torch.load(data_path)
    sampled_point = data_dict["sampled_point"]
    table_points = data_dict["table_points"]

    # Use torch to find the index of sampled_point in table_points
    index = torch.where((table_points == sampled_point).all(dim=1))[0]
    assert len(index) == 1

    data_dict["sampled_point_index"] = index.item()
    torch.save(data_dict, data_path)