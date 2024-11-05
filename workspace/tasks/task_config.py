import os
import numpy as np
from typing import Any, Dict

all_task_config: Dict[str, Any] = {
    "square_table": {
        "task_name": "square_table",
        "furniture_name": "square_table",
        "part_names": ["square_table_top", "square_table_leg4"],
        "disassembled_pose_path": "/home/zhp/workspace/furniture-bench/workspace/tasks/relative_poses/square_table_square_table_top_square_table_leg4.npy",
        "part_frictions": [0.03, 2],
        "randomness": {
            "pos": [(-0.02,0.02), (-0.02, 0.02), (0,0)],
            "ori": [(0,0), (0,0), (-np.pi, np.pi)]
        }
    },
    "lamp": {
        "task_name": "lamp",
        "furniture_name": "lamp",
        "part_names": ["lamp_base", "lamp_bulb"],
        "disassembled_pose_path": "/home/zhp/workspace/furniture-bench/workspace/tasks/relative_poses/lamp_lamp_base_lamp_bulb.npy",
        "part_frictions": [0.03, 0.5],
        "part_mass": [1, 0.5],
        "randomness": {
            "pos": [(-0.02,0.02), (-0.02, 0.02), (0,0)],
            "ori": [(0,0), (0,0), (-np.pi, np.pi)]
        }
    },
    "desk": {
        "task_name": "desk",
        "furniture_name": "desk",
        "part_names": ["desk_top", "desk_leg4"],
        "disassembled_pose_path": "/home/zhp/workspace/furniture-bench/workspace/tasks/relative_poses/desk_desk_top_desk_leg4.npy",
        "part_frictions": [0.03, 2],
        "part_mass": [0.5, 0.25],
        "randomness": {
            "pos": [(-0.02,0.02), (-0.02, 0.02), (0,0)],
            "ori": [(0,0), (0,0), (-np.pi, np.pi)]
        }
    },
    "drawer_top": {
        "task_name": "drawer_top",
        "furniture_name": "drawer",
        "part_names": ["drawer_box", "drawer_container_top"],
        "disassembled_pose_path": "/home/zhp/workspace/furniture-bench/workspace/tasks/relative_poses/drawer_drawer_box_drawer_container_top.npy",
        "part_frictions": [0.1, 0.1],
        "randomness": {
            "pos": [(-0.02,0.02), (-0.02, 0.02), (0,0)],
            "ori": [(0,0), (0,0), (-np.pi/6, np.pi/3)]
        }
    },
    "drawer_bottom": {
        "task_name": "drawer_bottom",
        "furniture_name": "drawer",
        "part_names": ["drawer_box", "drawer_container_bottom"],
        "disassembled_pose_path": "/home/zhp/workspace/furniture-bench/workspace/tasks/relative_poses/drawer_drawer_box_drawer_container_bottom.npy",
        "part_frictions": [0.1, 0.1],
        "randomness": {
            "pos": [(-0.02,0.02), (-0.02, 0.02), (0,0)],
            "ori": [(0,0), (0,0), (-np.pi/6, np.pi/3)]
        },
    },
    "cabinet_door_left":{
        "task_name": "cabinet_door_left",
        "furniture_name": "cabinet",
        "part_names": ["cabinet_body", "cabinet_door_left"],
        "disassembled_pose_path": "/home/zhp/workspace/furniture-bench/workspace/tasks/relative_poses/cabinet_cabinet_body_cabinet_door_left.npy",
        # "part_frictions": [0.1, 0.1],
        "randomness": {
            "pos": [(-0.02,0.02), (-0.02, 0.02), (0,0)],
            "ori": [(0,0), (0,0), (np.pi/3, np.pi/2+np.pi/3)],
            # "ori": [(0,0), (0,0), (0,0)]
        }
    }
}