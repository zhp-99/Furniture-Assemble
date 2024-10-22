import os
from typing import Any, Dict

task_config: Dict[str, Any] = {
    "square_table": {
        "part_names": ["square_table_top", "square_table_leg4"],
        "disassembled_pose_path": "/home/zhp/workspace/furniture-bench/workspace/tasks/relative_poses/square_table_square_table_top_square_table_leg4.npy",
        "part_frictions": [0.03, 2]
    },
    "lamp": {
        "part_names": ["lamp_base", "lamp_bulb"],
        "disassembled_pose_path": "/home/zhp/workspace/furniture-bench/workspace/tasks/relative_poses/lamp_lamp_base_lamp_bulb.npy",
        "part_frictions": [0.03, 0.5],
        "part_mass": [1, 0.5]
    },
    "desk": {
        "part_names": ["desk_top", "desk_leg4"],
        "disassembled_pose_path": "/home/zhp/workspace/furniture-bench/workspace/tasks/relative_poses/desk_desk_top_desk_leg4.npy",
        "part_frictions": [0.03, 2],
        "part_mass": [1, 0.5]
    },
    "drawer": {
        "part_names": ["drawer_box", "drawer_container_top"],
    }
}