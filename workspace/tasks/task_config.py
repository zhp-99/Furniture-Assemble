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
        "part_frictions": [0.03, 0.3]
    }
}