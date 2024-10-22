from tasks.utils import *
import numpy as np

target_ori_1 = rot_mat([np.pi, 0, -np.pi / 2 - np.pi / 36], "cuda")[:3, :3]
print(rot_mat_to_angles(target_ori_1)/np.pi)

mat_1 = np.array(
    [[ 0.6413,  0.0102,  0.7672,  0.3185],
        [ 0.7672, -0.0164, -0.6412,  0.1191],
        [ 0.0060,  0.9998, -0.0184,  0.5391],
        [ 0.0000,  0.0000,  0.0000,  1.0000]]
)
mat_2 = np.array(
    [[ 0.6421,  0.0110,  0.7665,  0.3186],
        [ 0.7666, -0.0153, -0.6420,  0.1191],
        [ 0.0047,  0.9998, -0.0182,  0.5391],
        [ 0.0000,  0.0000,  0.0000,  1.0000]]
)

print(rot_mat_to_angles(mat_1[:3, :3])/np.pi)
print(rot_mat_to_angles(mat_2[:3, :3])/np.pi)