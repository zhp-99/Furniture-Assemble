from isaacgym import gymapi, gymtorch
import torch
import numpy as np
import furniture_bench.controllers.control_utils as C

# Matrix Operations
def rot_mat(angles, hom: bool = False):
    """Given @angles (x, y, z), compute rotation matrix
    Args:
        angles: (x, y, z) rotation angles.
        hom: whether to return a homogeneous matrix.
    """
    x, y, z = angles
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx
    if hom:
        M = np.zeros((4, 4), dtype=np.float32)
        M[:3, :3] = R
        M[3, 3] = 1.0
        return M
    return R

def rot_mat_to_angles(R):
    """
    Given a 3x3 rotation matrix, compute the rotation angles (x, y, z).
    Args:
        R: 3x3 rotation matrix.
    Returns:
        angles: (x, y, z) rotation angles.
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-4

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def rot_mat_to_angles_tensor(R, device):
    """
    Given a 3x3 rotation matrix, compute the rotation angles (x, y, z).
    Args:
        R: 3x3 rotation matrix.
    Returns:
        angles: (x, y, z) rotation angles.
    """
    sy = torch.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-4

    if not singular:
        x = torch.atan2(R[2, 1], R[2, 2])
        y = torch.atan2(-R[2, 0], sy)
        z = torch.atan2(R[1, 0], R[0, 0])
    else:
        x = torch.atan2(-R[1, 2], R[1, 1])
        y = torch.atan2(-R[2, 0], sy)
        z = 0

    return torch.tensor([x, y, z], device=device)

def rot_mat_tensor(x, y, z, device):
    return torch.tensor(rot_mat([x, y, z], hom=True), device=device).float()

def rel_rot_mat(s, t):
    s_inv = torch.linalg.inv(s)
    return t @ s_inv

def satisfy(
    current,
    target,
    pos_error_threshold=None,
    ori_error_threshold=None,
) -> bool:
    default_pos_error_threshold = 0.01
    default_ori_error_threshold = 0.2

    if pos_error_threshold is None:
        pos_error_threshold = default_pos_error_threshold
    if ori_error_threshold is None:
        ori_error_threshold = default_ori_error_threshold

    if ((current[:3, 3] - target[:3, 3]).abs().sum() < pos_error_threshold) and (
        (target[:3, :3] - current[:3, :3]).abs().sum() < ori_error_threshold
    ):
        return True
    return False

def gripper_less(gripper_width, target_width):
    if gripper_width <= target_width:
        return True
    return False

def small_wait(env):
    for i in range(10):
        action = torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=env.device).unsqueeze(0)
        # new_action = torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=env.device).unsqueeze(0)
        # action = torch.cat((action, new_action), dim=1)

        env.step(action)

def wait(env):
    for i in range(30):
        action = torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=env.device).unsqueeze(0)
        # new_action = torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=env.device).unsqueeze(0)
        # action = torch.cat((action, new_action), dim=1)

        env.step(action)

# Control Operations
def get_action(env, start_pos, start_quat, target_pos, target_quat, gripper, slow=False):
    delta_pos = target_pos - start_pos

    # Scale translational action.
    delta_pos_sign = delta_pos.sign()
    delta_pos = torch.abs(delta_pos) * 2
    for i in range(3):
        if delta_pos[i] > 0.03:
            delta_pos[i] = 0.03 + (delta_pos[i] - 0.03) * np.random.normal(1.5, 0.1)
    delta_pos = delta_pos * delta_pos_sign

    # Clamp too large action.
    max_delta_pos = 0.11 + 0.01 * torch.rand(3, device=env.device)
    max_delta_pos[2] -= 0.04
    delta_pos = torch.clamp(delta_pos, min=-max_delta_pos, max=max_delta_pos)
    if slow:
        delta_pos = delta_pos / 2

    delta_quat = C.quat_mul(C.quat_conjugate(start_quat), target_quat)

    gripper = torch.tensor([gripper], device=env.device)
    action = torch.concat([delta_pos, delta_quat, gripper]).unsqueeze(0)
    return action

def reach_target(env, target_ee_states, thresholds, is_gripper, gripper_spend_time=10, pose_spend_time=30, slow=False):
    target_pos_1, target_quat_1, gripper_1 = target_ee_states[0]
    # target_pos_2, target_quat_2, gripper_2 = target_ee_states[1]
    pos_err_1, ori_err_1 = thresholds[0]
    pos_err_2, ori_err_2 = thresholds[1]   

    spend_time = 0

    while True:
        ee_pos_1, ee_quat_1 = env.get_ee_pose_world()
        ee_pos_1, ee_quat_1 = ee_pos_1.squeeze(), ee_quat_1.squeeze()
        action_1 = get_action(env, ee_pos_1, ee_quat_1, target_pos_1, target_quat_1, gripper_1, slow=slow)
        action = action_1

        ee_pose_1 = C.to_homogeneous(ee_pos_1, C.quat2mat(ee_quat_1))
        target_pose_1 = C.to_homogeneous(target_pos_1, C.quat2mat(target_quat_1))

        gripper_width = 1
        half_width = 0

        if is_gripper:
            if gripper_less(gripper_width, 2 * half_width + 0.001) or spend_time > gripper_spend_time:
                return True
        else:
            if satisfy(ee_pose_1, target_pose_1, pos_err_1, ori_err_1): # and satisfy(ee_pose_2, target_pose_2, pos_err_2, ori_err_2, spend_time=spend_time):
                return True
            if spend_time > pose_spend_time:
                return False

        env.step(action)
        spend_time += 1
    

# Point Cloud Operations
def depth_image_to_point_cloud_GPU(depth_image, camera_view_matrix, camera_proj_matrix, width:float, height:float, depth_bar:float=None, device:torch.device='cuda:0'):
    vinv = torch.inverse(camera_view_matrix)
    proj = camera_proj_matrix
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    centerU = width/2
    centerV = height/2

    u = torch.linspace(0, width - 1, width, device=device)
    v = torch.linspace(0, height - 1, height, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    Z = depth_image
    x_para = -Z*fu/width
    y_para = Z*fv/height
    X = (u-centerU) * x_para
    Y = (v-centerV) * y_para

    # valid = Z > -0.8
    position = torch.stack([X, Y, Z, torch.ones_like(X)], dim=-1)
    position = position.view(-1, 4)
    position = position@vinv
    points = position[:, :3]

    # normal_image = normal_image.permute(1, 2, 0).view(-1, 3)  # 将法线图像转换为 (N, 3) 形状
    # normal_image = normal_image @ vinv[:3, :3]

    x_dz = (depth_image[1:height-1, 2:width] - depth_image[1:height-1, 0:width-2])*0.5
    y_dz = (depth_image[2:height, 1:width-1] - depth_image[0:height-2, 1:width-1])*0.5
    dx = x_para
    dy = y_para
    dx = dx[1:height-1, 1:width-1]
    dy = dy[1:height-1, 1:width-1]

    normal_x = -x_dz/dx
    normal_y = -y_dz/dy
    normal_z = torch.ones((height-2, width-2), device=device)

    normal_l = torch.sqrt(normal_x*normal_x + normal_y*normal_y + normal_z*normal_z)
    normal_x = normal_x/normal_l
    normal_y = normal_y/normal_l
    normal_z = normal_z/normal_l

    normal_map = torch.stack([normal_x, normal_y, normal_z], dim=-1)
    normal_map_full = torch.zeros((height, width,3)).to(depth_image.device)
    normal_map_full[1:height-1, 1:width-1, :] = normal_map
    normals = normal_map_full.view(-1, 3)
    normals = normals @ vinv[:3, :3]

    return points, normals

def capture_pc(env, camera_handle):
    # camera_handle = env.camera_handles["front"][0]
    cam_proj = torch.tensor(env.isaac_gym.get_camera_proj_matrix(env.sim, env.envs[0], camera_handle)).to(env.device)
    cam_view = torch.tensor(env.isaac_gym.get_camera_view_matrix(env.sim, env.envs[0], camera_handle)).to(env.device)

    depth_render_type = gymapi.IMAGE_DEPTH
    depth_image = gymtorch.wrap_tensor(
                        env.isaac_gym.get_camera_image_gpu_tensor(
                            env.sim, env.envs[0], camera_handle, depth_render_type
                        )
                    )
    
    points, normals = depth_image_to_point_cloud_GPU(
        depth_image,
        cam_view,
        cam_proj,
        width=env.img_size[0],
        height=env.img_size[1],

    )

    seg_render_type = gymapi.IMAGE_SEGMENTATION
    seg_image = gymtorch.wrap_tensor(
                        env.isaac_gym.get_camera_image_gpu_tensor(
                            env.sim, env.envs[0], camera_handle, seg_render_type
                        )
                    )
    segments = seg_image.flatten()

    return points, normals, segments

def farthest_point_sample_GPU(points, npoint): 

    """
    Input:
        points: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint]
    """

    B, N, C = points.shape
    centroids = torch.zeros((B, npoint), dtype=torch.long, device=points.device)
    distance = torch.ones((B, N), device=points.device) * 1e10

    batch_indices = torch.arange(B, device=points.device)
    
    barycenter = torch.sum(points, dim=1) / N  
    barycenter = barycenter.view(B, 1, C)

    dist = torch.sum((points - barycenter) ** 2, dim=-1)  # (B,N)
    farthest = torch.argmax(dist, dim=1)  # (B)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((points - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask] 

        farthest = torch.argmax(distance, dim=1)
    # sampled_points = points[batch_indices, centroids, :]

    return centroids

def normal_to_direction(normal):

    z_axis = torch.tensor([0, 0, 1], device=normal.device, dtype=normal.dtype)
    cos_theta = torch.dot(normal, z_axis) / (torch.norm(normal) * torch.norm(z_axis))
    theta = torch.acos(cos_theta) 


    angle_with_xy_plane = np.pi / 2 - theta


    normal_xy_projection = normal.clone()
    normal_xy_projection[2] = 0


    normal_xy_projection = normal.clone()
    normal_xy_projection[2] = 0

    x_axis = torch.tensor([1, 0, 0], device=normal.device, dtype=normal.dtype)
    y_axis = torch.tensor([0, 1, 0], device=normal.device, dtype=normal.dtype)

    # 使用 atan2 计算角度
    phi = torch.atan2(torch.dot(normal_xy_projection, y_axis), torch.dot(normal_xy_projection, x_axis))


    return angle_with_xy_plane.item(), phi.item()
