import numpy as np
# import matplotlib.pyplot as plt
import cv2
import os
import argparse
import rosbag
import shutil
import time
import multiprocessing
from dataclasses import dataclass, field
# from tqdm import trange
from scipy.spatial.transform import Slerp, Rotation as R
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from tqdm import tqdm
from pcd_utils import voxel_filter, getBEV
from laz_utils import save_pcd_laz

gen_bev_params = {'pc_range':40, 'res':0.4, 'blur_size':25, 'median_size':3, 'high_intensity_threshold':0.995}


@dataclass
class DatasetConfig:
    """Configuration for processing a dataset."""
    bag_file: str
    gt_file: str
    lidar_topic: str
    B_R_L: R
    B_t_L: np.ndarray
    gt_time_col: int = 0
    gt_pos_cols: list = field(default_factory=lambda: [1, 2, 3])  # columns for position in GT
    fov_lim: int = 0
    timestamp_field: str = 't'  # 't', 'timestamp', or 'offset_time'
    timestamp_scale: float = 1e-9  # scale factor for timestamp
    timestamp_offset: float = 0.0  # offset for timestamp (e.g., -0.1 for MCD OS)
    filter_nearby: bool = False  # whether to filter points within 1.0m
    gt_sort: bool = False  # whether to sort GT by timestamp
    gt_quat_cols: list = field(default_factory=lambda: [4,5,6,7])  # columns for quaternion in GT
    pre_time_offset: float = 0.0  # additional offset for pre-filtering (e.g., +30 for Snail)
    # Umeyama alignment for ntu_night_13
    use_umeyama: bool = False  # whether to use Umeyama alignment
    gt_file_align: str = ''  # second GT file for alignment (pose_inW.csv)
    gt_time_scale: float = 1.0  # time scale for GT (e.g., 1e-9 for ntu_night_13)
    invert_pc: int = 0

class FpsTimer:
    def __init__(self, name = '', interval=100.0):
        self.interval = interval
        self.last_time = time.time()
        self.frame_count = 0
        self.name = name

    def tick(self, frame_count=1):
        self.frame_count += frame_count
        current_time = time.time()
        if current_time - self.last_time >= self.interval:
            fps = self.frame_count / (current_time - self.last_time)
            print(f"{self.name} FPS: {fps:.2f}")
            self.last_time = current_time
            self.frame_count = 0


def get_2d_convex_polygon(points):
    points_2d = points[:, :2]
    if len(points_2d) < 3: return None
    hull = ConvexHull(points_2d)
    return Polygon(points_2d[hull.vertices])

def write_task_results(write_lines, hull_txt, pose_txt):
    """Write hull and pose results to files."""
    with open(hull_txt, "a") as f:
        for hull_line, _ in write_lines:
            f.write(hull_line)
    with open(pose_txt, "a") as f:
        for _, pose_line in write_lines:
            f.write(pose_line)

def compute_pose_at_timestamp(timestamp, gt, gt_pos, slerp, gt_time_col=0):
    """
    Compute interpolated position and rotation at given timestamp.

    Returns:
        interp_rots: Rotation object
        interp_pos: Position array [1, 3]
    """
    interp_rots = slerp(timestamp)
    interp_pos = np.empty([1, gt_pos.shape[1]])
    for j in range(gt_pos.shape[1]):
        interp_pos[:, j] = np.interp(timestamp, gt[:, gt_time_col], gt_pos[:, j])
    return interp_rots, interp_pos

def transform_merged_to_body(merged_array, interp_rots, interp_pos, pc_range, fov_lim):
    """
    Transform merged world points to body frame.

    Returns:
        points: Points in body frame with intensity
    """
    last_interp_rotsT = interp_rots.inv()
    last_pos_T = -last_interp_rotsT.apply(interp_pos)

    merged_body = last_interp_rotsT.apply(merged_array[:,:3]) + last_pos_T

    if fov_lim == 1:
        points = merged_body - np.array([pc_range, 0.0, 0.0], dtype=np.float32)
    else:
        points = np.copy(merged_body)

    points = np.hstack((points, merged_array[:,3].reshape(-1,1)))
    points = points[np.where(np.abs(points[:,0])<pc_range)[0],:]
    points = points[np.where(np.abs(points[:,1])<pc_range)[0],:]
    points = points[np.where(np.abs(points[:,2])<pc_range)[0],:]
    points = points.astype(np.float32)
    return points

def compute_points_world_for_hull(points, interp_rots, interp_pos, pc_range, fov_lim):
    """
    Compute points in world frame for hull computation.
    """
    if fov_lim == 1:
        points_world = interp_rots.apply(points[:,:3] + np.array([pc_range, 0.0, 0.0], dtype=np.float32)) + interp_pos
    else:
        points_world = interp_rots.apply(points[:,:3]) + interp_pos
    return points_world

def generate_hull_line(points_world):
    """Generate hull line string from world points."""
    poly = get_2d_convex_polygon(points_world)
    return ",".join(f"{x:.6f},{y:.6f}" for x, y in poly.exterior.coords) + '\n'

def generate_pose_line(timestamp, interp_pos, interp_rots):
    """Generate pose line string."""
    pos_str = ' '.join([f'{elem:.6f}' for elem in interp_pos[0,:]])
    quat_str = ' '.join([f'{elem:.6f}' for elem in interp_rots.as_quat()])
    return str(timestamp) + ' ' + pos_str + ' ' + quat_str + '\n'

def save_bev_and_pcd(points, write_idx, img_path, img_intensity_path,
                      img_color_path, img_intensity_color_path, pcd_path):
    """
    Save BEV images and point cloud.
    """
    # Save point cloud
    ds_points = voxel_filter(points[:,:4], 0.1)
    save_pcd_laz(pcd_path + '/'+f"{write_idx:06d}.laz", ds_points)

    # Generate and save BEV images
    ds_points1 = voxel_filter(ds_points, gen_bev_params['res'])
    img, img_intensity = getBEV(ds_points1, **gen_bev_params)
    cv2.imwrite(img_path+'/'+f"{write_idx:06d}.png", img)
    cv2.imwrite(img_intensity_path+'/'+f"{write_idx:06d}.png", img_intensity)

    # Generate and save color images
    norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    gray_uint8 = norm_img.astype(np.uint8)
    color_bev = cv2.applyColorMap(gray_uint8, cv2.COLORMAP_TURBO)
    norm_img = cv2.normalize(img_intensity, None, 0, 255, cv2.NORM_MINMAX)
    gray_uint8 = norm_img.astype(np.uint8)
    color_intensity = cv2.applyColorMap(gray_uint8, cv2.COLORMAP_TURBO)
    cv2.imwrite(img_color_path+'/'+f"{write_idx:06d}.png", color_bev)
    cv2.imwrite(img_intensity_color_path+'/'+f"{write_idx:06d}.png", color_intensity)

def extract_points_from_msg(msg, timestamp, config, pc_range):
    """
    Extract points, intensity, and point timestamps from ROS message.

    Returns:
        points: (N, 3) array of xyz coordinates
        intensity: (N, 1) array of intensity values
        points_time: (N,) array of timestamps
    """
    if config.fov_lim == 1:  # Livox lidar
        points = np.array([[p.x, p.y, p.z] for p in msg.points], dtype=np.float32)
        intensity = np.array([[p.reflectivity] for p in msg.points], dtype=np.float32)
        points_time = np.array([[timestamp + p.offset_time * 1e-9] for p in msg.points], dtype=np.float64)
    else:  # Standard PointCloud2
        all_points = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"))
        all_points = [p for p in all_points]
        all_points = np.array(all_points, dtype=np.float32)
        points = all_points[:, :3]
        intensity = all_points[:, 3].reshape(-1, 1)

        # Read timestamp field
        points_time = pc2.read_points(msg, field_names=config.timestamp_field)
        points_time = np.array([p for p in points_time], dtype=np.float64 if config.timestamp_field == 'timestamp' else np.int32)
        if config.timestamp_field == 'timestamp':
            points_time = points_time * config.timestamp_scale + config.timestamp_offset
        else:
            points_time = timestamp + points_time * config.timestamp_scale + config.timestamp_offset
    points_time = points_time.reshape(-1)
    return points, intensity, points_time

def filter_points(points, intensity, points_time, pc_range, filter_nearby, fov_lim):
    """
    Filter points by range and optionally remove nearby points.

    Returns:
        filtered points_xyz, intensity, and timestamps
    """
    if fov_lim != 1:
        # Stack data: xyz + intensity + time -> (N, 5)
        points = np.hstack((points, intensity))
        points = np.hstack((points, points_time.reshape(-1, 1)))

        # Apply range filter
        points = points[np.where(np.abs(points[:, 0]) < pc_range)[0], :]
        points = points[np.where(np.abs(points[:, 1]) < pc_range)[0], :]
        points = points[np.where(np.abs(points[:, 2]) < pc_range)[0], :]

        # Apply nearby filter
        if filter_nearby:
            points = points[np.where(np.abs(points[:, 0]) > 1.0)[0], :]
            points = points[np.where(np.abs(points[:, 1]) > 1.0)[0], :]
            points = points[np.where(np.abs(points[:, 2]) > 1.0)[0], :]

        # Extract components from (N, 5) array
        intensity = points[:, 3].reshape(-1, 1)
        points_time = points[:, 4]
        points_xyz = points[:, :3]
    else:
        # For fov_lim == 1: no range filter
        # Apply nearby filter if needed
        if filter_nearby:
            mask = (np.abs(points[:, 0]) > 1.0) & (np.abs(points[:, 1]) > 1.0) & (np.abs(points[:, 2]) > 1.0)
            points_xyz = points[mask]
            intensity = intensity[mask]
            points_time = points_time[mask]
        else:
            points_xyz = points

    return points_xyz, intensity, points_time

def apply_motion_compensation(points, intensity, points_time, config, gt, gt_pos, slerp):
    """
    Apply motion compensation to transform points to world frame.

    Returns:
        points_world: (N, 4) array of points in world frame with intensity
    """
    # Transform to body frame using extrinsics
    points = config.B_R_L.apply(points) + config.B_t_L

    # Interpolate rotation and position
    interp_rots = slerp(points_time).as_matrix()
    interp_pos = np.empty([points.shape[0], gt_pos.shape[1]])
    for j in range(gt_pos.shape[1]):
        interp_pos[:, j] = np.interp(points_time, gt[:, config.gt_time_col], gt_pos[:, j])

    # Transform to world frame
    points_rotated = np.einsum('nij,nj->ni', interp_rots, points[:, :3])
    points_world = points_rotated + interp_pos
    points_world = np.hstack((points_world, intensity))

    return points_world

def common_task_run(task):
    """
    Common task run function for all datasets.
    """
    merged_array = task['merged_array']
    timestamp = task['timestamp']
    write_idx = task['write_idx']
    img_path = task['img_path']
    img_intensity_path = task['img_intensity_path']
    img_color_path = task['img_color_path']
    img_intensity_color_path = task['img_intensity_color_path']
    pcd_path = task['pcd_path']
    gt_pos = task['gt_pos']
    slerp = task['slerp']
    gt = task['gt']
    pc_range = task['pc_range']
    fov_lim = task.get('fov_lim', 0)
    gt_time_col = task.get('gt_time_col', 0)
    invert_pc = task.get('invert_pc', 0)

    # Compute pose at timestamp
    interp_rots, interp_pos = compute_pose_at_timestamp(timestamp, gt, gt_pos, slerp, gt_time_col)
    if invert_pc:
        b_R_L = R.from_euler('x', np.deg2rad(180)).as_matrix()
        interp_rots = interp_rots.from_matrix(interp_rots.as_matrix()@b_R_L)

    # Transform to body frame
    points = transform_merged_to_body(merged_array, interp_rots, interp_pos, pc_range, fov_lim)

    # Compute points in world for hull
    points_world = compute_points_world_for_hull(points, interp_rots, interp_pos, pc_range, fov_lim)

    # Generate hull line
    hull_line = generate_hull_line(points_world)

    # Save BEV and point cloud
    save_bev_and_pcd(points, write_idx, img_path, img_intensity_path,
                     img_color_path, img_intensity_color_path, pcd_path)

    # Generate pose line
    pose_line = generate_pose_line(timestamp, interp_pos, interp_rots)

    return hull_line, pose_line


fps_timer = FpsTimer(name='BEV-Gen', interval=50.0)

def create_output_paths(bev_save_path, bag_seq):
    """
    Create output directories and file paths for BEV generation.

    Args:
        bev_save_path: Base path for saving BEV images
        bag_seq: Bag sequence name

    Returns:
        dict: Dictionary containing all paths
    """
    # Create base path if not exists
    if not os.path.exists(bev_save_path): os.makedirs(bev_save_path)

    # Create pose path
    pose_path = bev_save_path + "/poses/"
    if not os.path.exists(pose_path): os.makedirs(pose_path)
    pose_txt = pose_path + bag_seq + '.txt'
    if os.path.exists(pose_txt): os.remove(pose_txt)

    # Create hull path
    hull_path = bev_save_path + "/hull/"
    if not os.path.exists(hull_path): os.makedirs(hull_path)
    hull_txt = hull_path + bag_seq + '.txt'
    if os.path.exists(hull_txt): os.remove(hull_txt)

    # Create image paths
    img_path = bev_save_path + '/' + bag_seq + '/bev_imgs'
    if os.path.exists(img_path): shutil.rmtree(img_path)
    os.makedirs(img_path, exist_ok=True)

    img_intensity_path = bev_save_path + '/' + bag_seq + '/intensity_imgs'
    if os.path.exists(img_intensity_path): shutil.rmtree(img_intensity_path)
    os.makedirs(img_intensity_path, exist_ok=True)

    img_color_path = bev_save_path + '/' + bag_seq + '/color_imgs'
    if os.path.exists(img_color_path): shutil.rmtree(img_color_path)
    os.makedirs(img_color_path, exist_ok=True)

    img_intensity_color_path = bev_save_path + '/' + bag_seq + '/color_intensity_imgs'
    if os.path.exists(img_intensity_color_path): shutil.rmtree(img_intensity_color_path)
    os.makedirs(img_intensity_color_path, exist_ok=True)

    # Create point cloud path
    pcd_path = bev_save_path + '/' + bag_seq + '/laz'
    if os.path.exists(pcd_path): shutil.rmtree(pcd_path)
    os.makedirs(pcd_path, exist_ok=True)

    return {
        'pose_txt': pose_txt,
        'hull_txt': hull_txt,
        'img_path': img_path,
        'img_intensity_path': img_intensity_path,
        'img_color_path': img_color_path,
        'img_intensity_color_path': img_intensity_color_path,
        'pcd_path': pcd_path,
    }


def get_mcd_config(args, bag_seq):
    """Get configuration for MCD dataset."""
    if args.fov_lim == 1:  # mid-70
        bag_file = args.vel_path + '/' + bag_seq + '/' + bag_seq + '_mid70.bag'
        B_t_L = np.array([-0.01051484, -0.00898978, 0.03735647])
        B_R_L = R.from_quat([9.99963214e-01, -9.92370755e-05, -8.42165954e-03, -1.62395386e-03])
        lidar_topic = '/livox/lidar'
        timestamp_field = 'offset_time'
    else:  # os1-128
        bag_file = args.vel_path + '/' + bag_seq + '/' + bag_seq + '_os1_128.bag'
        B_T_L = np.array([0.999934655205122900, 0.003477624535771754, -0.010889970036688295, -0.060649229060416594,
                          0.003587143302461965, -0.999943027982117100, 0.010053516443599904, -0.012837544242408117,
                          -0.010854387257665576, -0.010091923381711220, -0.999890161647627000, -0.020492606896077407,
                          0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 1.000000000000000000]).reshape(4, 4)
        B_t_L = B_T_L[0:3, 3]
        B_R_L = R.from_matrix(B_T_L[:3, :3])
        lidar_topic = '/os_cloud_node/points'
        timestamp_field = 't'

    if bag_seq == 'ntu_night_13':
        # Use odometry.csv with Umeyama alignment to pose_inW.csv
        gt_file = args.vel_path + '/mcd_gt/' + bag_seq + '/odometry.csv'
        gt_file_align = args.vel_path + '/mcd_gt/' + bag_seq + '/pose_inW.csv'
        gt_time_col = 0
        gt_quat_cols = [6, 7, 8, 9]  # odometry.csv: quat at columns 6,7,8,9
        gt_pos_cols = [3, 4, 5]  # odometry.csv: x,y,z at columns 3,4,5
        use_umeyama = True
        gt_time_scale = 1e-9
    else:
        gt_file = args.vel_path + '/mcd_gt/' + bag_seq + '/pose_inW.csv'
        gt_file_align = ''
        gt_time_col = 1
        gt_quat_cols = [5, 6, 7, 8]
        gt_pos_cols = [2, 3, 4]  # pose_inW.csv: x,y,z at columns 2,3,4
        use_umeyama = False
        gt_time_scale = 1.0

    return DatasetConfig(
        bag_file=bag_file,
        gt_file=gt_file,
        lidar_topic=lidar_topic,
        B_R_L=B_R_L,
        B_t_L=B_t_L,
        gt_time_col=gt_time_col,
        fov_lim=args.fov_lim,
        timestamp_field=timestamp_field,
        timestamp_scale=1e-9,
        timestamp_offset=-0.1 if args.fov_lim != 1 else 0.0,
        filter_nearby=False,
        gt_quat_cols=gt_quat_cols,
        gt_pos_cols=gt_pos_cols,
        use_umeyama=use_umeyama,
        gt_file_align=gt_file_align,
        gt_time_scale=gt_time_scale,
        invert_pc=1,
    )

def get_snail_config(args, bag_seq):
    """Get configuration for Snail dataset."""
    bag_seq_area = bag_seq.split("-")[0]
    bag_seq_date = bag_seq.split("-")[1]
    bag_seq_num = bag_seq.split("-")[2]
    bag_file = args.vel_path + '/' + bag_seq_area + '/' + bag_seq + '.bag'
    gt_file = args.vel_path + '/full_trajs/' + bag_seq_date + '/' + bag_seq_num + '/utm50r_T_xt32.txt'

    return DatasetConfig(
        bag_file=bag_file,
        gt_file=gt_file,
        lidar_topic='/hesai/pandar',
        B_R_L=R.from_matrix(np.eye(3)),
        B_t_L=np.array([0.0, 0.0, 0.0]),
        gt_time_col=0,
        fov_lim=0,
        timestamp_field='timestamp',
        timestamp_scale=1.0,
        timestamp_offset=0.0,
        filter_nearby=False,
        pre_time_offset=30.0,  # Skip first 30 seconds
    )

def get_hunter_config(args, bag_seq):
    """Get configuration for Hunter dataset."""
    bag_file = args.vel_path + '/' + bag_seq + '.bag'
    gt_file = args.vel_path + '/Pose-' + bag_seq + '.txt'

    if args.fov_lim == 1:
        livox_T_rs = np.array([0.999987,-0.00467039,-0.0021162,-0.0867252,
                                0.00470885,0.999817,0.018556,-0.0138852,
                                0.00202915,-0.0185657,0.999827,0.218342,
                                -0,-0,-0,1]).reshape(4,4)
        rs_T_livox = np.linalg.inv(livox_T_rs)
        B_R_L = R.from_matrix(rs_T_livox[:3, :3])
        B_t_L = rs_T_livox[:3, 3]
        lidar_topic = '/livox/lidar'
        timestamp_field = 'offset_time'
    else:
        B_R_L = R.from_matrix(np.eye(3))
        B_t_L = np.array([0.0, 0.0, 0.0])
        lidar_topic = '/rslidar_points'
        timestamp_field = 'timestamp'

    return DatasetConfig(
        bag_file=bag_file,
        gt_file=gt_file,
        lidar_topic=lidar_topic,
        B_R_L=B_R_L,
        B_t_L=B_t_L,
        gt_time_col=0,
        fov_lim=args.fov_lim,
        timestamp_field=timestamp_field,
        timestamp_scale=1.0,
        filter_nearby=True,
    )

def get_husky_config(args, bag_seq):
    """Get configuration for Husky dataset."""
    bag_file = args.vel_path + '/' + bag_seq + '.bag'
    gt_file = args.vel_path + '/Pose-' + bag_seq + '.txt'

    return DatasetConfig(
        bag_file=bag_file,
        gt_file=gt_file,
        lidar_topic='/ouster/points',
        B_R_L=R.from_matrix(np.eye(3)),
        B_t_L=np.array([0.0, 0.0, 0.0]),
        gt_time_col=0,
        fov_lim=0,
        timestamp_field='t',
        timestamp_scale=1e-9,
        filter_nearby=True,
        gt_sort=True,
    )

def load_ground_truth(config):
    """Load and process ground truth data."""
    delimiter = ',' if config.gt_file.endswith('.csv') else ' '
    skiprows = 1 if config.gt_file.endswith('.csv') else 0

    gt = np.loadtxt(config.gt_file, delimiter=delimiter, skiprows=skiprows)

    if config.gt_sort:
        gt = gt[gt[:, config.gt_time_col].argsort()]
        _, idx = np.unique(gt[:, config.gt_time_col], return_index=True)
        gt = gt[idx]

    # Apply time scale
    if hasattr(config, 'gt_time_scale') and config.gt_time_scale != 1.0:
        gt[:, config.gt_time_col] = gt[:, config.gt_time_col] * config.gt_time_scale

    gt_pos = gt[:, config.gt_pos_cols]

    # Umeyama alignment for ntu_night_13
    if hasattr(config, 'use_umeyama') and config.use_umeyama and config.gt_file_align:
        from pcd_utils import estimate_rigid_transform
        gt_raw = np.loadtxt(config.gt_file_align, delimiter=',', skiprows=1)
        # Filter gt_raw to match gt time range
        gt_raw = gt_raw[(gt_raw[:, 1] > np.min(gt[:, config.gt_time_col])) &
                        (gt_raw[:, 1] < np.max(gt[:, config.gt_time_col]))]
        # Interpolate gt_pos to gt_raw timestamps for alignment
        interp_pos = np.empty([gt_raw.shape[0], gt_pos.shape[1]])
        for j in range(gt_pos.shape[1]):
            interp_pos[:, j] = np.interp(gt_raw[:, 1], gt[:, config.gt_time_col], gt_pos[:, j])
        # Compute Umeyama alignment using pcd_utils (A=interp_pos, B=gt_raw[:, 2:5])
        R_align, t_align, _ = estimate_rigid_transform(interp_pos, gt_raw[:, 2:5])
        # Apply alignment to gt_pos: B = R @ A + t
        gt_pos = (R_align @ gt_pos.T).T + t_align
        # Create aligned rotations
        gt_rots = R.from_matrix(R_align) * R.from_quat(gt[:, config.gt_quat_cols])
        slerp = Slerp(gt[:, config.gt_time_col], gt_rots)
    else:
        slerp = Slerp(gt[:, config.gt_time_col], R.from_quat(gt[:, config.gt_quat_cols]))

    return gt, gt_pos, slerp

def process_bag_dataset(args, bag_seq, paths, config):
    """
    Unified function to process bag dataset for BEV generation.
    """
    pose_txt = paths['pose_txt']
    hull_txt = paths['hull_txt']
    img_path = paths['img_path']
    img_intensity_path = paths['img_intensity_path']
    img_color_path = paths['img_color_path']
    img_intensity_color_path = paths['img_intensity_color_path']
    pcd_path = paths['pcd_path']

    bag = rosbag.Bag(config.bag_file, 'r')
    gt, gt_pos, slerp = load_ground_truth(config)

    pc_range = gen_bev_params['pc_range'] 
    write_idx = 0
    merged_map = []
    task_pool = []
    task_batch = max(os.cpu_count()-1, 1)
    pool = multiprocessing.Pool(processes=task_batch)

    total_msgs = bag.get_message_count(topic_filters=[config.lidar_topic])
    for topic, msg, t in tqdm(bag.read_messages(topics=[config.lidar_topic]), total=total_msgs, desc=config.lidar_topic):
        timestamp = msg.header.stamp.to_sec()
        # Time filtering
        if timestamp < gt[0, config.gt_time_col] + config.pre_time_offset: continue
        if timestamp < gt[0, config.gt_time_col] or timestamp > gt[-1, config.gt_time_col]: continue
        # Extract points from message
        points, intensity, points_time = extract_points_from_msg(msg, timestamp, config, pc_range)

        # Time filtering on points
        # print(np.min(points_time), np.max(points_time), gt[0, config.gt_time_col], gt[-1, config.gt_time_col])
        if np.max(points_time) > gt[-1, config.gt_time_col] or np.min(points_time) < gt[0, config.gt_time_col]: continue

        # Filter points
        points, intensity, points_time = filter_points(points, intensity, points_time, pc_range, config.filter_nearby, config.fov_lim)

        # Apply motion compensation
        points_world = apply_motion_compensation(points, intensity, points_time, config, gt, gt_pos, slerp)

        merged_map.append(points_world)
        merged_array = np.vstack(merged_map)
        if len(merged_map) < args.bach_size: continue

        task_now = {
            'merged_array': merged_array, 'timestamp': timestamp,
            'write_idx': write_idx, 'img_path': img_path, 'pc_range': pc_range,
            'img_intensity_path': img_intensity_path, 'img_color_path': img_color_path,
            'img_intensity_color_path': img_intensity_color_path, 'pcd_path': pcd_path,
            'gt_pos': gt_pos, 'slerp': slerp, 'gt': gt, 'invert_pc': config.invert_pc,
            'fov_lim': config.fov_lim, 'gt_time_col': config.gt_time_col
        }
        task_pool.append(task_now)
        write_idx += 1
        merged_map.pop(0)
        if len(task_pool) <= task_batch: continue

        write_lines = pool.map(common_task_run, task_pool)
        write_task_results(write_lines, hull_txt, pose_txt)
        task_pool.clear()
        fps_timer.tick(frame_count=task_batch)

    if len(task_pool) > 0:
        write_lines = pool.map(common_task_run, task_pool)
        write_task_results(write_lines, hull_txt, pose_txt)

    pool.close()
    pool.join()


def process_mcd_dataset(args, bag_seq, paths):
    """
    Process MCD dataset for BEV generation.

    Args:
        args: Command line arguments
        bag_seq: Bag sequence name
        paths: Dictionary containing output paths
    """
    config = get_mcd_config(args, bag_seq)
    process_bag_dataset(args, bag_seq, paths, config)


def process_snail_dataset(args, bag_seq, paths):
    """
    Process Snail dataset for BEV generation.

    Args:
        args: Command line arguments
        bag_seq: Bag sequence name
        paths: Dictionary containing output paths
    """
    config = get_snail_config(args, bag_seq)
    process_bag_dataset(args, bag_seq, paths, config)


def process_hunter_dataset(args, bag_seq, paths):
    """
    Process Hunter dataset for BEV generation.

    Args:
        args: Command line arguments
        bag_seq: Bag sequence name
        paths: Dictionary containing output paths
    """
    config = get_hunter_config(args, bag_seq)
    process_bag_dataset(args, bag_seq, paths, config)


def process_husky_dataset(args, bag_seq, paths):
    """
    Process Husky dataset for BEV generation.

    Args:
        args: Command line arguments
        bag_seq: Bag sequence name
        paths: Dictionary containing output paths
    """
    config = get_husky_config(args, bag_seq)
    process_bag_dataset(args, bag_seq, paths, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UniLGL-Gen-BEV-Images')
    parser.add_argument('--vel_path', type=str, default="/mnt/share_disk/KITTI/dataset/sequences/00/velodyne/", help='path to data')
    parser.add_argument('--bev_save_path', type=str, default="./KITTI_new_imgs/00/imgs/", help='path to data')
    parser.add_argument('--dataset', type=str, default="MCD", help='MCD/KITTI360')
    parser.add_argument('--seq', type=str, default="ntu_night_13", help='ntu_night_13/2013_05_28_drive_0000_sync')
    parser.add_argument('--fov_lim', type=int, default=0, help='livox/os')
    parser.add_argument('--bach_size', type=int, default=1, help='livox/os')

    args = parser.parse_args()
    bag_seq = args.seq
    paths = create_output_paths(args.bev_save_path, bag_seq)
    pose_txt = paths['pose_txt']
    hull_txt = paths['hull_txt']
    img_path = paths['img_path']
    img_intensity_path = paths['img_intensity_path']
    img_color_path = paths['img_color_path']
    img_intensity_color_path = paths['img_intensity_color_path']
    pcd_path = paths['pcd_path']

    if args.dataset == 'MCD':
        process_mcd_dataset(args, bag_seq, paths)

    elif args.dataset == 'Snail':
        process_snail_dataset(args, bag_seq, paths)

    elif args.dataset == 'Hunter':
        process_hunter_dataset(args, bag_seq, paths)

    elif args.dataset == 'Husky':
        process_husky_dataset(args, bag_seq, paths)
