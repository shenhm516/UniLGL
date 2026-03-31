import sys
sys.path.append('.')
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from pcd_utils import filter_matches
import faiss
import faiss.contrib.torch_utils
import dataset.infer_datasets as infer_datasets
import time

# Patchwork++ ground segmentation for ground level estimation
_patchwork_instance = None
_ground_timing_stats = {'patchwork': [], 'svd': [], 'total': []}

def get_patchwork_instance():
    """Get or create a singleton Patchwork++ instance."""
    global _patchwork_instance
    if _patchwork_instance is None:
        import pypatchworkpp
        params = pypatchworkpp.Parameters()
        params.verbose = False
        _patchwork_instance = pypatchworkpp.patchworkpp(params)
    return _patchwork_instance

def fit_plane_svd(points, max_points=5000, max_range=15.0):
    """
    Fit a plane to points using SVD.

    Args:
        points: (N, 3) array of 3D points
        max_points: Maximum number of points to use for SVD (subsampling for speed)
        max_range: Maximum xy distance from origin to include (meters)

    Returns:
        centroid: (3,) array, centroid point on the plane
        normal: (3,) array, plane normal (pointing upward, z > 0)
    """
    if len(points) < 3:
        return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])

    # Filter points within max_range (xy distance from origin)
    xy_dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    mask = xy_dist <= max_range
    filtered_points = points[mask]

    if len(filtered_points) < 3:
        filtered_points = points  # Fallback to all points if too few after filtering

    # Subsample if too many points (SVD is O(n^3) for full matrix)
    if len(filtered_points) > max_points:
        indices = np.random.choice(len(filtered_points), max_points, replace=False)
        filtered_points = filtered_points[indices]

    # Center the points
    centroid = np.mean(filtered_points, axis=0)
    centered = filtered_points - centroid

    # SVD to find the plane normal (smallest singular value)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]  # Normal is the last row of Vt

    # Ensure normal points upward (z > 0)
    if normal[2] < 0:
        normal = -normal

    return centroid, normal

def estimate_ground_level_patchwork(points, use_median=True, fit_plane=False, verbose_timing=False):
    """
    Estimate ground level using Patchwork++ ground segmentation.

    Args:
        points: (N, 4) or (N, 3) point cloud array [x, y, z, intensity] or [x, y, z]
        use_median: If True, use median instead of mean for ground level estimation
        fit_plane: If True, also fit a plane to ground points
        verbose_timing: If True, print timing information

    Returns:
        ground_level: Estimated ground level (z-coordinate at centroid)
        ground_points: Ground point cloud (N_ground, 3)
        (centroid, normal): (optional, if fit_plane=True) plane centroid and normal
    """
    t_start = time.time()
    patchwork = get_patchwork_instance()

    # Prepare point cloud with intensity (required for RNR)
    if points.shape[1] >= 4:
        # Use full point cloud with intensity for RNR
        cloud = points[:, :4].astype(np.float32)
    elif points.shape[1] == 3:
        cloud = points[:, :3].astype(np.float32)
    else:
        raise ValueError(f"Points must have at least 3 columns, got {points.shape[1]}")

    # Run Patchwork++ ground segmentation
    t_patchwork_start = time.time()
    patchwork.estimateGround(cloud)
    ground_points = patchwork.getGround()
    t_patchwork = time.time() - t_patchwork_start

    if ground_points.shape[0] == 0:
        # No ground points found, fallback to minimum z
        result = [np.min(points[:, 2]), np.zeros((0, 3))]
        if fit_plane:
            result.append((np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])))
        return tuple(result)

    # Estimate ground level from ground points
    z_values = ground_points[:, 2]
    if use_median:
        ground_level = np.median(z_values)
    else:
        ground_level = np.mean(z_values)

    result = [ground_level, ground_points]

    t_svd = 0.0
    if fit_plane:
        t_svd_start = time.time()
        centroid, normal = fit_plane_svd(ground_points, max_range=10)
        t_svd = time.time() - t_svd_start
        result.append((centroid, normal))

    t_total = time.time() - t_start

    # Record timing stats
    _ground_timing_stats['patchwork'].append(t_patchwork * 1000)
    _ground_timing_stats['svd'].append(t_svd * 1000)
    _ground_timing_stats['total'].append(t_total * 1000)

    if verbose_timing:
        print(f"[Ground Estimation] Patchwork++: {t_patchwork*1000:.2f}ms, SVD: {t_svd*1000:.2f}ms, Total: {t_total*1000:.2f}ms")

    return tuple(result)

def estimate_ground_plane_patchwork(points, verbose_timing=False):
    """
    Estimate ground plane using Patchwork++ ground segmentation.

    Args:
        points: (N, 4) or (N, 3) point cloud array
        verbose_timing: If True, print timing information

    Returns:
        (centroid, normal): plane centroid (3,) and normal (3,) pointing upward
        ground_points: Ground point cloud (N_ground, 3)
    """
    ground_level, ground_points, (centroid, normal) = estimate_ground_level_patchwork(
        points, use_median=True, fit_plane=True, verbose_timing=verbose_timing
    )
    return (centroid, normal), ground_points

def pose_to_matrix(pose):
    ts, x, y, z, qx, qy, qz, qw = pose
    rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = [x, y, z]
    # print(mat, x, y, z)
    # print(x, y, z, qw, qx, qy, qz)
    return mat

def pose_to_matrix_2d(pose):
    ts, x, y, z, qx, qy, qz, qw = pose
    rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
    mat = np.eye(3)
    mat[:2, :2] = rot[:2, :2]
    mat[:2, 2] = [x, y]
    # print(mat, x, y)
    return mat



def interpolate(token_map):
    # check if token_map is numpy array
    with torch.no_grad():
        if isinstance(token_map, np.ndarray):
            token_map = torch.from_numpy(token_map).float().to('cuda' if torch.cuda.is_available() else 'cpu')
        
        B = token_map.shape[0]
        aff = torch.zeros(B, 2, 3).to(token_map.device)
        aff[:, 0, 0]=1
        aff[:, 0, 1]=0
        aff[:, 1, 0]=0
        aff[:, 1, 1]=1
        
        grid = F.affine_grid(aff, torch.Size((B, token_map.shape[1], 201, 201)), align_corners=True).type(token_map.type())
        token_map = F.grid_sample(token_map, grid, align_corners=True, mode='bicubic')
        token_map = F.normalize(token_map, dim=1)
    return token_map

def draw_heatmap_token(token_map, caption=None):
    # average across the channel dimension
    token_map = token_map.mean(axis=2)

    # Create the heatmap
    plt.figure(figsize=(5, 5))
    plt.imshow(token_map, cmap='viridis', interpolation='nearest')
    plt.axis('off')  # Hide axes
    # if caption:
    #     plt.title(caption)
    # else:
    #     plt.title("Token Map Heatmap")
    # plt.colorbar()  # Optional: add a color bar
    plt.savefig(f"match_results/{caption}.png", bbox_inches='tight', pad_inches=0.0, dpi=600)
    return plt

def mat_coord_to_body(mat_coords, x_max_ind):
    xcs = mat_coords[:, 0]
    ycs = mat_coords[:, 1]
    
    xcps = (x_max_ind - xcs) * 0.4
    ycps = (100 - ycs) * 0.4
    body_coords = np.stack((xcps, ycps), axis=1)
    return body_coords

def do_feature_matching(
    query_img,
    db_img,
    query_local_feat,
    db_local_feat,
    match_threshold=0.4,
    caption='lidar',
    use_fast=True,
    faiss_gpu=False,
    ratio_filter=False,
    ratio_filter_thres=0.8,
    gnc_filter=False,
    gnc_filter_thres=1.0,
    vis=False):
    
    if vis:
        draw_heatmap_token(query_local_feat, caption=f'query_local_feat_{caption}')
        draw_heatmap_token(db_local_feat, caption=f'db_local_feat_{caption}')
        cv2.imwrite(f'./match_results/query_img_{caption}.png', query_img)
        cv2.imwrite(f'./match_results/db_img_{caption}.png', db_img)
    
    if use_fast:
        fast = cv2.FastFeatureDetector_create()
        query_keypoints = fast.detect(query_img, None)
        db_keypoints = fast.detect(db_img, None)
        query_local_pos = (np.array([kp.pt[1] for kp in query_keypoints], dtype=int), np.array([kp.pt[0] for kp in query_keypoints], dtype=int))
        db_local_pos = (np.array([kp.pt[1] for kp in db_keypoints], dtype=int), np.array([kp.pt[0] for kp in db_keypoints], dtype=int))
    else:
        db_local_pos = np.where(db_img > 0)
        query_local_pos = np.where(query_img > 0)
    # print(db_local_feat.shape)
    # tt = db_local_feat[db_local_pos]
    # print(tt.shape)
    num_k = 2 if ratio_filter else 1
    if faiss_gpu:
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        local_faiss_index = faiss.GpuIndexFlatL2(res, db_local_feat.shape[2], flat_config)
        local_faiss_index.add(db_local_feat[db_local_pos])
        D, I = local_faiss_index.search(query_local_feat[query_local_pos], num_k)
    else:
        local_faiss_index = faiss.IndexFlatL2(db_local_feat.shape[2])
        local_faiss_index.add(db_local_feat[db_local_pos])
        D, I = local_faiss_index.search(query_local_feat[query_local_pos], num_k)
    
    # t0 = time.time()

    query_local_match_coords, db_local_match_coords = filter_matches(
        D, I, query_local_pos, db_local_pos, match_threshold, ratio_filter, ratio_filter_thres, gnc_filter, gnc_filter_thres
    )

    # t1 = time.time()
    # print(f"Feature matching took {(t1 - t0)*1000:.3f} ms for {len(query_local_match_coords)} matches")
    
    if query_local_match_coords.shape[0] == 0:
        query_local_match_coords = np.zeros((0, 2))
        db_local_match_coords = np.zeros((0, 2))
    
    # draw keypoints on the images
    if vis:
        query_keypoints = [cv2.KeyPoint(float(coord[1]), float(coord[0]), 1.0) for coord in query_local_match_coords]
        db_keypoints = [cv2.KeyPoint(float(coord[1]), float(coord[0]), 1.0) for coord in db_local_match_coords]
        assert len(query_keypoints) == len(db_keypoints), "Keypoints length mismatch"
        matches = [cv2.DMatch(i, i, 0) for i in range(len(query_keypoints))]
        match_img = cv2.drawMatches(query_img, query_keypoints, db_img, db_keypoints, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(f'./match_results/match_test_{caption}.png', match_img)
    
    return query_local_match_coords, db_local_match_coords


def do_feature_matching_faiss_gpu(
    query_img:Tensor,
    db_img:Tensor,
    query_local_feat:Tensor,
    db_local_feat:Tensor,
    match_threshold=0.4,
    caption='lidar',
    filter_cpp=False,
    ratio_filter=False,
    ratio_filter_thres=0.8,
    gnc_filter=False,
    gnc_filter_thres=1.0,
    prealloc_index=None):
    
    # if vis:
    #     draw_heatmap_token(query_local_feat.cpu().numpy(), caption=f'query_local_feat_{caption}')
    #     draw_heatmap_token(db_local_feat.cpu().numpy(), caption=f'db_local_feat_{caption}')
    #     cv2.imwrite(f'./match_results/query_img_{caption}.png', query_img.cpu().numpy())
    #     cv2.imwrite(f'./match_results/db_img_{caption}.png', db_img.cpu().numpy())
    
    db_local_pos = torch.nonzero(db_img > 0, as_tuple=True)
    query_local_pos = torch.nonzero(query_img > 0, as_tuple=True)
    # print(db_local_feat.shape)
    # tt = db_local_feat[db_local_pos]
    # print(tt.shape)
    num_k = 2 if ratio_filter else 1
    
    local_faiss_index = prealloc_index
    if local_faiss_index is None:
        res = faiss.StandardGpuResources()
        res.setTempMemory(512 * 1024 * 1024)  # 512 MB
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        local_faiss_index = faiss.GpuIndexFlatL2(res, db_local_feat.shape[2], flat_config)

    local_faiss_index.add(db_local_feat[db_local_pos])
    D, I = local_faiss_index.search(query_local_feat[query_local_pos], num_k)
    
    db_local_pos = (db_local_pos[0].cpu().numpy(), db_local_pos[1].cpu().numpy())
    query_local_pos = (query_local_pos[0].cpu().numpy(), query_local_pos[1].cpu().numpy())
    D = D.cpu().numpy()
    I = I.cpu().numpy()
    local_faiss_index.reset()
    # t0 = time.time()

    query_local_match_coords, db_local_match_coords = filter_matches(
        D, I, query_local_pos, db_local_pos, match_threshold, ratio_filter, ratio_filter_thres, gnc_filter, gnc_filter_thres
    )


    # t1 = time.time()
    # print(f"Feature matching took {(t1 - t0)*1000:.3f} ms for {len(query_local_match_coords)} matches")
    
    if query_local_match_coords.shape[0] == 0:
        query_local_match_coords = np.zeros((0, 2))
        db_local_match_coords = np.zeros((0, 2))
    
    # draw keypoints on the images
    # if vis:
    #     query_keypoints = [cv2.KeyPoint(float(coord[1]), float(coord[0]), 1.0) for coord in query_local_match_coords]
    #     db_keypoints = [cv2.KeyPoint(float(coord[1]), float(coord[0]), 1.0) for coord in db_local_match_coords]
    #     assert len(query_keypoints) == len(db_keypoints), "Keypoints length mismatch"
    #     matches = [cv2.DMatch(i, i, 0) for i in range(len(query_keypoints))]
    #     match_img = cv2.drawMatches(query_img, query_keypoints, db_img, db_keypoints, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    #     cv2.imwrite(f'./match_results/match_test_{caption}.png', match_img)
    
    return query_local_match_coords, db_local_match_coords, local_faiss_index

