import numpy as np
from typing import Tuple, List, Optional, Union, overload

# Type aliases for readability
# 3x3 rotation matrix, 3D translation vector, etc.
Matrix3d = np.ndarray
Vector3d = np.ndarray
# N,3 point cloud data
PointCloud = np.ndarray

def pcd_to_feat_mat(
    pcd: PointCloud,
    voxel_num: int,
    voxel_size: float,
    ground_plane: np.ndarray
) -> np.ndarray:
    """
    Convert point cloud data to feature matrix (N, N, 9).

    Args:
        pcd: Input point cloud (N, 4).
        voxel_num: Number of voxels.
        voxel_size: Voxel size in meters.
        ground_plane: Ground plane parameters [a, b, c, d] where ax + by + cz + d = 0.
    """
    ...

def get_points_in_feat_mat(
    feat_mat: np.ndarray,
    x: int,
    y: int,
    range: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get points in feature matrix around a center (x, y).

    Returns:
        Tuple containing (min_abs_z_point, max_intensity_point).
    """
    ...

def estimate_rigid_transform(
    A: PointCloud,
    B: PointCloud
) -> Tuple[Matrix3d, Vector3d, Vector3d]:
    """
    Estimate rigid transformation using SVD.

    Returns:
        Tuple[R, t, singular_values]
    """
    ...

def get_3d_matches_with_2d_matches(
    pcd_A: PointCloud,
    pcd_B: PointCloud,
    matches_2d_A: np.ndarray,
    matches_2d_B: np.ndarray,
    voxel_num: int = 201,
    voxel_size: float = 0.4,
    search_range: int = 1,
    x_max_ind_a: int = 100,
    x_max_ind_b: int = 100,
    ground_plane_A: np.ndarray = np.array([0, 0, 1, 0]),
    ground_plane_B: np.ndarray = np.array([0, 0, 1, 0]),
    point_option: int = 0
) -> Tuple[PointCloud, PointCloud]:
    """
    Get 3D corresponding points from 2D pixel matches.

    Args:
        ground_plane_A: Ground plane for pcd_A [a, b, c, d].
        ground_plane_B: Ground plane for pcd_B [a, b, c, d].

    Returns:
        Tuple[matches_3d_A, matches_3d_B]
    """
    ...

# ---------------------------------------------------------
# GNC Localization (All Matches) Overloads
# ---------------------------------------------------------

@overload
def localization_gnc_3d_with_all_2d_matches(
    pcd_A: PointCloud,
    pcd_B: PointCloud,
    matches_2d_A_lidar: np.ndarray,
    matches_2d_B_lidar: np.ndarray,
    matches_2d_A_intensity: np.ndarray,
    matches_2d_B_intensity: np.ndarray,
    voxel_num: int = 201,
    voxel_size: float = 0.4,
    search_range: int = 1,
    x_max_ind_a: int = 100,
    x_max_ind_b: int = 100,
    gnc_max_iter: int = 100,
    gnc_threshold: float = 0.4,
    ground_plane_A: np.ndarray = np.array([0, 0, 1, 0]),
    ground_plane_B: np.ndarray = np.array([0, 0, 1, 0]),
    gnc_inlier_ratio: float = 0.5,
    profiling: bool = False
) -> Tuple[Matrix3d, Vector3d, List[int], List[np.ndarray], List[np.ndarray], float]: ...

@overload
def localization_gnc_3d_with_all_2d_matches(
    pcd_A: PointCloud,
    pcd_B: PointCloud,
    matches_2d_A_lidar: np.ndarray,
    matches_2d_B_lidar: np.ndarray,
    matches_2d_A_intensity: np.ndarray,
    matches_2d_B_intensity: np.ndarray,
    voxel_num: int = 201,
    voxel_size: float = 0.4,
    search_range: int = 1,
    x_max_ind_a: int = 100,
    x_max_ind_b: int = 100,
    gnc_max_iter: int = 100,
    gnc_threshold: float = 0.4,
    ground_plane_A: np.ndarray = np.array([0, 0, 1, 0]),
    ground_plane_B: np.ndarray = np.array([0, 0, 1, 0]),
    gnc_inlier_ratio: float = 0.5,
    profiling: bool = True
) -> Tuple[Matrix3d, Vector3d, List[int], List[np.ndarray], List[np.ndarray], float, float, float]: ...

def filter_matches(
    faiss_D: np.ndarray,
    faiss_I: np.ndarray,
    query_indices: Tuple[np.ndarray, np.ndarray],
    db_indices: Tuple[np.ndarray, np.ndarray],
    threshold: float = 0.4,
    ratio_filter: bool = False,
    ratio_filter_threshold: float = 0.8,
    filter_gnc: bool = False,
    filter_gnc_threshold: float = 0.4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter matches based on distance threshold, ratio test, or GNC.

    Args:
        query_indices: Tuple of (x_indices, y_indices) arrays.
        db_indices: Tuple of (x_indices, y_indices) arrays.

    Returns:
        Tuple[query_match_indices, db_match_indices] (Nx2 arrays)
    """
    ...

def precision_recall_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    downsample: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision-recall curve.

    Returns:
        Tuple[precision, recall, thresholds]
    """
    ...

def getBEV(
    all_points: PointCloud,
    pc_range: float = 40,
    res: float = 0.4,
    blur_size: int = 25,
    median_size: int = 5,
    high_intensity_threshold: float = 0.995
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate BEV maps from pointcloud.

    Returns:
        Tuple[global_map, intensity_map] (uint8 arrays)
    """
    ...

def voxel_filter(
    point_cloud: PointCloud,
    leaf_size: float = 0.4
) -> PointCloud:
    """
    Voxel grid filter for downsampling point cloud.
    """
    ...