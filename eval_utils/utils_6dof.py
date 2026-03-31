import sys
sys.path.append('.')
import torch
import numpy as np
from tqdm import tqdm
import os
import gzip
import pickle
import time
import dataset.infer_datasets as infer_datasets
from scipy import sparse
from scipy.spatial.transform import Rotation as R
from eval_utils.utils import pose_to_matrix, interpolate, do_feature_matching_faiss_gpu
from pcd_utils import localization_gnc_3d_with_all_2d_matches
# numpy output precision
np.set_printoptions(precision=3, suppress=True, linewidth=200)

# Point Option: 0 - use min_abs_z, 1 - use max_intensity
def do_localization_all_gnc_faiss_gpu(query_img, db_img, query_pcd, db_pcd, query_local_feat, db_local_feat, x_max_ind_query=100, x_max_ind_db=100, threshold=0.4, return_inliers = False, ground_centroid_query=None, ground_normal_query=None, ground_centroid_db=None, ground_normal_db=None, use_patchwork=True, gpu_index_handle=None, profiling=False):
    """
    Ground localization with optional Patchwork++ ground plane estimation.

    Args:
        ground_centroid_query: (3,) array, centroid point on the ground plane.
        ground_normal_query: (3,) array, normal vector of the ground plane.
        ground_centroid_db: Same for database point cloud.
        ground_normal_db: Same for database point cloud.
        use_patchwork: Whether to use Patchwork++ for ground plane estimation.
    """
    from eval_utils.utils import estimate_ground_plane_patchwork

    time_doloc = time.time()

    # Estimate ground plane using Patchwork++ if not provided
    if (ground_centroid_query is None or ground_normal_query is None) and use_patchwork:
        (ground_centroid_query, ground_normal_query), _ = estimate_ground_plane_patchwork(query_pcd, verbose_timing=False)
    elif ground_centroid_query is None:
        ground_centroid_query = np.array([0.0, 0.0, 0.0])
        ground_normal_query = np.array([0.0, 0.0, 1.0])

    if (ground_centroid_db is None or ground_normal_db is None) and use_patchwork:
        (ground_centroid_db, ground_normal_db), _ = estimate_ground_plane_patchwork(db_pcd, verbose_timing=False)
    elif ground_centroid_db is None:
        ground_centroid_db = np.array([0.0, 0.0, 0.0])
        ground_normal_db = np.array([0.0, 0.0, 1.0])

    query_local_match_coords_lidar, db_local_match_coords_lidar, gpu_index_handle = do_feature_matching_faiss_gpu(
        query_img[0, :, :], db_img[0, :, :], query_local_feat[0], db_local_feat[0], match_threshold=threshold, caption='lidar',
        filter_cpp=True, ratio_filter=False, ratio_filter_thres=0.9, gnc_filter=False, gnc_filter_thres=1.0, prealloc_index=gpu_index_handle
    )
    query_local_match_coords_intensity, db_local_match_coords_intensity, gpu_index_handle = do_feature_matching_faiss_gpu(
        query_img[3, :, :], db_img[3, :, :], query_local_feat[1], db_local_feat[1], match_threshold=threshold, caption='intensity',
        filter_cpp=True, ratio_filter=False, ratio_filter_thres=0.9, gnc_filter=False, gnc_filter_thres=1.0, prealloc_index=gpu_index_handle
    )

    time_match2d = time.time()
    R, t, inliers, query_inliers, db_inliers, inliers_ratio, duration_match3d, duration_solve = \
        localization_gnc_3d_with_all_2d_matches(query_pcd, db_pcd,
                                            query_local_match_coords_lidar, db_local_match_coords_lidar,
                                            query_local_match_coords_intensity, db_local_match_coords_intensity,
                                            201, 0.4, 1, x_max_ind_query, x_max_ind_db, 100, 1.0,
                                            ground_centroid_query.astype(np.float32),
                                            ground_normal_query.astype(np.float32),
                                            ground_centroid_db.astype(np.float32),
                                            ground_normal_db.astype(np.float32),
                                            0.5, True)
    
    

    duration_match = time_match2d - time_doloc + duration_match3d 
    # print('duration_match_2d:', (time_match2d - time_doloc) * 1000.0, 'duration_match_3d:', duration_match3d * 1000.0)
    if R is None or t is None or np.all(R == 0):
        H = None
    else:
        H = np.hstack((R, t.reshape(-1, 1)))

    return_list = [H]
    if return_inliers:
        return_list.extend([query_inliers, db_inliers, inliers_ratio])
    if profiling:
        return_list.extend([duration_match, duration_solve])

    return_list.append(gpu_index_handle)

    return tuple(return_list)


def test_predictions(model, model_name, query_dataset, db_dataset, query_ids, device='cuda', process_idx=0, downsample=1, use_patchwork=True):
    predictions_file = f'{model_name}_{db_dataset.cache_str}_{query_dataset.raw_seq}_{db_dataset.raw_seq}_predictions.pkl.gz'
    predictions_path = os.path.join('./cache_match/', predictions_file)
    # pkl_path = os.path.join('./match_results/', f'{model_name}_{db_dataset.cache_str}_{query_dataset.raw_seq}_{db_dataset.raw_seq}_predictions.pkl')
    with gzip.open(predictions_path, 'rb') as f:
        predictions = pickle.load(f)
    
    print(f"Loaded {len(predictions)} predictions from {predictions_path}")
    
    cache_str = db_dataset.cache_str
    cache_filename = f'{cache_str}_{db_dataset.raw_seq}_{query_dataset.raw_seq}.npz'
    npz_path = os.path.join('./cache_ious/', cache_filename)
    iou_matrix = sparse.load_npz(npz_path)
    iou_thres = 0.25
    
    model.to(device)
    model.eval()
    
    avg_err_theta = 0.0
    avg_err_trans = 0.0
    fail_times = 0
    success_times = 0
    x_max_ind_query = 100
    x_max_ind_db = 100
    if db_dataset.cache_str in ['MCD', 'Garden']:
        x_max_ind_db = 200
    if query_dataset.cache_str in ['MCD', 'Garden', 'MCD-Cross', 'Garden-Cross']:
        x_max_ind_query = 200

    totol_localization_times = 0
    # Filter query_ids based on iou_thres and skip_querys
    query_ids_ = []
    for query_id in query_ids:
        if query_id in query_dataset.skip_querys:
            continue
        
        start_ptr = iou_matrix.indptr[query_id]
        end_ptr = iou_matrix.indptr[query_id + 1]
        db_indices = iou_matrix.indices[start_ptr:end_ptr]
        ious = iou_matrix.data[start_ptr:end_ptr]

        positives = []
        for k in range(len(db_indices)):
            db_idx = db_indices[k]
            iou = ious[k]
            if iou > iou_thres: positives.append(db_idx)
        if len(positives) == 0: continue
            
        query_ids_.append(query_id)
    query_ids = query_ids_
    pr_success_times = 0
    gpu_index_handle = None
    for enum_id in tqdm(range(0, len(query_ids), downsample), desc=f"Process-{process_idx}:", position=process_idx):
        totol_localization_times += 1
        query_id = query_ids[enum_id]
        matched_idx = predictions[query_id][0]
        matched_idx_iou = 0.0

        start_ptr = iou_matrix.indptr[query_id]
        end_ptr = iou_matrix.indptr[query_id + 1]
        db_indices = iou_matrix.indices[start_ptr:end_ptr]
        ious = iou_matrix.data[start_ptr:end_ptr]
        positives = []
        
        for k in range(len(db_indices)):
            db_idx = db_indices[k]
            iou = ious[k]
            if iou > iou_thres: positives.append(db_idx)                
            if db_idx == matched_idx: matched_idx_iou = iou
                

        pr_success = matched_idx in positives
        pr_success_times += int(pr_success)
        # print(f"Query ID: {query_id}, Matched Index: {matched_idx}")
        
        query_input, query_pcd, _ = query_dataset.__getitem__(query_id, get_pcd=True)
        db_input, db_pcd, _ = db_dataset.__getitem__(matched_idx, get_pcd=True)
        # db_img = (db_input.transpose(1, 2, 0) * 256).astype(np.uint8)
        db_tensor = torch.from_numpy(db_input).float().to(device).unsqueeze(0)
        with torch.no_grad():
            _, db_local_feat, db_global_desc = model(db_tensor, return_local_feats=True)

        # query_img = (query_input.transpose(1, 2, 0) * 256).astype(np.uint8)
        query_tensor = torch.from_numpy(query_input).float().to(device).unsqueeze(0)
        with torch.no_grad():
            _, query_local_feat, query_global_desc = model(query_tensor, return_local_feats=True)

        local_feat = torch.cat((query_local_feat, db_local_feat), dim=0)
        interp_local_feat = interpolate(local_feat.flatten(0, 1)).permute(0, 2, 3, 1).detach()
        H, query_inliers, db_inliers, inliers_ratio, duration_match, duration_solve, gpu_index_handle = do_localization_all_gnc_faiss_gpu(query_tensor.squeeze(0), db_tensor.squeeze(0), query_pcd, db_pcd, interp_local_feat[0:2], interp_local_feat[2:4], x_max_ind_query, x_max_ind_db, threshold=0.2, use_patchwork=use_patchwork, return_inliers=True, gpu_index_handle=gpu_index_handle, profiling=True)
        
        if H is not None:
            q_pose = pose_to_matrix(query_dataset.poses[query_id])
            db_pose = pose_to_matrix(db_dataset.poses[matched_idx])
            relative_gt = np.linalg.inv(db_pose).dot((q_pose)) # query_to_db = world_to_db * query_to_world
            # print('relative_gt:\n', relative_gt)
            
            relative_H = np.vstack((H, np.array([[0, 0, 0, 1]])))
            err = np.linalg.inv(relative_H).dot(relative_gt)
            err_rotvec = R.from_matrix(err[:3, :3]).as_rotvec()
            err_theta = np.rad2deg(np.linalg.norm(err_rotvec))  # convert to degrees
            err_trans = np.sqrt(err[0,3]**2+err[1,3]**2+err[2,3]**2)
        else:
            err_theta = 2000000
            err_trans = 1000000

        if err_theta > 5 or err_trans > 2:
            fail_times += 1
        else:
            success_times += 1
            avg_err_theta += err_theta
            avg_err_trans += err_trans

    if success_times == 0:
        success_times = 1  # Avoid division by zero
    avg_err_theta /= success_times
    avg_err_trans /= success_times
    success_rate = (totol_localization_times - fail_times) / totol_localization_times * 100
    pr_success_rate = pr_success_times / totol_localization_times

    # Save GL result to CSV
    os.makedirs('./eval_results/', exist_ok=True)
    csv_path = './eval_results/gl_results.csv'
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        if not file_exists:
            f.write('model,dataset,query_seq,db_seq,gl_success_rate,pr_success_rate,avg_err_theta,avg_err_trans,total_loc,success_cnt,fail_cnt\n')
        f.write(f'{model_name},{db_dataset.cache_str},{query_dataset.raw_seq},{db_dataset.raw_seq},{success_rate:.2f},{pr_success_rate:.4f},{avg_err_theta:.2f},{avg_err_trans:.2f},{totol_localization_times},{success_times},{fail_times}\n')

    return totol_localization_times, fail_times, success_times, avg_err_theta, avg_err_trans
    
def evaluate_gl(model, model_name, datasets_config, device='cuda', downsample=1, use_patchwork=False, **kwargs):
    for dataset_name, seqs in datasets_config:
        print(f"Evaluating global localization on {dataset_name} with scenes: {seqs}")

        eval_datasets = []
        for seq in seqs:
            real_dataset_name = dataset_name
            if dataset_name.startswith('GardenHusky') and seq == 'garden_db':
                real_dataset_name = 'Garden360'
            elif dataset_name.startswith('MCD-Cross'):
                real_dataset_name = 'MCD-OS' if seq == 'ntu_night_08' else 'MCD'
            elif dataset_name.startswith('Garden-Cross'):
                real_dataset_name = 'Garden360' if seq == 'garden_db' else 'Garden'

            test_set = infer_datasets.InferDataset(seq, real_dataset_name, **kwargs)
            test_set.cache_str = dataset_name
            eval_datasets.append(test_set)

        database_set = eval_datasets[0]
        for query_set in eval_datasets[1:]:
            print(f"Testing {query_set.raw_seq} against {database_set.raw_seq}")
            test_predictions(model, model_name, query_set, database_set, range(len(query_set)), device=device, use_patchwork=use_patchwork, downsample=downsample)