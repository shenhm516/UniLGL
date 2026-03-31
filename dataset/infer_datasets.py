import sys
sys.path.append('.')
import os
# from os.path import join, exists
import numpy as np
import cv2
import time
import torch.utils.data as data
import functools
from scipy import sparse
# import h5py

import faiss
from shapely.geometry import Polygon
import pickle
from tqdm import tqdm
from multiprocessing import Pool
from pcd_utils import voxel_filter, getBEV
from dataset.laz_utils import save_pcd_laz, load_pcd_laz

all_datasets_config = [
    ['MCD', ['ntu_night_08', 'ntu_day_01', 'ntu_day_02', 'ntu_day_10', 'ntu_night_13']],
    ['MCD-OS', ['ntu_night_08', 'ntu_day_01', 'ntu_day_02', 'ntu_day_10', 'ntu_night_13']],
    ['MCD-Cross', ['ntu_night_08', 'ntu_day_01', 'ntu_day_02', 'ntu_day_10', 'ntu_night_13']],
    ['Snail', ['81r-20240116-data2_split_up', '81r-20240116-data2_split_down']],
    ['Snail', ['81r-20240116_eve-data3_split_up', '81r-20240116_eve-data3_split_down']],
    ['Snail', ['81r-20240123-data2_split_up', '81r-20240123-data2_split_down']],
    ['Garden', ['garden_db', 'seq1_green', 'seq2_blue','seq3_red', 'seq4_yellow']],
    ['Garden-Cross', ['garden_db', 'seq1_green', 'seq2_blue','seq3_red', 'seq4_yellow']],
    ['GardenHusky', ['garden_db', 'btm_left', 'btm_right', 'top_left', 'top_right']],
]


split_config = {
    '81r-20240116-data2': 8300,
    '81r-20240116_eve-data3': 7000,
    '81r-20240123-data2': 11300,
}

# Skip queries that in static or very low quality
skip_config = {
    'seq1_green': [1040, 2000],
    'top_right': [-2300, 0],
    'btm_right': [-1500, 0],
    'seq3_red': [-400, 0],
}

class InferDataset(data.Dataset):
    def __init__(self,
                 seq,
                 dataset_name,
                 dataset_path = '/workspace/datasets/UniLPR/',
                 return_options = [True, True], # [return_img, return_intensity]
                 quat_mode='xyzw', # 'xyzw' or 'wxyz'
                 gen_bev_pcd=False,
                 gen_bev_params = {'pc_range':40, 'res':0.4, 'blur_size':25, 'median_size':5, 'high_intensity_threshold':0.995},
                 use_laz=False
                 ):
        super().__init__()

        self.return_options = return_options
        self.gen_bev_pcd = gen_bev_pcd
        self.gen_bev_params = gen_bev_params
        self.use_laz = use_laz

        if self.gen_bev_pcd:
            print('Generate BEV from PCD with params:', gen_bev_params)
        if self.use_laz:
            print('Using LAZ format for point clouds.')
        self.raw_seq = seq
        split = -1
        if seq.endswith('_split_up'):
            seq = seq[:-len('_split_up')]
            split = 0
        elif seq.endswith('_split_down'):
            seq = seq[:-len('_split_down')]
            split = 1

        # bev path
        dataset_path = dataset_path + dataset_name + '/'
        imgs_p = [f for f in os.listdir(dataset_path+seq+'/bev_imgs/') if f.endswith('.png')]
        imgs_p.sort()
        self.imgs_path = [dataset_path+seq+'/bev_imgs/'+i for i in imgs_p]

        # gt_pose
        self.poses = np.loadtxt(dataset_path+'poses/'+seq+'.txt')
        if quat_mode == 'wxyz':
            self.poses[:, [4, 5, 6, 7]] = self.poses[:, [7, 4, 5, 6]]
        self.seq = seq
        self.cache_str = dataset_name

        self.polys = []
        with open(dataset_path+'/hull/'+seq+'.txt', "r") as f:
            for line in f:
                hull = line.strip().split(',')
                coords = np.array([(float(hull[i]), float(hull[i + 1])) for i in range(0, len(hull), 2)])
                # self.polys.append(Polygon(coords))
                self.polys.append(coords)
        
        if split >= 0:
            if seq not in split_config:
                raise ValueError(f'Split index for sequence {seq} not found in split_config dictionary.')
            split_idx = split_config[seq]
            if split == 0:
                self.imgs_path = self.imgs_path[:split_idx]
                self.poses = self.poses[:split_idx]
                self.polys = self.polys[:split_idx]
            elif split == 1:
                self.imgs_path = self.imgs_path[split_idx:]
                self.poses = self.poses[split_idx:]
                self.polys = self.polys[split_idx:]
        
        self.skip_querys = [-1]        
        if seq in skip_config:
            skip_range = skip_config[seq]
            if skip_range[0] < 0:
                skip_range[0] += len(self.poses)
                skip_range[1] += len(self.poses)
                
            self.skip_querys = list(range(skip_range[0], skip_range[1]))

    def get_pcd(self, index):
        if self.use_laz:
            pcd_path = self.imgs_path[index].replace('bev_imgs', 'laz').replace('.png', '.laz')
            pcd = load_pcd_laz(pcd_path)
            if pcd is None:
                raise FileNotFoundError(f"Laz file not found or failed to load: {pcd_path}")
        else:
            # t0 = time.time()
            pcd_path = self.imgs_path[index].replace('bev_imgs', 'pcd').replace('.png', '.npz')
            pcd = np.load(pcd_path)['points'].astype(np.float32)
            # t1 = time.time()
            # print(f"Loaded PCD {pcd_path} in {(t1 - t0) * 1000:.3f} milliseconds.")
        
        assert pcd.shape[1] == 4, 'pcd shape should be (N, 4), but got {}'.format(pcd.shape)
        return pcd
    
    def bev_from_pcd(self, index):
        all_points = self.get_pcd(index).astype(np.float32)
        all_points = voxel_filter(all_points, self.gen_bev_params['res'])
        return getBEV(all_points, **self.gen_bev_params)

    def __getitem__(self, index, get_pcd=False):
        if self.gen_bev_pcd:
            img_1_pcd, img_2_pcd = self.bev_from_pcd(index)

        if self.return_options[0]:
            if self.gen_bev_pcd:
                img_1 = img_1_pcd
            else:
                img_1 = cv2.imread(self.imgs_path[index], 0)

            img_1 = (img_1.astype(np.float32))
            img_1 = img_1[np.newaxis, :, :].repeat(3,0) / 256.0

        if self.return_options[1]:
            if self.gen_bev_pcd:
                img_2 = img_2_pcd
            else:
                img_2_path = self.imgs_path[index].replace('bev_imgs', 'intensity_imgs')
                img_2 = cv2.imread(img_2_path, 0)
            img_2 = (img_2.astype(np.float32))
            img_2 = img_2[np.newaxis, :, :].repeat(3,0) / 256.0
        
        if self.return_options[0] and self.return_options[1]:
            img_res = np.concatenate((img_1, img_2), axis=0)
        elif self.return_options[0]:
            img_res = img_1
        elif self.return_options[1]:
            img_res = img_2
        else:
            raise ValueError('Both return_options cannot be False')
        
        if get_pcd:
            pcd = self.get_pcd(index)
            return img_res, pcd, index
        
        return img_res, index

    def __len__(self):
        return len(self.imgs_path)

def compute_iou_polygon(poly1, poly2):
    poly1 = Polygon(poly1)
    poly2 = Polygon(poly2)
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0.0

def compute_query_ious(database, query, query_idx):
    # print(multiprocessing.current_process().name)
    query_poly = query.polys[query_idx]
    idx_ious_list = []
    for db_idx in range(len(database.polys)):
        db_poly = database.polys[db_idx]
        iou = compute_iou_polygon(query_poly, db_poly)
        if iou > 0.01:
            idx_ious_list.append((db_idx, iou))
    idx_ious_list.sort(key=lambda x: x[1], reverse=True)
    return idx_ious_list

def cache_ious(database, querys, cache_path):
    for query_dataset in querys:
        calc_func = functools.partial(compute_query_ious, database, query_dataset)
        with Pool(processes=16) as pool:
            all_ious = list(tqdm(pool.imap(calc_func, range(len(query_dataset.polys))), total=len(query_dataset.polys)))
        save_path = os.path.join(cache_path, database.cache_str + '_' + database.raw_seq + '_' + query_dataset.raw_seq + '.pkl')
        
        # --- Optimization starts ---
        rows = []
        cols = []
        data = []

        print("Constructing sparse matrix...")
        for q_idx, matches in enumerate(all_ious):
            if not matches:
                continue
            # Unpack match results
            db_indices, ious = zip(*matches)

            # Record coordinates and values
            rows.extend([q_idx] * len(db_indices))
            cols.extend(db_indices)
            data.extend(ious)

        # Convert to numpy arrays, specify types to save space
        rows = np.array(rows, dtype=np.int32)
        cols = np.array(cols, dtype=np.int32)
        data = np.array(data, dtype=np.float32)

        # Build CSR matrix (row compressed format, suitable for slice reading)
        # shape must be specified, otherwise if the last query has no matches, matrix row count will be smaller
        n_queries = len(query_dataset.polys)
        n_database = len(database.polys)
        iou_matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n_queries, n_database)).tocsr()

        # Save as .npz format (built-in compression)
        save_name = f"{database.cache_str}_{database.raw_seq}_{query_dataset.raw_seq}.npz"
        save_path = os.path.join(cache_path, save_name)
        sparse.save_npz(save_path, iou_matrix)
        
        print(f"Saved optimized IOU cache to {save_path}")

def load_or_create_iou_cache(database, query_dataset, cache_path='./cache_ious/'):
    """
    Load IOU cache if exists, otherwise create it.

    Args:
        database: database InferDataset object
        query_dataset: single query InferDataset object
        cache_path: path to cache directory

    Returns:
        iou_matrix: sparse CSR matrix of IOU values
    """
    cache_filename = f'{database.cache_str}_{database.raw_seq}_{query_dataset.raw_seq}.npz'
    npz_path = os.path.join(cache_path, cache_filename)

    if not os.path.exists(npz_path):
        print(f"IOU cache not found at {npz_path}, creating...")
        os.makedirs(cache_path, exist_ok=True)
        cache_ious(database, [query_dataset], cache_path)

    return sparse.load_npz(npz_path)

def calc_loop_closure_nums(datasets, use_dis_thres=False):
    
    # for nclt, we use the seq 2012-02-15 for database, other sequences for query
    
    gt_thres = 5
    iou_thres = 0.25
    for i in range(1, len(datasets)):
        all_positives = 0
        positives_num = 0
        cache_filename = f"{datasets[0].cache_str}_{datasets[0].raw_seq}_{datasets[i].raw_seq}.npz"
        npz_path = os.path.join('./cache_ious/', cache_filename)    
        # pkl_path = os.path.join('./cache_ious/', datasets[0].cache_str + '_' + datasets[0].raw_seq + '_' + datasets[i].raw_seq + '.pkl')
        # with open(pkl_path, 'rb') as f:
        #     all_ious = pickle.load(f)
        iou_matrix = sparse.load_npz(npz_path)
        print(f"Loaded IOU matrix from {npz_path}")
        
        for q_idx in range(len(datasets[i].polys)):
            start_ptr = iou_matrix.indptr[q_idx]
            end_ptr = iou_matrix.indptr[q_idx + 1]
            db_indices = iou_matrix.indices[start_ptr:end_ptr]
            ious = iou_matrix.data[start_ptr:end_ptr]
            # idx_ious_list = all_ious[query_idx]
            positives = []
            for k in range(len(db_indices)):
                db_idx = db_indices[k]
                iou = ious[k]
                if use_dis_thres:
                    gt_dis = np.linalg.norm((datasets[i].poses[q_idx] - datasets[0].poses[db_idx])[1:4])
                    if gt_dis > gt_thres:
                        iou = 0
                if iou > iou_thres:
                    positives.append(db_idx)
            positives = np.array(positives)


            if len(positives)>0:
                all_positives += 1
                positives_num += len(positives)

        average_positives = positives_num / all_positives if all_positives > 0 else 0
        print('for %s, querys: %d, all_positives: %d, average_positives: %.4f' % (datasets[i].seq, len(datasets[i].polys), all_positives, average_positives))





def evaluateResults(global_descs, datasets, use_dis_thres=False):
    
    # for nclt, we use the seq 2012-02-15 for database, other sequences for query
    
    gt_thres = 5
    iou_thres = 0.25
    faiss_index = faiss.IndexFlatL2(global_descs[0].shape[1]) 
    faiss_index.add(global_descs[0])

    recalls_mcd = []
    for i in range(1, len(datasets)):
        _, predictions = faiss_index.search(global_descs[i], 1)  #top1
            
        all_positives = 0
        tp = 0
        
        pkl_path = os.path.join('./cache_ious/', datasets[0].cache_str + '_' + datasets[0].raw_seq + '_' + datasets[i].raw_seq + '.pkl')
        with open(pkl_path, 'rb') as f:
            all_ious = pickle.load(f)
        
        for q_idx, pred in enumerate(predictions):
            if q_idx in datasets[i].skip_querys:
                continue
            
            query_idx = q_idx
            idx_ious_list = all_ious[query_idx]
            positives = []
            for db_idx, iou in idx_ious_list:
                if use_dis_thres:
                    gt_dis = np.linalg.norm((datasets[i].poses[query_idx] - datasets[0].poses[db_idx])[1:4])
                    if gt_dis > gt_thres:
                        iou = 0
                if iou > iou_thres:
                    positives.append(db_idx)
            positives = np.array(positives)


            if len(positives)>0:
                all_positives += 1
                if pred[0] in positives:
                    tp += 1

        recall_top1 = tp / all_positives #tp/(tp+fp)
        recalls_mcd.append(recall_top1)
        print('recall for %s: %.4f, all_positives: %d' % (datasets[i].seq, recall_top1, all_positives))

    return recalls_mcd


if __name__ == '__main__':
    extra_args = {
        'dataset_path': '/mnt/nas/dataset/rosbag/UniLPR/',
        'use_laz': False,
        'gen_bev_pcd': False,
        'gen_bev_params': {'pc_range':40, 'res':0.4, 'blur_size':25, 'median_size':5, 'high_intensity_threshold':0.995}
    }

    cache_path = './cache_ious/'
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    for dataset_name, seqs in all_datasets_config:
        database_set_name = dataset_name
        query_set_name = dataset_name
        if dataset_name.startswith('GardenHusky'): # for long term PR
            database_set_name = 'Garden360'
        if dataset_name.startswith('MCD-Cross'):
            database_set_name = 'MCD-OS'
            query_set_name = 'MCD'
        if dataset_name.startswith('Garden-Cross'):
            database_set_name = 'Garden360'
            query_set_name = 'Garden'
        database = InferDataset(seqs[0], database_set_name, **extra_args)
        database.cache_str = dataset_name
        querys = [InferDataset(seq, query_set_name, **extra_args) for seq in seqs[1:]]

        cache_ious(database, querys, cache_path)
        # calc_loop_closure_nums([database] + querys, use_dis_thres=False)