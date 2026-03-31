import os
from os.path import join, exists
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import shutil
import torch.utils.data as data
import pickle
import h5py
import gzip
from shapely.geometry import Polygon, Point
from matplotlib.patches import Polygon as MplPolygon
from tqdm import tqdm
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as R
import functools
import hashlib
        
def collate_fn(batch):

    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None, None

    query, positive, negatives, metadatas = zip(*batch)

    query=np.array(query)
    positive=np.array(positive)
    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    
    negatives = torch.cat(negatives, 0)
    # indices = list(indices)

    return query, positive, negatives, metadatas

def compute_iou_polygon(poly1, poly2):
    poly1 = Polygon(poly1)
    poly2 = Polygon(poly2)
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0.0

def compare_pkl(pkl1, pkl2):
    with open(pkl1, 'rb') as f:
        cache_dataset1 = pickle.load(f)
    with open(pkl2, 'rb') as f:
        cache_dataset2 = pickle.load(f)

    if cache_dataset1['seq_pars'] != cache_dataset2['seq_pars']:
        print('seq_pars not equal')
        return False
    if cache_dataset1['dataset_path'] != cache_dataset2['dataset_path']:
        print('dataset_path not equal')
        return False
    if cache_dataset1['pos_thres'] != cache_dataset2['pos_thres']:
        print('pos_thres not equal')
        return False
    if cache_dataset1['neg_thres'] != cache_dataset2['neg_thres']:
        print('neg_thres not equal')
        return False
    if cache_dataset1['cache_version'] != cache_dataset2['cache_version']:
        print('cache_version not equal')
        return False
    if cache_dataset1['use_dis_thres'] != cache_dataset2['use_dis_thres']:
        print('use_dis_thres not equal')
        return False
    
    if len(cache_dataset1['positives']) != len(cache_dataset2['positives']):
        print('positives length not equal')
        return False
    if len(cache_dataset1['negatives']) != len(cache_dataset2['negatives']):
        print('negatives length not equal')
        return False
    if len(cache_dataset1['imgs_path']) != len(cache_dataset2['imgs_path']):
        print('imgs_path length not equal')
        return False
    if len(cache_dataset1['imgs_database_path']) != len(cache_dataset2['imgs_database_path']):
        print('imgs_database_path length not equal')
        return False
    
    for query_idx in tqdm(range(len(cache_dataset1['imgs_path']))):
        if cache_dataset1['imgs_path'][query_idx] != cache_dataset2['imgs_path'][query_idx]:
            print('imgs_path not equal:', query_idx)
            return False
        if (cache_dataset1['positives'][query_idx] != cache_dataset2['positives'][query_idx]).any():    
            print('positives not equal:', query_idx)
            return False
        if (cache_dataset1['negatives'][query_idx] != cache_dataset2['negatives'][query_idx]).any():
            print('negatives not equal:', query_idx)
            return False
        
    for database_idx in tqdm(range(len(cache_dataset1['imgs_database_path']))):
        if cache_dataset1['imgs_database_path'][database_idx] != cache_dataset2['imgs_database_path'][database_idx]:
            print('imgs_database_path not equal:', database_idx)
            return False

    print('pkl files are equal')
    return True

def get_seq_info(dataset_path = './datasets/', seq=['MCD', 'ntu_night_08']):
    seq_p = seq[0] + '/' + seq[1]
    if not exists(dataset_path+seq_p+'/bev_imgs/'):
        raise ValueError('Please generate bev images first!')
    
    # bev path
    imgs_p = [f for f in os.listdir(dataset_path+seq_p+'/bev_imgs/') if f.endswith('.png')]
    imgs_p.sort()
    imgs_path = [dataset_path+seq_p+'/bev_imgs/'+i for i in imgs_p]

    # gt_pose, only first 3000 frames of KITTI for training
    poses = np.loadtxt(dataset_path+seq[0]+'/poses/'+seq[1]+'.txt')

    polys = []
    with open(dataset_path+seq[0]+'/hull/'+seq[1]+'.txt', "r") as f:
        for line in f:
            hull = line.strip().split(',')
            # print(hull)
            coords = np.array([(float(hull[i]), float(hull[i + 1])) for i in range(0, len(hull), 2)])
            # polys.append(Polygon(coords))
            polys.append(coords)
            
    seq_info = {
        'seq': seq,
        'img_path': imgs_path,
        'poses': poses,
        'polys': polys
    }
    return seq_info

def get_seq_info_merged(dataset_path = './datasets/', seq=['MCD', 'ntu_night_08+ntu_day_01']):
    merge_seqs = seq[1].split('+')
    seq_infos = [get_seq_info(dataset_path, [seq[0], single_seq]) for single_seq in merge_seqs]
    img_path_merged = []
    poses_merged = []
    polys_merged = []
    for seq_info in seq_infos:
        img_path_merged += seq_info['img_path']
        poses_merged += seq_info['poses'].tolist()
        polys_merged += seq_info['polys']
    poses_merged = np.array(poses_merged)
    # print(f"Merged sequence info {seq} - Images: {len(img_path_merged)}, Poses: {len(poses_merged)}, Polys: {len(polys_merged)}")
    assert len(img_path_merged) == len(poses_merged) == len(polys_merged), "Merged sequence information is inconsistent"
    seq_info_merged = {
        'seq': seq,
        'img_path': img_path_merged,
        'poses': poses_merged,
        'polys': polys_merged
    }
    return seq_info_merged
    

def generate_filename(filename, hash_length=32):
    # Use MD5 hash algorithm
    hash_obj = hashlib.md5(filename.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()

    # Take the first hash_length characters as short filename
    short_filename = hash_hex[:hash_length]
    return short_filename

def vis_poly(poly1:Polygon, poly2:Polygon, save_path):
    fig, ax = plt.subplots()

    # Convert shapely polygon to matplotlib drawable object
    mpl_poly1 = MplPolygon(
        list(poly1.exterior.coords), 
        closed=True, 
        edgecolor='blue', 
        facecolor='lightblue',
        alpha=0.5,
        linestyle='--',
        linewidth=2
    )
    mpl_poly2 = MplPolygon(
        list(poly2.exterior.coords), 
        closed=True, 
        edgecolor='red', 
        facecolor='salmon', 
        alpha=0.5
    )
    ax.add_patch(mpl_poly1)
    ax.add_patch(mpl_poly2)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    ax.autoscale_view()
    
    # print('save_poly:', save_path)
    plt.savefig(save_path)
    plt.close(fig)
    
def pool_calc_pos_neg(par_database, 
                      par_query, 
                      database_offset, 
                      pos_thres, 
                      neg_thres, 
                      use_dis_thres, 
                      show_poly, 
                      poly_seq_path, 
                      query_idx):
    q_poly = par_query['polys'][query_idx]
    q_pose = par_query['poses'][query_idx]
    
    ious = []
    for db_idx in range(len(par_database['img_path'])):        
        dis = np.linalg.norm((q_pose - par_database['poses'][db_idx])[1:4])

        if use_dis_thres:    
            if dis > 5: ious.append(0)
            else:
                db_poly = par_database['polys'][db_idx]
                iou = compute_iou_polygon(q_poly, db_poly)
                ious.append(iou)
        else:
            db_poly = par_database['polys'][db_idx]
            iou = compute_iou_polygon(q_poly, db_poly)
            ious.append(iou)
            
    ious = np.array(ious)
    indexes = np.argsort(ious)[::-1]
    remap_index = indexes[np.where(ious[indexes]>pos_thres)[0]]
    
    if par_database['seq'][0]  == par_query['seq'][0]  \
        and par_database['seq'][1] == par_query['seq'][1]:
        # Remove indices that are the same as query
        positives = []
        for pos_idx in remap_index:
            if abs(pos_idx - query_idx) > 0:
                positives.append(pos_idx + database_offset)
        positives = np.array(positives)
    else:
        positives = remap_index + database_offset
        
    negs = indexes[np.where(ious[indexes]<=neg_thres)[0]]
    negatives = negs + database_offset
    
    non_negs = indexes[np.where(np.logical_and(ious[indexes]>neg_thres, ious[indexes]<=pos_thres))[0]]
    non_negatives = non_negs + database_offset
    
    if show_poly:
        poly_seq_q_path = poly_seq_path + str(query_idx) + '/'
        os.makedirs(poly_seq_q_path)
        for pos_idx in positives:
             
            poly_query = par_query['polys'][query_idx]
            poly_pos = par_database['polys'][pos_idx]

            poly_query = Polygon(poly_query)
            poly_pos = Polygon(poly_pos)
            save_path = poly_seq_q_path + str(pos_idx)+'.pdf'
            
            vis_poly(poly_query, poly_pos, save_path)
    
    return query_idx, positives, negatives, non_negatives

class TrainingDataset(data.Dataset):
    
    def build_infos(self):
        
        self.positives = []
        self.negatives = []
        self.non_negatives = []
        self.poses = []
        self.database_poses = []
        self.polys = []
        self.database_polys = []
        self.imgs_path = []
        self.imgs_database_path = []
        self.database_infos = []
        self.query_infos = []
        
        for seq_par in self.seq_pars:
            par_database = get_seq_info_merged(self.dataset_path, seq_par[0])
            par_query = get_seq_info_merged(self.dataset_path, seq_par[1])
            database_offset = sum([len(info['img_path']) for info in self.database_infos])
            assert database_offset == len(self.imgs_database_path)
            
            self.database_infos.append(par_database)
            self.query_infos.append(par_query)
            # self.imgs_path += par_query['img_path']
            self.imgs_database_path += par_database['img_path']
            self.database_poses += par_database['poses'].tolist()
            self.database_polys += par_database['polys']
            
            positives = []
            negatives = []
            non_negatives = []
            valid_querys = []
            
            poly_seq_path = None
            if self.show_poly:
                if not os.path.exists('show_poly'):
                    os.makedirs('show_poly')
                
                poly_seq_path = 'show_poly/' + str(seq_par[0][0]) + '_' + str(seq_par[0][1]) + '_' + str(seq_par[1][1]) +'/'
                if os.path.exists(poly_seq_path):
                    shutil.rmtree(poly_seq_path)
                os.makedirs(poly_seq_path)
                
            calc_func = functools.partial(pool_calc_pos_neg, 
                                          par_database, 
                                          par_query, 
                                          database_offset, 
                                          self.pos_thres, self.neg_thres, 
                                          self.use_dis_thres, 
                                          self.show_poly, 
                                          poly_seq_path)
            with Pool(16) as p:
                results = list(tqdm(p.imap(calc_func, range(len(par_query['img_path']))), total=len(par_query['img_path'])))
            for result in results:
                query_idx, pos, neg, non_neg = result
                if len(pos) == 0:
                    continue
                valid_querys.append(query_idx)
                positives.append(pos)
                negatives.append(neg)
                non_negatives.append(non_neg)

            valid_img_path = [par_query['img_path'][i] for i in valid_querys]
            valid_poses = [par_query['poses'][i] for i in valid_querys]
            valid_polys = [par_query['polys'][i] for i in valid_querys]
            self.imgs_path += valid_img_path
            # self.query_infos[-1]['valid_img_path'] = valid_img_path
            # self.query_infos[-1]['valid_poses'] = valid_poses
            # self.query_infos[-1]['valid_polys'] = valid_polys
            self.query_infos[-1]['valid_length'] = len(valid_img_path)
            self.positives += positives
            self.negatives += negatives
            self.non_negatives += non_negatives
            self.poses += valid_poses
            self.polys += valid_polys
            
        assert len(self.imgs_path) == len(self.positives)
        assert len(self.imgs_path) == len(self.negatives)
        assert len(self.imgs_path) == len(self.non_negatives)
        assert len(self.imgs_path) == len(self.poses)
        assert len(self.imgs_path) == len(self.polys)
        assert len(self.imgs_database_path) == len(self.database_poses)
        assert len(self.imgs_database_path) == len(self.database_polys)        
                        
        # clear cache
        for info in self.database_infos:
            info['length'] = len(info['img_path'])
            info.pop('img_path', None)
            info.pop('poses', None)
            info.pop('polys', None)

        for info in self.query_infos:
            info['length'] = len(info['img_path'])
            info.pop('img_path', None)
            info.pop('poses', None)
            info.pop('polys', None)
        
        if self.cache_dataset:
            # save infos to pickle file
            cache_dataset = {
                'cache_version': self.cache_version, 
                'seq_pars': self.seq_pars,
                'use_dis_thres': self.use_dis_thres,
                'dataset_path': self.dataset_path,
                'pos_thres': self.pos_thres,
                'neg_thres': self.neg_thres,
                'positives': self.positives,
                'negatives': self.negatives,
                'non_negatives': self.non_negatives,
                'imgs_path': self.imgs_path,
                'imgs_database_path': self.imgs_database_path,
                'poses': self.poses,
                'polys': self.polys,
                'database_poses': self.database_poses,
                'database_polys': self.database_polys,
                'database_infos': self.database_infos,
                'query_infos': self.query_infos
            }
            with gzip.open(self.cache_path, 'wb') as f:
                pickle.dump(cache_dataset, f)
            
    def is_valid_cache(self, cache_dataset):
        return cache_dataset['seq_pars'] == self.seq_pars and \
                cache_dataset['dataset_path'] == self.dataset_path and \
                cache_dataset['pos_thres'] == self.pos_thres and \
                cache_dataset['neg_thres'] == self.neg_thres and \
                cache_dataset['cache_version'] == self.cache_version and \
                cache_dataset['use_dis_thres'] == self.use_dis_thres
                    
            
    def load_infos(self):
        has_valid_cache = False
        if self.cache_dataset and exists(self.cache_path):
            with gzip.open(self.cache_path, 'rb') as f:
                cache_dataset = pickle.load(f)
                if self.is_valid_cache(cache_dataset):
                    print('load cache dataset:', self.cache_path)
                    has_valid_cache = True
                    self.positives = cache_dataset['positives']
                    self.negatives = cache_dataset['negatives']
                    self.non_negatives = cache_dataset['non_negatives']
                    self.imgs_path = cache_dataset['imgs_path']
                    self.imgs_database_path = cache_dataset['imgs_database_path']
                    self.poses = cache_dataset['poses']
                    self.polys = cache_dataset['polys']
                    self.database_poses = cache_dataset['database_poses']
                    self.database_polys = cache_dataset['database_polys']
                    self.database_infos = cache_dataset['database_infos']
                    self.query_infos = cache_dataset['query_infos']
                    
        if not has_valid_cache:
            print('cache dataset not valid, build new dataset info')
            self.build_infos()
        
        for seq_par_idx, seq_par in enumerate(self.seq_pars):
            print('seq_par:', seq_par, 'database_len:', self.database_infos[seq_par_idx]['length'], \
                  'query_len:', self.query_infos[seq_par_idx]['length'], 'query_valid_len:', self.query_infos[seq_par_idx]['valid_length'])

    
    def __init__(self, 
                 dataset_path = '/mnt/nas/dataset/rosbag/UniLPR/',
                 show_poly=False,
                 cache_dataset=True,
                 do_rotation_aug=False,
                 do_mask_out_aug=False,
                 do_trans_aug=False,
                 do_patch_matching=False,
                 do_points_reduce_aug=False,
                 do_query_pos_aug=False,
                 use_dis_thres=False,
                 semi_hard=True,
                 xyz_only=False,
                 intensity_only=False,
                 cache_dir='cache_datasets',
                 cache_prefix='par_iou_',
                 pos_thres=0.25,
                 neg_thres=0.20,
                 seq_pars=[
                    [['MCD', 'ntu_night_08'], ['MCD', 'ntu_day_01']],
                    [['MCD-OS', 'ntu_night_08'], ['MCD-OS', 'ntu_day_01']],
                    [['Snail', '81r-20240116-data2'], ['Snail', '81r-20240116-data2']],
                    [['Garden', 'garden_db'], ['Garden', 'garden_db']],
                    [['Garden360', 'garden_db'], ['Garden360', 'garden_db']]
                 ]):
        super().__init__()

        self.cache_version = 1.2
        self.seq_pars = seq_pars
        self.dataset_path = dataset_path
        self.show_poly = show_poly
        self.cache_dataset = cache_dataset

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_str = cache_dir + '/' + cache_prefix
        for seq_par in self.seq_pars:
            self.cache_str += str(seq_par[0][0]) + '_' + str(seq_par[0][1]) + '_' + str(seq_par[1][1]) + '_'
        self.cache_str += 'pos_thres_' + str(int(pos_thres*100)) + '_neg_thres_' + str(int(neg_thres*100)) + '.pkl.gz'
        self.cache_path = cache_dir + '/' + cache_prefix + generate_filename(self.cache_str) + '.pkl.gz'
        print('cache_str:', self.cache_str)
        print('cache_path:', self.cache_path)
        self.use_dis_thres = use_dis_thres
        # neg, pos threshold
        self.pos_thres = pos_thres
        self.neg_thres = neg_thres
        self.num_neg = 10
        self.do_rotation_aug = do_rotation_aug
        self.do_trans_aug = do_trans_aug
        self.do_query_pos_aug = do_query_pos_aug
        self.do_mask_out_aug = do_mask_out_aug
        self.do_patch_matching = do_patch_matching
        self.do_points_reduce_aug = do_points_reduce_aug
        print('dataset_path:', self.dataset_path)
        print('pos_thres:', self.pos_thres, 'neg_thres:', self.neg_thres)
        print('use_dis_thres:', self.use_dis_thres)
        print('do_rotation_aug:', self.do_rotation_aug)
        print('do_trans_aug', self.do_trans_aug)
        print('do_mask_out_aug:', self.do_mask_out_aug)
        print('do_query_pos_aug:', self.do_query_pos_aug)
        print('do_points_reduce_aug:', self.do_points_reduce_aug)
        print('do_patch_matching:', self.do_patch_matching)
        self.load_infos()

        self.xyz_only = xyz_only
        self.intensity_only = intensity_only
        assert not (self.xyz_only and self.intensity_only), 'xyz_only and intensity_only cannot be both True'
        self.semi_hard = semi_hard
        print('semi_hard:', self.semi_hard)
        self.mining = False
        self.vis = False
        self.cache = None # filepath of HDF5 containing feature vectors for images

    # refresh cache for hard mining
    def refreshCache(self):
        h5 = h5py.File(self.cache, mode='r')
        self.h5feat_database = np.array(h5.get("features_database"))
        self.h5feat_query = np.array(h5.get("features_query"))

    def pose_to_matrix(self, pose):
        ts, x, y, z, qx, qy, qz, qw = pose
        rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
        mat = np.eye(4)
        mat[:3, :3] = rot
        mat[:3, 3] = [x, y, z]
        # print(mat, x, y, z)
        # print(x, y, z, qw, qx, qy, qz)
        return mat
    
    def get_seq_name(self, idx, is_database=False):
        if is_database:
            for info in self.database_infos:
                if idx < info['length']:
                    seq_name = info['seq']
                    break
                idx -= info['length']
        else:
            for info in self.query_infos:
                if idx < info['valid_length']:
                    seq_name = info['seq']
                    break
                idx -= info['valid_length']
        # print('get_seq_name:', seq_name)
        return seq_name
    
    def mask_out_augment(self, img1, img2):
        p = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
        p = np.array(p) / np.sum(p)
        mask_num = np.random.choice([0, 1, 2, 3, 4, 5], p=p)
        # print('mask_num:', mask_num)
        # mask_num = 5
        for i in range(mask_num):
            mask_w = np.random.randint(10, 30)
            mask_h = np.random.randint(10, 30)
            x = np.random.randint(0, img1.shape[1] - mask_w)
            y = np.random.randint(0, img1.shape[0] - mask_h)
            img1[y:y+mask_h, x:x+mask_w] = 0
            img2[y:y+mask_h, x:x+mask_w] = 0
        return img1, img2
    
    def points_reduce_augment(self, img1, img2):
        if np.random.rand() > 0.5:
            return img1, img2

        # Randomly remove points from the image
        h, w = img1.shape[:2]
        all_points_indices = np.arange(h * w)
        random_num = np.random.randint(0, h * w * 0.4)
        random_indices = np.random.choice(all_points_indices, random_num, replace=False)
        img1 = img1.flatten()
        img2 = img2.flatten()
        img1[random_indices] = 0
        img2[random_indices] = 0
        img1 = img1.reshape(h, w)
        img2 = img2.reshape(h, w)
        return img1, img2
    
    def draw_patch(self, patch_img, patch_center, patch_size):
        patch_x, patch_y = patch_center[0], patch_center[1]
        patch_xl = int(patch_x)
        patch_xr = int(patch_xl + patch_size + 1)
        patch_yl = int(patch_y)
        patch_yr = int(patch_yl + patch_size + 1)
        patch_img[patch_xl:patch_xr, patch_yl:patch_yr] = 1.0

    
    def mat_coord_to_world(self, p2w, affine_mat, x_max_ind, mat_coords):
        xcs = mat_coords[:, 0]
        ycs = mat_coords[:, 1]
        if affine_mat is not None:
            inv_rot = cv2.invertAffineTransform(affine_mat)
            rot_centers = np.stack([ycs, xcs, np.ones_like(xcs)], 0)
            ori_centers = (inv_rot @ rot_centers.reshape(3, -1))
            xcs = ori_centers[1,...]
            ycs = ori_centers[0,...]
        
        xcps = (x_max_ind - xcs) * 0.4
        ycps = (100 - ycs) * 0.4
        zcps = np.zeros_like(xcps)
        pose_3d = np.stack([xcps, ycps, zcps], -1) # (N, 3)
        pose_world = pose_3d @ p2w[:3, :3].T + p2w[:3, 3]
        # pose_world = pose_world[:, :2] # only x, y
        return pose_world
    
    def world_to_mat_coord(self, p2w, affine_mat, x_max_ind, pose_world):
        pose_3d = pose_world - p2w[:3, 3]
        pose_3d = pose_3d @ p2w[:3, :3]
        xcps = pose_3d[:, 0]
        ycps = pose_3d[:, 1]
        xcs = x_max_ind - xcps * 2.5
        ycs = 100 - ycps * 2.5
        
        if affine_mat is not None:
            rot_centers = np.stack([ycs, xcs, np.ones_like(xcs)], 0)
            ori_centers = (affine_mat @ rot_centers.reshape(3, -1))
            xcs = ori_centers[1,...]
            ycs = ori_centers[0,...]
            
        mat_coords = np.stack([xcs, ycs], -1) # (N, 2)
        return mat_coords
    
    
    def get_img(self, idx, is_database=False, return_matching = False, temp_aug_off = False, trans_aug_scale=1.0):
        if is_database:
            if idx >= len(self.imgs_database_path):
                raise ValueError('database index out of range')
            img_1 = cv2.imread(self.imgs_database_path[idx], 0)
            img_2_path = self.imgs_database_path[idx].replace('bev_imgs', 'intensity_imgs')
            img_2 = cv2.imread(img_2_path, 0)
            now_pose = self.pose_to_matrix(self.database_poses[idx])
            if self.vis:
                print('database index:', idx, 'img_1:', self.imgs_database_path[idx], 'img_2:', img_2_path)
        else:
            if idx >= len(self.imgs_path):
                raise ValueError('query index out of range')
            img_1 = cv2.imread(self.imgs_path[idx], 0)
            img_2_path = self.imgs_path[idx].replace('bev_imgs', 'intensity_imgs')
            img_2 = cv2.imread(img_2_path, 0)
            now_pose = self.pose_to_matrix(self.poses[idx])
            if self.vis:
                print('query index:', idx, 'img_1:', self.imgs_path[idx], 'img_2:', img_2_path)
                
        if self.do_mask_out_aug and not temp_aug_off:
            img_1, img_2 = self.mask_out_augment(img_1, img_2)

        if self.do_points_reduce_aug and not temp_aug_off:
            img_1, img_2 = self.points_reduce_augment(img_1, img_2)

        affine_mat = None
        if self.do_rotation_aug and not temp_aug_off:
            affine_mat = cv2.getRotationMatrix2D((img_1.shape[1]//2, img_1.shape[0]//2), np.random.randint(0,360), 1)
            
            
        if self.do_trans_aug and not temp_aug_off:
            if affine_mat is None:
                affine_mat = np.eye(2, 3)
            affine_mat[0, 2] += np.random.randint(-5, 5) * trans_aug_scale
            affine_mat[1, 2] += np.random.randint(-5, 5) * trans_aug_scale
        
        if affine_mat is not None:
            img_1 = cv2.warpAffine(img_1, affine_mat, img_1.shape[:2])
            img_2 = cv2.warpAffine(img_2, affine_mat, img_2.shape[:2])
        
        if return_matching:            
            x_max_ind = 100
            if self.get_seq_name(idx, is_database)[0] in ['MCD', 'Garden']:
                x_max_ind = 200
            
            matcoord_to_world = functools.partial(self.mat_coord_to_world,
                                                  now_pose, 
                                                  affine_mat, 
                                                  x_max_ind)
            world_to_matcoord = functools.partial(self.world_to_mat_coord,
                                                  now_pose, 
                                                  affine_mat, 
                                                  x_max_ind)
            
            matching_funcs = {
                'matcoord_to_world': matcoord_to_world,
                'world_to_matcoord': world_to_matcoord
            }   
        
        # img = img.transpose(2,0,1)
        img_1 = (img_1.astype(np.float32))/256
        img_1 = img_1[np.newaxis, :, :].repeat(3,0)

        img_2 = (img_2.astype(np.float32))/256
        img_2 = img_2[np.newaxis, :, :].repeat(3,0)

        img = np.concatenate((img_1, img_2), axis=0)
        
        if self.xyz_only:
            return_img = img_1
        elif self.intensity_only:
            return_img = img_2
        else:
            return_img = img
        
        if return_matching:
            return return_img, matching_funcs
        else:
            return return_img
    
    def patch_matching(
        self,
        img_1,
        img_2,
        matching_funcs_1, 
        matching_funcs_2,
        poly_1,
        poly_2,
        patch_size_1,
        patch_size_2):

        if self.vis:
            os.makedirs('patch_matching', exist_ok=True)
            vis_poly(poly_1, poly_2, 'patch_matching/poly_1_2.pdf')
        
        intersection_poly = poly_1.intersection(poly_2)
        
        if patch_size_2 == -1:
            # random patch size 2 choose from (3,10)
            patch_size_2 = np.random.randint(3, 7)
        
        
        if False:
            patch_img_1 = np.zeros((201, 201), dtype=np.float32)
            coords_1 = np.arange(int(201/patch_size_1)) * patch_size_1 + patch_size_1 / 2
            xs, ys = np.meshgrid(coords_1, coords_1)
            patch_centers_1 = np.stack([xs.flatten(), ys.flatten()], axis=1)
            patch_centers_world_1 = matching_funcs_1['matcoord_to_world'](patch_centers_1)
            valid_patch_1 = []
            for i in range(patch_centers_1.shape[0]):
                if intersection_poly.contains(Point(patch_centers_world_1[i][0], patch_centers_world_1[i][1])):
                    self.draw_patch(patch_img_1, patch_centers_1[i], patch_size_1)
                    valid_patch_1.append((patch_centers_1[i], patch_centers_world_1[i]))
        
        if self.vis:
            patch_img_2 = np.zeros((201, 201), dtype=np.float32)
            
        coords_2 = np.arange(int(201/patch_size_2)) * patch_size_2 + patch_size_2 // 2
        xs, ys = np.meshgrid(coords_2, coords_2)
        patch_centers_2 = np.stack([xs.flatten(), ys.flatten()], axis=1)
        patch_centers_world_2 = matching_funcs_2['matcoord_to_world'](patch_centers_2)
        patch_centers_1 = matching_funcs_1['world_to_matcoord'](patch_centers_world_2)

        valid_matching = []
        for i in range(patch_centers_2.shape[0]):
            if patch_centers_1[i][0] < 0 or patch_centers_1[i][1] < 0 or \
            patch_centers_1[i][0] >= 201 or patch_centers_1[i][1] >= 201:
                continue
            if not intersection_poly.contains(Point(patch_centers_world_2[i][0], patch_centers_world_2[i][1])):
                continue
            valid_matching.append(i)
            if self.vis:
                self.draw_patch(patch_img_2, patch_centers_2[i], patch_size_2)

        valid_matching = np.array(valid_matching, dtype=int) # [M,]
        valid_patch_centers_1 = patch_centers_1[valid_matching] # [M, 2]
        valid_patch_centers_2 = patch_centers_2[valid_matching] # [M, 2]
        # valid_matching = np.where(
        #     (patch_centers_1[:, 0] >= 0) & (patch_centers_1[:, 1] >= 0) & 
        #     (patch_centers_1[:, 0] < 201) & (patch_centers_1[:, 1] < 201)
        # )[0]
        # valid_patch_centers_1 = patch_centers_1[valid_matching] # [M, 2]
        # valid_patch_centers_2 = patch_centers_2[valid_matching] # [M, 2]
        patch_matching = np.concatenate([valid_patch_centers_1, valid_patch_centers_2], axis=1) # [M, 4]

        if self.vis:
            for i in range(valid_patch_centers_2.shape[0]):
                self.draw_patch(patch_img_2, valid_patch_centers_2[i], patch_size_2)

        # valid_patch_2 = []
        # for i in range(patch_centers_2.shape[0]):
        #     if intersection_poly.contains(Point(patch_centers_world_2[i][0], patch_centers_world_2[i][1])):
        #         valid_patch_2.append((patch_centers_2[i], patch_centers_world_2[i]))
        #         if self.vis:
        #             self.draw_patch(patch_img_2, patch_centers_2[i], patch_size_2)
                
        # valid_patch2_pos = np.array([p[1] for p in valid_patch_2], dtype=np.float32)
        # valid_patch2_match_coords = matching_funcs_1['world_to_matcoord'](valid_patch2_pos)
        
        # patch_matching = []
        # for i in range(patch_centers_1.shape[0]):
        #     patch_pos_1 = patch_centers_1[i]
        #     if patch_pos_1[0] < 0 or patch_pos_1[1] < 0 or \
        #     patch_pos_1[0] >= 201 or patch_pos_1[1] >= 201:
        #         continue
        #     patch_pos_2 = patch_centers_2[i]
        #     patch_matching.append(np.array([patch_pos_1[0], patch_pos_1[1], patch_pos_2[0], patch_pos_2[1]], dtype=int))
        # patch_matching = np.stack(patch_matching, axis=0)
        
        if self.vis:
            img_combined_scale = 1
            img_white_space = 5
            img_combined = np.zeros((201, 402 + img_white_space, 3), dtype=np.float32)
            img_combined[:, :201, :] = img_1[:3, :, :].transpose(1, 2, 0)
            img_combined[:, 201 + img_white_space:, :] = img_2[:3, :, :].transpose(1, 2, 0)
            img_combined[:, 201:201 + img_white_space, :] = 1.0 # white space
            img_combined = (img_combined * 255).astype(np.uint8).copy()
            img_combined = cv2.resize(img_combined, ((402 + img_white_space) * img_combined_scale, 201 * img_combined_scale))
            for i in range(patch_matching.shape[0]):
                patch_pos_1 = patch_matching[i][:2]
                patch_pos_2 = patch_matching[i][2:]
                random_color = np.random.randint(0, 255, size=(3,), dtype=int).tolist()
                cv2.line(
                    img_combined, 
                    (int(patch_pos_1[1] * img_combined_scale), int(patch_pos_1[0] * img_combined_scale)), 
                    (int((patch_pos_2[1] + 201 + img_white_space) * img_combined_scale), int(patch_pos_2[0] * img_combined_scale)),
                    random_color, 1)

            cv2.imwrite('patch_matching/combined_img.png', img_combined)
            cv2.imwrite('patch_matching/patch_img_2.png', (patch_img_2 * 255).astype(np.uint8))
            with open('patch_matching/patch_matching.txt', 'w') as f:
                patch_matching = patch_matching.astype(int)
                for i in range(patch_matching.shape[0]):
                    f.write(f"{patch_matching[i][0]} {patch_matching[i][1]} {patch_matching[i][2]} {patch_matching[i][3]}\n")
        
        return patch_matching, patch_centers_2, valid_matching
            
        
    def __getitem__(self, index):
        
        if self.mining:
            q_feat = self.h5feat_query[index]

            # pos_feat = self.h5feat_database[self.positives[index]]
            # dis_pos = np.sqrt(np.sum((q_feat.reshape(1,-1)-pos_feat)**2,axis=1))

            # min_idx = np.where(dis_pos==np.max(dis_pos))[0][0] 
            pos_idx = np.random.choice(self.positives[index], 1)[0]#
            pos_feat = self.h5feat_database[pos_idx].reshape(1, -1)
            dis_pos = np.sqrt(np.sum((q_feat.reshape(1, -1) - pos_feat) ** 2, axis=1))
            # pos_idx = self.positives[index][min_idx]

            neg_feat = self.h5feat_database[self.negatives[index].tolist()]
            dis_neg = np.sqrt(np.sum((q_feat.reshape(1,-1)-neg_feat)**2,axis=1))
            
            if self.semi_hard:
                semi_hard_idx = np.where(np.logical_and(dis_neg - dis_pos < 0.3, dis_pos < dis_neg))[0]
                if semi_hard_idx.shape[0] == 0:
                    semi_hard_idx = np.where(dis_neg - dis_pos < 0.3)[0]
                if semi_hard_idx.shape[0] == 0:
                    semi_hard_idx = np.arange(len(self.negatives[index])).astype(int)
                neg_idx = np.random.choice(semi_hard_idx, self.num_neg)
                neg_idx = self.negatives[index][neg_idx]
            else:
                dis_loss = (-dis_neg) + 0.3
                dis_inc_index_tmp = dis_loss.argsort()[:-self.num_neg-1:-1]
                neg_idx = self.negatives[index][dis_inc_index_tmp[:self.num_neg]]

              
        else:
            pos_idx = np.random.choice(self.positives[index], 1)[0]
            # pos_idx = self.positives[index][0]
            
            neg_idx = np.random.choice(np.arange(len(self.negatives[index])).astype(int), self.num_neg)
            neg_idx = self.negatives[index][neg_idx]
            
        if self.vis:
            # pos_idx = self.positives[index][40]
            pos_idx = self.positives[index][10]
            # pos_idx = 1035
            
            neg_idx = np.random.choice(np.arange(len(self.negatives[index])).astype(int), self.num_neg)
            neg_idx = self.negatives[index][neg_idx]
            # neg_idx = [9438]
            print('query index:', index, 'positive index:', pos_idx, 'negative index:', neg_idx)
        

        query, query_matching = self.get_img(index, False, True)
        if self.do_query_pos_aug and np.random.rand() < 0.05:
            positive, positive_matching = self.get_img(index, False, True, trans_aug_scale=2) # use augmentation to process the query image to get positive
            query_poly = Polygon(self.polys[index])
            positive_poly = Polygon(self.polys[index])
            query_seq = self.get_seq_name(index, False)
            positive_seq = self.get_seq_name(index, False)
        else:
            positive, positive_matching = self.get_img(pos_idx, True, True)
            query_poly = Polygon(self.polys[index])
            positive_poly = Polygon(self.database_polys[pos_idx])
            query_seq = self.get_seq_name(index, False)
            positive_seq = self.get_seq_name(pos_idx, True)
        
        # Just for debug
        # positive = query.copy()
        # positive_matching = copy.deepcopy(query_matching)
        # positive_poly = copy.deepcopy(query_poly)
        
        metadata = {
            'query_index': index,
            'positive_index': pos_idx,
            'negative_indices': neg_idx,
            'query_seq': query_seq,
            'positive_seq': positive_seq,
        }
        
        if self.do_patch_matching:
            matches_AB, focus_points_AB, matched_index_AB = self.patch_matching(
                query,
                positive,
                query_matching,
                positive_matching,
                query_poly,
                positive_poly,
                1,
                -1,
            )
            matches_BA, focus_points_BA, matched_index_BA = self.patch_matching(
                positive,
                query,
                positive_matching,
                query_matching,
                positive_poly,
                query_poly,
                1,
                -1,
            )
            metadata['matches_AB'] = matches_AB
            metadata['focus_points_AB'] = focus_points_AB
            metadata['matched_index_AB'] = matched_index_AB
            metadata['matches_BA'] = matches_BA
            metadata['focus_points_BA'] = focus_points_BA
            metadata['matched_index_BA'] = matched_index_BA
            if self.vis:
                print('matches shape:', matches_AB.shape)
                print('matche sample:', matches_AB[0])
        # self.patch_matching(
        #     query,
        #     query,
        #     query_matching,
        #     query_matching,
        #     Polygon(self.polys[index]),
        #     Polygon(self.polys[index]),
        #     1,
        #     16 * 201 / 224,
        # )
                

        negatives = []
        for neg_i in neg_idx:
            negative = self.get_img(neg_i, True)
            negatives.append(torch.from_numpy(negative.astype(np.float32)))

        negatives = torch.stack(negatives, 0)

        # q_img = cv2.imread(self.imgs_path[index])
        # p_img = cv2.imread(self.imgs_path[pos_idx])
        # n_img = cv2.imread(self.imgs_path[neg_idx[0]])
        # print(index,pos_idx,neg_idx[0])
        # print(query.shape)
        # cv2.imshow('q_img',query[:3, :, :].transpose(1, 2, 0))
        # cv2.imshow('p_img',positive)
        # cv2.imshow('n_img',negatives[0])
        # cv2.waitKey(0)
        return query, positive, negatives, metadata

    def __len__(self):
        return len(self.imgs_path)
    
    def len_database(self):
        return len(self.imgs_database_path)
    
    def len_query(self):
        return len(self.imgs_path)


class TrainingDataset_downsample(data.Dataset):
    """
    Downsampled dataset for training
    """
    def __init__(self, downsample_rates=[1, 1, 1, 3, 3], **kwargs):
        super().__init__()
        self.dataset = TrainingDataset(**kwargs)
        # downsample_rates is a list of downsample rates for each sequence
        self.downsample_rates = downsample_rates
        self.seq_pars = self.dataset.seq_pars
        self.seq_infos = self.dataset.query_infos
        assert len(self.seq_pars) == len(self.seq_infos), 'seq_pars and seq_infos must have the same length'
        self.seq_lens = [info['valid_length'] for info in self.seq_infos]
        assert sum(self.seq_lens) == len(self.dataset), 'seq_lens must match the length of original dataset'
        self.downsampled_lens = [int(length / rate) for length, rate in zip(self.seq_lens, self.downsample_rates)]
        print('downsample lens:', self.downsampled_lens)
        
    def __len__(self):
        return sum(self.downsampled_lens)
    
    def __getitem__(self, index):
        seq_idx = 0
        while index >= self.downsampled_lens[seq_idx]:
            index -= self.downsampled_lens[seq_idx]
            seq_idx += 1
        
        downsample_rate = self.downsample_rates[seq_idx]
        # query_index = index * downsample_rate
        query_index = index * downsample_rate + int(np.random.randint(0, downsample_rate))
        if query_index >= self.seq_lens[seq_idx]:
            query_index = self.seq_lens[seq_idx] - 1
        
        # get the original index in the dataset
        original_index = sum(self.seq_lens[:seq_idx]) + query_index
        
        return self.dataset.__getitem__(original_index)
        


if __name__ == '__main__':
    # use time to set numpy random seed
    import time
    np.random.seed(int(time.time() * 1000) % 1000000)
    # dataset = TrainingDataset(show_poly=False, do_mask_out_aug=False, do_rotation_aug=True, do_patch_matching=True)
    # print('dataset length:', len(dataset))
    # dataset.vis = True
    # dataset.__getitem__(200)
    
    dataset = TrainingDataset_downsample(show_poly=False,
                                         do_mask_out_aug=False,
                                         do_points_reduce_aug=False,
                                         do_rotation_aug=True,
                                         do_query_pos_aug=False,
                                         do_trans_aug=False,
                                         do_patch_matching=True,
                                         downsample_rates=[1, 1, 1, 3, 3])
    print('dataset length:', len(dataset))
    dataset.dataset.vis = True
    dataset.__getitem__(2897)
    
    # pkl_1 = 'cache_dataset_par_iou.pkl'
    # pkl_2 = 'cache_dataset_par_pool.pkl'
    # compare_pkl(pkl_1, pkl_2)
    # training_data_loader = DataLoader(dataset=dataset, num_workers=4, batch_size=4, shuffle=True, collate_fn=collate_fn)
    # for iteration, (query, positives, negatives, metadatas) in enumerate(tqdm(training_data_loader), 1):
    #     indices = [metadata['query_index'] for metadata in metadatas]
    #     print('iteration:', iteration, 'query indices:', indices)
    #     if iteration > 10:
    #         break