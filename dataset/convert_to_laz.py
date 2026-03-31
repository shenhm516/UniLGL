import os
import numpy as np
import laspy
from tqdm import tqdm
from infer_datasets import InferDataset
from laz_utils import save_pcd_laz, load_pcd_laz

def convert_dataset_to_laz(dataset):
    """
    Convert point cloud data in InferDataset to LAZ format and save.
    Precision: 0.1mm
    Path: Replace 'pcd'/'bev_imgs' in original path with 'laz'
    """

    # Check if necessary compression backend is installed
    try:
        import lazrs
    except ImportError:
        print("Warning: 'lazrs' backend not found. Writing .laz files might fail or be slow.")
        print("Please run: pip install \"laspy[lazrs]\"")

    # Iterate through the dataset
    for i in tqdm(range(len(dataset)), desc=f"Converting {dataset.seq} to LAZ"):
        # 1. Get point cloud data (N, 4) -> x, y, z, intensity
        # Note: Read npz directly without voxel filter or other processing, preserve raw data
        points = dataset.get_pcd(i)

        # 2. Build save path
        # Original logic reference: dataset.imgs_path[i] is '.../bev_imgs/xxxx.png'
        # Target path: '.../laz/xxxx.laz'
        src_path = dataset.imgs_path[i]
        save_path = src_path.replace('bev_imgs', 'laz').replace('.png', '.laz')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_pcd_laz(save_path, points)

# --- Usage example ---
if __name__ == '__main__':
    datasets_config = [
        # ['MCD', ['ntu_night_08', 'ntu_day_02', 'ntu_day_10', 'ntu_night_13']],
        # ['MCD-OS', ['ntu_night_08', 'ntu_day_02', 'ntu_day_10', 'ntu_night_13']],
        # ['Snail', ['81r-20240123-data2_split_up', '81r-20240123-data2_split_down']],
        # ['Snail', ['81r-20240116_eve-data3_split_up', '81r-20240116_eve-data3_split_down']],
        # ['Garden', ['garden_db', 'seq1_green', 'seq2_blue', 'seq3_red', 'seq4_yellow']],
        # ['GardenHusky', ['garden_db', 'btm_left', 'btm_right', 'top_left', 'top_right']],
        ['Snail', ['81r-20240116-data2_split_up', '81r-20240116-data2_split_down']],
    ]
    db_seqs = ['garden_db', 'ntu_night_08']

    for dataset_name, seq_list in datasets_config:
        if dataset_name.startswith('GardenHusky'):
            database_set_name = 'Garden360'
        for seq in seq_list:
            dataset = InferDataset(
                seq=seq,
                dataset_name=database_set_name if seq in db_seqs else dataset_name,
                dataset_path='/mnt/nas/dataset/rosbag/UniLPR/',
                use_laz=False  # Don't use LAZ loading here, we want to convert raw PCD data
            )
            dataset.cache_str = dataset_name


            convert_dataset_to_laz(dataset)
