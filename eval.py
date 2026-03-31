import torch
from model.lidar_dual_vit import LDModel_CLS
from eval_utils.utils_3dof import evaluate_pr
from eval_utils.utils_6dof import evaluate_gl
dataset_args = {
    'dataset_path': '/mnt/nas/dataset/rosbag/UniLPR/BEVData/',
    'use_laz': True,
    'gen_bev_pcd': False,
    'gen_bev_params': {'pc_range':40, 'res':0.4, 'blur_size':25, 'median_size':3, 'high_intensity_threshold':0.995}
}

datasets_config = [
    ['MCD', ['ntu_night_08', 'ntu_day_02', 'ntu_day_10', 'ntu_night_13']],
    ['MCD-OS', ['ntu_night_08', 'ntu_day_02', 'ntu_day_10', 'ntu_night_13']],
    ['Snail', ['81r-20240123-data2_split_up', '81r-20240123-data2_split_down']],
    ['Snail', ['81r-20240116_eve-data3_split_up', '81r-20240116_eve-data3_split_down']],
    ['Garden', ['garden_db', 'seq1_green', 'seq2_blue', 'seq3_red', 'seq4_yellow']],   
    ['GardenHusky', ['garden_db', 'btm_left', 'btm_right', 'top_left', 'top_right']],
]

if __name__ == "__main__":
    model = LDModel_CLS(
        cls_token_proj=False,
        with_cp=False,
    )
    checkpoint_path = './runs/UniLGL/checkpoint_epoch_19.pth.tar'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")
    evaluate_pr(model, 'unilpr', datasets_config, device='cuda', **dataset_args)
    evaluate_gl(model, 'unilpr', datasets_config, downsample=1, device='cuda', use_patchwork=False, **dataset_args)