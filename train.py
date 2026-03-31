import argparse
from math import ceil
import random
import shutil
import json
from os.path import join, exists, isfile
from os import makedirs
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import h5py

from sklearn.decomposition import PCA

from tensorboardX import SummaryWriter
import numpy as np

from tqdm import tqdm
import faiss
from model.losses import TripletLoss, FocusInfoNCELoss, HardInfoNCELoss


import dataset.multi_par_dataset_iou as train_dataset
from eval_utils.utils_3dof import evaluate_recall

eval_datasets_config = [
    ['MCD', ['ntu_night_08', 'ntu_day_02', 'ntu_day_10', 'ntu_night_13']],
    ['MCD-OS', ['ntu_night_08', 'ntu_day_02', 'ntu_day_10', 'ntu_night_13']],
    ['Snail', ['81r-20240116_eve-data3_split_up', '81r-20240116_eve-data3_split_down']],
    ['Snail', ['81r-20240123-data2_split_up', '81r-20240123-data2_split_down']],
    ['Garden', ['garden_db', 'seq1_green', 'seq2_blue','seq3_red', 'seq4_yellow']],
    # ['Garden360', ['garden_db', 'seq1_green', 'seq2_blue','seq3_red', 'seq4_yellow']],
    ['GardenHusky', ['garden_db', 'btm_left', 'btm_right', 'top_left', 'top_right']],
]

train_seq_pairs = [
    [['MCD', 'ntu_night_08'], ['MCD', 'ntu_day_01']],
    [['MCD-OS', 'ntu_night_08'], ['MCD-OS', 'ntu_day_01']],
    # [['MCD-OS', 'ntu_night_08'], ['MCD', 'ntu_day_01']],
    [['Snail', '81r-20240116-data2'], ['Snail', '81r-20240116-data2']],
    [['Garden', 'garden_db'], ['Garden', 'garden_db']],
    [['Garden360', 'garden_db'], ['Garden360', 'garden_db']],
    # [['Garden360', 'garden_db'], ['Garden', 'garden_db']],
]

model_extra_args = {
    'cls_token_proj': False,
    'with_cp': False,
}

train_extra_args = {
    'do_rotation_aug': True,
    'do_mask_out_aug': False,
    'do_trans_aug': False,
    'do_query_pos_aug': False,
    'do_points_reduce_aug': False,
    'semi_hard': False,
    'dataset_path': '/mnt/nas/dataset/rosbag/UniLPR/BEVData/',
    'downsample_rates': [1, 1, 1, 3, 3],
}

infer_extra_args = {
    'dataset_path': '/mnt/nas/dataset/rosbag/UniLPR/BEVData/',
}

def get_args():
    parser = argparse.ArgumentParser(description='UniLGL')
    # parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test'])
    
    parser.add_argument('--batchSize', type=int, default=4, 
            help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
    parser.add_argument('--cacheBatchSize', type=int, default=1, help='Batch size for caching and testing')
    parser.add_argument('--nEpochs', type=int, default=40, help='number of epochs to train for')
    parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
    parser.add_argument('--lr', type=float, default=1e-5 * 2, help='Learning Rate.')
    parser.add_argument('--lrStep', type=float, default=10, help='Decay LR ever N steps.')
    parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
    parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
    parser.add_argument('--with_local_loss', type=bool, default=True,
            help='Whether to use local loss or not. If True, the model will train local feature consistency.')
    parser.add_argument('--local_loss_weight', type=float, default=0.25)
    parser.add_argument('--test_before_train', action='store_true',)

    parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
    parser.add_argument('--seed', type=int, default=1024, help='Random seed to use.')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint.')


    parser.add_argument('--runsPath', type=str, default='./runs/', help='Path to save runs to.')
    parser.add_argument('--cachePath', type=str, default='./cache/main_par_vit_dual/', help='Path to save cache to.')


    parser.add_argument('--load_from', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
    parser.add_argument('--ckpt', type=str, default='best', 
            help='Load_from from latest or best checkpoint.', choices=['latest', 'best'])
    

    opt = parser.parse_args()
    return opt

def interpolate_local_feature(token_map):
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

def train_epoch(epoch, model, train_set_downsample):
    
    train_set = train_set_downsample.dataset
    
    epoch_loss = 0

    n_batches = (len(train_set_downsample) + opt.batchSize - 1) // opt.batchSize

    criterion = TripletLoss().to(device)
    # local_criterion = PatchContrastiveLoss().to(device)
    local_criterion = FocusInfoNCELoss(tau=0.2).to(device)
    # local_criterion = HardInfoNCELoss().to(device)
    
    model.eval()
    

    if epoch>=0:
        print('====> Building Cache for Hard Mining')
        train_set.mining=False
        train_set.cache = join(opt.cachePath, 'train_feat_cache.hdf5')
        with h5py.File(train_set.cache, mode='w') as h5: 
            pool_size = model.global_feat_dim

            h5feat_query = h5.create_dataset("features_query", 
                    [len(train_set), pool_size], 
                    dtype=np.float32)
            # training_data_loader = DataLoader(dataset=train_set, num_workers=8, 
            #     batch_size=8, shuffle=False, 
            #     collate_fn=train_dataset.collate_fn)
            with torch.no_grad():
                # for iteration, (query, positives, negatives, metadatas) in enumerate(tqdm(training_data_loader), 1):
                    
                #     query = query.to(device)
                #     _, _, global_descs = model(query)
                #     indices = [metadata['query_index'] for metadata in metadatas]
                #     h5feat_query[indices, :] = global_descs.detach().cpu().numpy()
                for query_idx in tqdm(range(len(train_set))):
                    query = train_set.get_img(query_idx, False, temp_aug_off=True)
                    query = torch.from_numpy(query).unsqueeze(0)
                    # print('query shape:', query.shape)
                    
                    query = query.to(device)
                    _, _, global_descs = model(query)
                    h5feat_query[query_idx, :] = global_descs.detach().cpu().numpy()
                    
            h5feat_database = h5.create_dataset("features_database",
                    [train_set.len_database(), pool_size], 
                    dtype=np.float32)
            
            with torch.no_grad():
                for db_idx in tqdm(range(train_set.len_database())):
                    
                    query = train_set.get_img(db_idx, True, temp_aug_off=True)
                    query = torch.from_numpy(query).unsqueeze(0)
                    # print('query shape:', query.shape)
                    
                    query = query.to(device)
                    _, _, global_descs = model(query)
                    h5feat_database[db_idx, :] = global_descs.detach().cpu().numpy()
                    
        train_set.mining=True
        train_set.refreshCache()
        # train_set.mining = False
        
    training_data_loader = DataLoader(dataset=train_set_downsample, num_workers=opt.threads, 
                batch_size=opt.batchSize, shuffle=True, 
                collate_fn=train_dataset.collate_fn)
    
    model.train()

    for iteration, (query, positives, negatives, metadatas) in enumerate(training_data_loader):

        B, C, H, W = query.shape
        input = torch.cat([query, positives, negatives])

        input = input.to(device)
        
        _, local_descs, global_descs = model(input, True)

        global_descs_Q, global_descs_P, global_descs_N = torch.split(global_descs, [B, B, negatives.shape[0]])
        

        optimizer.zero_grad()

        # no need to train the kps feature
        loss = 0
        batch_triplet_loss = 0
        num_negs = negatives.shape[0] // B
        for i in range(len(global_descs_Q)):
            max_loss = torch.max(criterion(global_descs_Q[i], global_descs_P[i], global_descs_N[num_negs*i:num_negs*(i+1)]))
            batch_triplet_loss += max_loss.item()
            loss += max_loss
            
        batch_local_loss = 0
        if opt.with_local_loss:
            interp_locals_lidar = local_descs[:2 * B, 0, :, :, :]
            interp_locals_intensity = local_descs[:2 * B, 1, :, :, :]
            interp_locals_lidar = interpolate_local_feature(interp_locals_lidar) # 2 * B, 384, 201, 201
            interp_locals_intensity = interpolate_local_feature(interp_locals_intensity)
            for batch_idx in range(B):
                query_local_lidar = interp_locals_lidar[batch_idx].permute(1, 2, 0) # 201, 201, 384
                positive_local_lidar = interp_locals_lidar[B + batch_idx].permute(1, 2, 0)
                query_local_intensity = interp_locals_intensity[batch_idx].permute(1, 2, 0)
                positive_local_intensity = interp_locals_intensity[B + batch_idx].permute(1, 2, 0)
                

                local_lidar_loss = 0
                local_intensity_loss = 0
                # Matches Query to Positive
                matches_AB = torch.tensor(metadatas[batch_idx]['matches_AB'], dtype=torch.long).to(device)
                focus_points_AB = torch.tensor(metadatas[batch_idx]['focus_points_AB'], dtype=torch.long).to(device)
                matched_index_AB = torch.tensor(metadatas[batch_idx]['matched_index_AB'], dtype=torch.long).to(device)
                local_lidar_loss += local_criterion(query_local_lidar, positive_local_lidar, matches_AB, focus_points_AB, matched_index_AB)
                local_intensity_loss += local_criterion(query_local_intensity, positive_local_intensity, matches_AB, focus_points_AB, matched_index_AB)

                # Matches Positive to Query
                matches_BA = torch.tensor(metadatas[batch_idx]['matches_BA'], dtype=torch.long).to(device)
                focus_points_BA = torch.tensor(metadatas[batch_idx]['focus_points_BA'], dtype=torch.long).to(device)
                matched_index_BA = torch.tensor(metadatas[batch_idx]['matched_index_BA'], dtype=torch.long).to(device)
                local_lidar_loss += local_criterion(positive_local_lidar, query_local_lidar, matches_BA, focus_points_BA, matched_index_BA)
                local_intensity_loss += local_criterion(positive_local_intensity, query_local_intensity, matches_BA, focus_points_BA, matched_index_BA)

                # matches_AB = torch.tensor(metadatas[batch_idx]['matches_AB'], dtype=torch.long).to(device)
                # local_lidar_loss += local_criterion(query_local_lidar, positive_local_lidar, matches_AB)
                # local_intensity_loss += local_criterion(query_local_intensity, positive_local_intensity, matches_AB)

                local_loss = (local_lidar_loss + local_intensity_loss) * 0.5 * opt.local_loss_weight
                loss += local_loss
                batch_local_loss += local_loss.item()
            
        loss /= opt.batchSize
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        batch_local_loss /= opt.batchSize
        batch_triplet_loss /= opt.batchSize
        epoch_loss += batch_loss
        # print(batch_triplet_loss, batch_local_loss)
        
        if iteration % 50 == 0 or n_batches <= 10:
            print("==> Epoch[{}]({}/{}): All_Loss: {:.4f} Local_Loss: {:.4f} Triplet_Loss: {:.4f}".format(epoch, iteration, 
                n_batches, batch_loss, batch_local_loss, batch_triplet_loss), flush=True)
            writer.add_scalar('Train/All_Loss', batch_loss, ((epoch-1) * n_batches) + iteration)                    
            writer.add_scalar('Train/Local_Loss', batch_local_loss, ((epoch-1) * n_batches) + iteration)                    
            writer.add_scalar('Train/Triplet_Loss', batch_local_loss, ((epoch-1) * n_batches) + iteration)

    optimizer.zero_grad()    
    avg_loss = epoch_loss / n_batches

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), 
            flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)

def saveCheckpoint(state, is_best, model_out_path, filename='checkpoint.pth.tar'):
    filename = model_out_path+'/'+filename
    torch.save(state, filename)
    # filename_epoch = model_out_path+'/'+'checkpoint_epoch_{}.pth.tar'.format(state['epoch'])
    # shutil.copyfile(filename, filename_epoch)
    if is_best:
        shutil.copyfile(filename, model_out_path+'/'+'model_best.pth.tar')

def saveCheckpointEpoch(state, model_out_path, epoch):
    filename = model_out_path+'/'+'checkpoint_epoch_{}.pth.tar'.format(epoch)
    torch.save(state, filename)
    print('===> Saved checkpoint to:', filename)


if __name__ == "__main__":
    opt = get_args()

    device = "cuda"

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    print('===> Building model')

    # from REIN import REIN
    # from lidar_vit import LModel_CLS
    from model.lidar_dual_vit import LDModel_CLS
    # from lidar_vit import MLFModel_GeM
    
    # model = MLFModel_CLS(norm_before=True)
    # model = MLFModel_CLS(cls_token_proj=False)
    model = LDModel_CLS(**model_extra_args)
        
    # model = MLFModel_GeM(with_cp=True)
    model = model.to(device)
    
    # initialize netvlad with pre-trained or cluster
    resume_epoch = 0
    if opt.resume:
        assert opt.load_from != '', 'Please specify --load_from to resume training.'
        print('===> Resuming training from:', opt.load_from)
        opt.ckpt = 'latest'

    if opt.load_from:
        if opt.ckpt.lower() == 'latest':
            resume_ckpt = join(opt.load_from,  'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            resume_ckpt = join(opt.load_from, 'model_best.pth.tar')

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage, weights_only=False)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            if opt.resume:
                resume_epoch = checkpoint['epoch']
            model = model.to(device)

            print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))
    else: pass

    if opt.test_before_train:
        evaluate_recall(model, eval_datasets_config, None, None, device)
    # preparing tensorboard
    writer = SummaryWriter(log_dir=join(opt.runsPath, datetime.now().strftime('%b%d_%H-%M-%S')))

    logdir = writer.file_writer.get_logdir()
    try:
        makedirs(logdir)
    except:
        pass
    
    if not exists(opt.cachePath):
        makedirs(opt.cachePath)

    with open(join(logdir, 'flags.json'), 'w') as f:
        f.write(json.dumps({k:v for k,v in vars(opt).items()}))
            
    print('===> Saving state to:', logdir)
    # copy the code to logdir
    shutil.copy(__file__, logdir)
    shutil.copy("dataset/multi_par_dataset_iou.py", logdir)
    shutil.copy("eval.py", logdir)
    shutil.copy("model/losses.py", logdir)
    shutil.copy("model/lidar_dual_vit.py", logdir)

    print('===> Loading dataset(s)')

    train_set_downsample = train_dataset.TrainingDataset_downsample(seq_pars=train_seq_pairs, do_patch_matching=opt.with_local_loss, **train_extra_args)
    print('===> Training dataset size:', len(train_set_downsample))
    # initilize model weights
    all_trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    print('===> Number of trainable parameters: %d' % sum([np.prod(p.size()) for p in all_trainable_params]))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    if opt.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('===> Resuming optimizer state from epoch:', resume_epoch)

    
    best_score = 0
    for epoch in range(resume_epoch + 1, opt.nEpochs):
        
        if epoch > 1:
            train_set_downsample.dataset.semi_hard = False
        
        print('===> Training semi_hard:', train_set_downsample.dataset.semi_hard)
        train_epoch(epoch, model, train_set_downsample)
        
        saveCheckpointEpoch({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
        }, logdir, epoch)
        
        mean_recall, _ = evaluate_recall(model, eval_datasets_config, writer, epoch, device, **infer_extra_args)

        is_best = mean_recall > best_score 
        if is_best:   best_score = mean_recall
        
        saveCheckpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'recalls': mean_recall,
                'best_score': best_score,
                'optimizer' : optimizer.state_dict(),
        }, is_best, logdir)

    print('===> Best Recall: %0.2f'%(best_score*100))
    writer.close()