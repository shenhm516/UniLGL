import sys
sys.path.append('.')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
# import loralib as lora
import torch.utils.checkpoint as cp

# from mag_fusion_utils import MLFModel, load_mlf_pretrain
# from dinov2 import DINOv2, load_dinov2_pretrain
from model.dino import vit_mlsmall, vit_mlbase , load_ml_pretrain
# from pooling import NetVLAD, CosPlace, MixVPR, GeMPool

def freeze_module(module):
    if isinstance(module, nn.Parameter):
        module.requires_grad = False
        return module
    
    for param in module.parameters():
        param.requires_grad = False
    module.eval()
    return module

def unfreeze_module(module):
    if isinstance(module, nn.Parameter):
        module.requires_grad = True
        return module
    
    for param in module.parameters():
        param.requires_grad = True
    module.train()
    return module



class LDModel_CLS(nn.Module):
    
    def __init__(self, 
                 vit_config='vits', 
                 vit_pretrain='pretrain/dino_deitsmall8_pretrain.pth',
                 dinov2=False, 
                 cut_depth=-1, 
                 num_cls_tokens=1,
                 freeze_layer=-1,
                 global_feat_dim=256,
                 cls_token_proj=True,
                 with_cp=False,
                 target_img_size=224,
                 headtail_train=False):
        super(LDModel_CLS, self).__init__()

        assert dinov2 == False, "dinov2 is not supported for LDModel_CLS"
        vit_factory = {
            'vits': vit_mlsmall,
            'vitb': vit_mlbase,
        }
        
        dinov2_factory = {
            'vits': 'dinov2_vits14',
            'vitb': 'dinov2_vitb14',
        }
        
        
        self.vit_means = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.vit_stds = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.target_img_size = target_img_size
        
        if dinov2:
            print("Loading dinov2:", dinov2_factory[vit_config])
            self.encoder = torch.hub.load('', dinov2_factory[vit_config], source='local', with_cp=with_cp)
        else:
            print("Loading dino:", vit_config, vit_pretrain)
            self.encoder = vit_factory[vit_config](patch_size=8, with_cp=with_cp)
            if vit_pretrain is not None:
                print("Loading pretrain:", vit_pretrain)
                self.encoder = load_ml_pretrain(self.encoder, vit_pretrain)
            else:
                print("No pretrain loaded!!!!")
        
        if headtail_train:
            self.encoder = freeze_module(self.encoder)
            # self.encoder.cls_token = unfreeze_module(self.encoder.cls_token)
            # self.encoder.pos_embed = unfreeze_module(self.encoder.pos_embed)
            self.encoder.patch_embed = unfreeze_module(self.encoder.patch_embed)
            for i in range(freeze_layer, 12):
                self.encoder.blocks[i] = unfreeze_module(self.encoder.blocks[i])
            self.encoder.norm = unfreeze_module(self.encoder.norm)
            
        if cut_depth > 0:
            # self.encoder.n_blocks = cut_depth
            self.encoder.blocks = self.encoder.blocks[:cut_depth]
        self.global_feat_dim_raw = self.encoder.embed_dim * num_cls_tokens
        self.global_feat_dim = self.global_feat_dim_raw
        
        self.cls_token_proj = cls_token_proj
        if cls_token_proj:
            self.global_feat_proj = nn.Linear(self.global_feat_dim_raw, self.global_feat_dim)
            self.global_feat_dim = global_feat_dim

        # init.xavier_uniform_(self.global_feat_proj.weight)
        # self.local_feat_proj = nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim)
                
    def forward(self, x_all, return_local_feats=False):
        # B, _, w_mag_ori, h_mag_ori = x_mag.shape
        B, _, w_lidar_ori, h_lidar_ori = x_all.shape
        # print(x_all.shape)
        # assert w_mag_ori == 201 and w_lidar_ori == 222
        # assert w_mag_ori == w_lidar_ori and h_mag_ori == h_lidar_ori, "x_mag and x_lidar must have the same spatial size"
        # assert w_mag_ori == h_mag_ori
        # assert w_lidar_ori == h_lidar_ori

        # Resize to target size
        # target_img_size = 224
        # x_mag = F.interpolate(x_mag, size=(target_img_size, target_img_size), mode='bilinear', align_corners=False)
            
        x_all = F.interpolate(x_all, size=(self.target_img_size, self.target_img_size), mode='bilinear', align_corners=True)
        x_lidar = x_all[:, :3, :, :]
        x_intensity = x_all[:, 3:, :, :]

        # B, _, w_mag, h_mag = x_mag.shape
        B, _, w_lidar, h_lidar = x_lidar.shape

        x_lidar_token_num = (w_lidar // self.encoder.patch_size) * (h_lidar // self.encoder.patch_size)
        

        aff = torch.zeros(B, 2, 3).to(x_lidar.device)
        aff[:, 0, 0]=1
        aff[:, 0, 1]=0
        aff[:, 1, 0]=0
        aff[:, 1, 1]=1

        # out = self.encoder.get_intermediate_layers(x_mag, x_lidar, n=1, reshape=False, return_class_token=True)[0]
        # out = self.encoder.get_intermediate_layers(x_mag, n=1, reshape=False, return_class_token=True)[0]
        # out = self.encoder.get_intermediate_layers(x_mag, x_lidar)[0]
        out = self.encoder.get_intermediate_layers(x_lidar, x_intensity)[0]
        # out = cp.checkpoint(self.encoder.get_intermediate_layers, x_lidar, use_reentrant=False)[0]
        global_feats = out[:, 0, :]
        local_feats = out[:, 1:, :]
        # local_feats, global_feats = out
        # local_feats = self.local_feat_proj(local_feats)
        if self.cls_token_proj:
            global_feats = self.global_feat_proj(global_feats)
        global_feats = F.normalize(global_feats, dim=-1)
        global_feats = global_feats.view(B, -1).contiguous()
        # print(local_feats.shape)
        # local_feats = local_feats.view(B, 2, w_mag // self.patch_size, h_mag // self.patch_size, -1).permute(0, 2, 3, 4, 1).contiguous()
        if return_local_feats:
            local_feats = local_feats.view(B, 2, w_lidar // self.encoder.patch_size, h_lidar // self.encoder.patch_size, -1) \
            .permute(0, 1, 4, 2, 3).contiguous()  # B 2 C H W
            # print(local_feats.shape)
            # grid = F.affine_grid(aff, torch.Size((B, local_feats.shape[1], w_lidar_ori, h_lidar_ori)), align_corners=True).type(x_lidar.type())
            # local_feats = F.grid_sample(local_feats, grid, align_corners=True, mode='bicubic')
            # local_feats = F.normalize(local_feats, dim=-1)
        else:
            local_feats = None
            
        return None, local_feats, global_feats
    
    
if __name__ == "__main__":
    # model = MLFModel_Vlad(rotations=-1).cuda()
    model = LDModel_CLS(cls_token_proj=False, with_cp=True).cuda()
    # model = LDModel_Vlad(rotations=-1, with_cp=True).cuda()
    # print(model)
    # x_mag = torch.randn(2, 3, 101, 101).cuda()
    x_all = torch.randn(2, 6, 201, 201).cuda()
    
    # masks = torch.ones(2, 1, 36, 36).bool()
    # out = model.get_intermediate_layers(x_mag, x_lidar, n=1, reshape=False, return_class_token=True)
    # for i, (x, cls_token) in enumerate(out):
    #     print(f"Layer {i}: x shape: {x.shape}, cls_token shape: {cls_token.shape}")
    # vlad_feats, local_feats, global_feats = model(x_lidar)
    vlad_feats, local_feats, global_feats = model(x_all)
    print(global_feats.shape)