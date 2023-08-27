import sys

import warnings
warnings.filterwarnings("ignore")

import os
import shutil
import tensor
import argparse
import numpy as np
import math
import copy
import sys
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.apis import init_model

import path
folder = path.Path(__file__).abspath()
sys.path.append(folder.parent.parent)

import torch
from ptq.export_onnx import compute_projection, get_points, TRTModel_pre

cfg_n_voxels=[[200, 200, 4]]
cfg_voxel_size=[[0.5, 0.5, 1.5]]
nv=6
box_code_size = 9

def get_project_output2(img, img_metas, mlvl_feats):
    stride_i = math.ceil(img.shape[-1] / mlvl_feats.shape[-1])
    mlvl_feat_split = torch.split(mlvl_feats, nv, dim=1)

    volume_list = []
    for seq_id in range(len(mlvl_feat_split)):
        volumes = []
        for batch_id, seq_img_meta in enumerate(img_metas):
            feat_i = mlvl_feat_split[seq_id][batch_id]
            print(feat_i.shape)
            img_meta = copy.deepcopy(seq_img_meta)
            img_meta["lidar2img"]["extrinsic"] = img_meta["lidar2img"]["extrinsic"][seq_id*6:(seq_id+1)*6]
            if isinstance(img_meta["img_shape"], list):
                img_meta["img_shape"] = img_meta["img_shape"][seq_id*6:(seq_id+1)*6]
                img_meta["img_shape"] = img_meta["img_shape"][0]
            height = math.ceil(img_meta["img_shape"][0] / stride_i)
            width = math.ceil(img_meta["img_shape"][1] / stride_i)

            projection = compute_projection(
                img_meta, stride_i, noise=0).to(feat_i.device)

            n_voxels, voxel_size = cfg_n_voxels[0], cfg_voxel_size[0]

            points = get_points(  # [3, vx, vy, vz]
                n_voxels=torch.tensor(n_voxels),
                voxel_size=torch.tensor(voxel_size),
                origin=torch.tensor(img_meta["lidar2img"]["origin"]),
            ).to(feat_i.device)

            volume, out2, x2, y2= backproject_inplace(
                feat_i[:, :, :height, :width], points, projection) 
            volumes.append(volume)
        volume_list.append(torch.stack(volumes))
    mlvl_volumes = torch.cat(volume_list, dim=1)

    return mlvl_volumes, out2, x2, y2


def backproject_inplace(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [64, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long() 
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long() 
    z = points_2d_3[:, 2] 
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0) 
    volume = torch.zeros(
        (n_channels, points.shape[-1]), device=features.device
    ).type_as(features)
    
    for i in range(n_images):
        volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]

    volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)

    return volume, valid, x, y


def arg_parser():
    parser = argparse.ArgumentParser(description='For bevfusion evaluation on nuScenes dataset.')
    parser.add_argument('--config', dest='config', type=str, default='configs/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f1.py')
    parser.add_argument('--checkpoint', dest='checkpoint', type=str, default='ptq/pth/epoch_20.pth')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')   
    args = parser.parse_args()
    return args

def dump_tensor(args):
    model = init_model(args.config, args.checkpoint, device=args.device)
    device = next(model.parameters()).device
    dataset = build_dataset(model.cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False)
    
    data_iter = iter(data_loader)
    for i in range(1):
        data = next(data_iter)
        
        root = f"dump7/{i:05d}"
        if not os.path.exists(root):
            os.makedirs(root)

        torch.save(data, f"{root}/example-data.pth")
        metas = data["img_metas"].data[0]
        image = data['img'].data[0]
        
        trtModel_pre = TRTModel_pre(model)
        image_infer = image.to(device)
        mlvl_feat = trtModel_pre.forward(image_infer)

        _, c_, h_, w_ = mlvl_feat.shape

        mlvl_feat = mlvl_feat.reshape(trtModel_pre.batch_size, -1, c_, h_, w_ )
    
        _, out, x2, y2= get_project_output2(image, metas, mlvl_feat)
        
        tmp = torch.ones(out.shape).to(device)
        out2 = tmp * out


        names = ["FRONT", "FRONT_RIGHT", "FRONT_LEFT", "BACK", "BACK_LEFT", "BACK_RIGHT"]
        j=0
        for file in data['img_metas'].data[0][0]['img_info']:
            path = file["filename"]
            shutil.copyfile(path, f"{root}/{j}-{names[j]}.jpg")
            j=j+1


        mlvl_anchors = model.bbox_head.anchor_generator.grid_anchors([(100, 100)])
        mlvl_anchors = [anchor.reshape(-1, box_code_size) for anchor in mlvl_anchors]

        tensor.save(out2, f"{root}/valid_c_idx.tensor", True)
        tensor.save(x2, f"{root}/x.tensor", True)
        tensor.save(y2, f"{root}/y.tensor", True)
        tensor.save(mlvl_anchors[0], f"{root}/anchors.tensor", True)


if __name__ == "__main__":
    args = arg_parser()
    np.random.seed(0)
    torch.manual_seed(0)
    dump_tensor(args)
    


