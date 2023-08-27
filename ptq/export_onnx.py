import argparse
from argparse import ArgumentParser
import math
import copy
import torch
import torch.nn as nn
import onnx
import onnxsim
from onnxsim import simplify
from mmseg.ops import resize
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
# import warnings
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.apis import init_model
import lean.quantize as quantize
import os
import numpy as np
from ptq_bev import quantize_net, fuse_conv_bn
from lean import tensor

box_code_size = 9
cfg_n_voxels=[[200, 200, 4]]
cfg_voxel_size=[[0.5, 0.5, 1.5]]
nv = 6

def simplify_onnx(onnx_path):
    onnx_model = onnx.load(onnx_path)
    model_simp, check = simplify(onnx_model)
    assert check, "simplify onnx model fail!"
    onnx.save(model_simp, onnx_path)
    print("finish simplify onnx!")


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    points = torch.stack(
        torch.meshgrid(
            [
                torch.arange(n_voxels[0]),
                torch.arange(n_voxels[1]),
                torch.arange(n_voxels[2]),
            ]
        )
    )
    new_origin = origin - n_voxels / 2.0 * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


def compute_projection(img_meta, stride, noise=0):
    projection = []
    intrinsic = torch.tensor(img_meta["lidar2img"]["intrinsic"][:3, :3])
    intrinsic[:2] /= stride
    extrinsics = map(torch.tensor, img_meta["lidar2img"]["extrinsic"])
    for extrinsic in extrinsics:
        if noise > 0:
            projection.append(intrinsic @ extrinsic[:3] + noise)
        else:
            projection.append(intrinsic @ extrinsic[:3])
    return torch.stack(projection)


def get_project_output(img, img_metas, mlvl_feats):
    stride_i = math.ceil(img.shape[-1] / mlvl_feats.shape[-1])
    mlvl_feat_split = torch.split(mlvl_feats, nv, dim=1)

    volume_list = []
    for seq_id in range(len(mlvl_feat_split)):
        volumes = []
        for batch_id, seq_img_meta in enumerate(img_metas):
            feat_i = mlvl_feat_split[seq_id][batch_id] 
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

            points = get_points(
                n_voxels=torch.tensor(n_voxels),
                voxel_size=torch.tensor(voxel_size),
                origin=torch.tensor(img_meta["lidar2img"]["origin"]),
            ).to(feat_i.device)

            volume = backproject_inplace(
                feat_i[:, :, :height, :width], points, projection) 
            volumes.append(volume)
        volume_list.append(torch.stack(volumes))
    mlvl_volumes = torch.cat(volume_list, dim=1)

    return mlvl_volumes


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
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]

    # method2：特征填充，只填充有效特征，重复特征直接覆盖
    volume = torch.zeros(
        (n_channels, points.shape[-1]), device=features.device
    ).type_as(features)
    for i in range(n_images):
        volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]

    volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume


def decode(anchors, deltas):
    """Apply transformation `deltas` (dx, dy, dz, dw, dh, dl, dr, dv*) to
    `boxes`.

    Args:
        anchors (torch.Tensor): Parameters of anchors with shape (N, 7).
        deltas (torch.Tensor): Encoded boxes with shape
            (N, 7+n) [x, y, z, w, l, h, r, velo*].

    Returns:
        torch.Tensor: Decoded boxes.
    """
    cas, cts = [], []
    xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
    xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(deltas, 1, dim=-1)

    za = za + ha / 2
    diagonal = torch.sqrt(la**2 + wa**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za

    lg = torch.exp(lt) * la
    wg = torch.exp(wt) * wa
    hg = torch.exp(ht) * ha
    rg = rt + ra
    zg = zg - hg / 2
    cgs = [t + a for t, a in zip(cts, cas)]
    return torch.cat([xg, yg, zg, wg, lg, hg, rg, *cgs], dim=-1)


class TRTModel_pre(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.seq = 1
        self.nv = 6
        self.batch_size = 1

    def forward(self, img):
        img = img.reshape([-1] + list(img.shape)[2:])
        x = self.model.backbone(img)
        mlvl_feats = self.model.neck(x)
        mlvl_feats = list(mlvl_feats)

        if self.model.multi_scale_id is not None:
            mlvl_feats_ = []
            for msid in self.model.multi_scale_id:
                # fpn output fusion
                if getattr(self.model, f'neck_fuse_{msid}', None) is not None:
                    fuse_feats = [mlvl_feats[msid]]
                    for i in range(msid + 1, len(mlvl_feats)):
                        resized_feat = resize(
                            mlvl_feats[i], 
                            size=mlvl_feats[msid].size()[2:], 
                            mode="bilinear", 
                            align_corners=False)
                        fuse_feats.append(resized_feat)
                
                    if len(fuse_feats) > 1:
                        fuse_feats = torch.cat(fuse_feats, dim=1)
                    else:
                        fuse_feats = fuse_feats[0]
                    fuse_feats = getattr(self.model, f'neck_fuse_{msid}')(fuse_feats)
                    mlvl_feats_.append(fuse_feats)
                else:
                    mlvl_feats_.append(mlvl_feats[msid])
            mlvl_feats = mlvl_feats_
        # v3 bev ms
        if isinstance(self.model.n_voxels, list) and len(mlvl_feats) < len(self.model.n_voxels):
            pad_feats = len(self.model.n_voxels) - len(mlvl_feats)
            for _ in range(pad_feats):
                mlvl_feats.append(mlvl_feats[0])

        # only support one layer feature
        assert len(mlvl_feats) == 1, "only support one layer feature !"
        mlvl_feat =  mlvl_feats[0]

        return mlvl_feat


class TRTModel_post(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.num_levels = 1
        self.anchors = tensor.load("example-data/anchors.tensor", return_torch=True)
        self.anchors = self.anchors.to(device)
        self.nms_pre = 1000

    def forward(self, mlvl_volumes):
        neck_3d_feature = self.model.neck_3d.forward(mlvl_volumes.to(self.device))
        cls_scores, bbox_preds, dir_cls_preds = self.model.bbox_head(neck_3d_feature)
      
        cls_score = cls_scores[0][0]
        bbox_pred = bbox_preds[0][0]
        dir_cls_pred = dir_cls_preds[0][0]
        
        dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        dir_cls_scores = torch.max(dir_cls_pred, dim=-1)[1]
        
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.model.bbox_head.num_classes)
        cls_score = cls_score.sigmoid()
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, self.model.bbox_head.box_code_size)
        
        max_scores, _ = cls_score.max(dim=1)
        _, topk_inds = max_scores.topk(self.nms_pre)
        anchors = self.anchors[topk_inds, :]
        bbox_pred_ = bbox_pred[topk_inds, :]
        scores = cls_score[topk_inds, :]
        dir_cls_score = dir_cls_scores[topk_inds]
        bboxes = decode(anchors, bbox_pred_)
        
        return scores, bboxes, dir_cls_score
        

def main():
    parser = ArgumentParser()

    parser.add_argument('--config', default="configs/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f1.py", help='Config file')
    parser.add_argument('--checkpoint', default="ptq/pth/bev_ptq_head.pth", help='Checkpoint file')

    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--outfile', type=str, default='model/resnet18int8head/', help='dir to save results')
    parser.add_argument(
        '--ptq', default=True, help='ptq or qat')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, device=args.device)
    model_int8 = quantize_net(model)
    model_int8 = fuse_conv_bn(model_int8)
    
    if args.ptq:
        ckpt     = torch.load(args.checkpoint, map_location=args.device)
        model_int8.load_state_dict(ckpt.module.state_dict(), strict =True)
    else:
        from mmcv.runner import load_checkpoint
        load_checkpoint(model_int8, args.checkpoint, map_location=args.device)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False)

    def get_input_meta(data_loader):
        data=None
        for i , data in enumerate(data_loader):
            if i >= 1:
                break
            data = data
            
        image = data['img'].data[0]
        img_metas = data["img_metas"].data[0]

        return image, img_metas

    device = next(model_int8.parameters()).device
    image, img_metas = get_input_meta(data_loader)

    image_input = torch.tensor(image).to(device)
    trtModel_pre = TRTModel_pre(model_int8)
    trtModel_pre.eval()
    output_names_pre = ['mlvl_feat']

    pre_onnx_path = os.path.join(args.outfile, 'fastbev_pre_trt_ptq.onnx')
    quantize.quant_nn.TensorQuantizer.use_fb_fake_quant = True
    torch.onnx.export(
        trtModel_pre,
        (image_input,),
        pre_onnx_path,
        input_names=['image'],
        output_names=output_names_pre,
        opset_version=13,
        enable_onnx_checker=False,
        training= torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
    )

    mlvl_feat = trtModel_pre.forward(image_input)
    _, c_, h_, w_ = mlvl_feat.shape
    mlvl_feat = mlvl_feat.reshape(trtModel_pre.batch_size, -1, c_, h_, w_ )
    mlvl_volumes = get_project_output(image_input, img_metas, mlvl_feat)

    mlvl_volume = mlvl_volumes.to(device)

    trtModel_post = TRTModel_post(model_int8, device)
    output_names_post = ["cls_score", "bbox_pred", "dir_cls_preds"]

    post_onnx_path = os.path.join(args.outfile, 'fastbev_post_trt_ptq.onnx')
    torch.onnx.export(
        trtModel_post,
        (mlvl_volume,),
        post_onnx_path,
        input_names=['mlvl_volume'],
        output_names=output_names_post,
        opset_version=13,
        enable_onnx_checker=False,
    )

    simplify_onnx(pre_onnx_path)
    simplify_onnx(post_onnx_path)


if __name__ == '__main__':
    main()

