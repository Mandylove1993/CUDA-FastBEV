import argparse
import json
import os
import pickle
import yaml
import cv2
import numpy as np
import torch
import re
from pyquaternion.quaternion import Quaternion

from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB


def read_txt_to_array(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    data_list = []
    for line in lines:
        elements = line.strip().split()
        data_list.extend(map(float, elements))

    data_arr = np.array(data_list)
    data_arr = data_arr.reshape(-1, 9)
    bboxes   = data_arr[:, :7]
    scores   = data_arr[:, 8]
    cls_arr  = data_arr[:, 7]

    return bboxes, scores, cls_arr


def read_data_file(file_path):
    info_dict = {}
    data = torch.load(os.path.join(file_path, "example-data.pth"))
    info_dict["cams"] = {}
    j = 0

    for file in data['img_metas'].data[0][0]['img_info']:
        file_arr = file['filename'].split('/')[-2]
        info_dict["cams"][file_arr]={}
        info_dict["cams"][file_arr]['img_path'] = f"{file_path}/{j}-{file_arr[4:]}.jpg"
        
        lidar2img_extra = data['img_metas'].data[0][0]['lidar2img']["lidar2img_extra"][j]
        lidar2img_aug = data['img_metas'].data[0][0]['lidar2img']["lidar2img_aug"][j]
        
        R = lidar2img_extra["sensor2lidar_rotation"]
        T = lidar2img_extra["sensor2lidar_translation"]
        I = lidar2img_extra["cam_intrinsic"]
        info_dict["cams"][file_arr]['sensor2lidar_rotation'] = R
        info_dict["cams"][file_arr]['sensor2lidar_translation'] = T
        info_dict["cams"][file_arr]['cam_intrinsic'] = I
        info_dict["cams"][file_arr]['lidar2img'] = {}
        info_dict["cams"][file_arr]['lidar2img']['extrinsic'] = data['img_metas'].data[0][0]['lidar2img']['extrinsic'][j]
        info_dict["cams"][file_arr]['lidar2img']['intrinsic'] = data['img_metas'].data[0][0]['lidar2img']['intrinsic'][:3,:3]
        info_dict["cams"][file_arr]['post_tran'] = lidar2img_aug['post_tran']
        info_dict["cams"][file_arr]['post_rot'] = lidar2img_aug['post_rot']
        
        j = j + 1
        
        
    return info_dict


def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(valid, np.logical_and(points[:, 0] < width, points[:, 1] < height))
    return valid


def depth2color(depth):
    gray = max(0, min((depth + 2.5) / 3.0, 1.0))
    max_lumi = 200
    colors = np.array([[max_lumi, 0, max_lumi], 
                       [max_lumi, 0, 0],
                       [max_lumi, max_lumi, 0],
                       [0, max_lumi, 0], 
                       [0, max_lumi, max_lumi], 
                       [0, 0, max_lumi]],
        dtype=np.float32)
    if gray == 1:
        return tuple(colors[-1].tolist())
    num_rank = len(colors) - 1
    rank     = np.floor(gray * num_rank).astype(np.int)
    diff     = (gray - rank / num_rank) * num_rank
    return tuple((colors[rank] + (colors[rank + 1] - colors[rank]) * diff).tolist())


def lidar2img(points_lidar, camrera_info):
    points_lidar_homogeneous  = np.concatenate([points_lidar, np.ones((points_lidar.shape[0], 1), dtype=points_lidar.dtype)], axis=1)
    
    lidar2camera              = camrera_info['lidar2img']['extrinsic']
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera             = points_camera_homogeneous[:, :3]
    valid                     = np.ones((points_camera.shape[0]), dtype=bool)
    valid                     = np.logical_and(points_camera[:, -1] > 0.5, valid)
    points_camera             = points_camera / points_camera[:, 2:3]
    camera2img                = camrera_info['lidar2img']['intrinsic']
    points_img                = points_camera @ camera2img.T
    post_aug                  = np.eye(3, dtype=np.float32)
    post_aug[:2, :2]          = camrera_info['post_rot'][:2,:2]
    post_aug[:2, 2]           = camrera_info['post_tran'][:2]
    points_img                = np.linalg.inv(post_aug) @ points_img.transpose(1,0)
    points_img                = points_img.transpose(1,0)[:, :2]
    return points_img, valid


def get_lidar2global(lidar2ego_rotation, lidar2ego_translation, ego2global_rotation, ego2global_translation):
    lidar2ego          = np.eye(4, dtype=np.float32)
    lidar2ego[:3, :3]  = Quaternion(lidar2ego_rotation).rotation_matrix
    lidar2ego[:3, 3]   = lidar2ego_translation
    ego2global         = np.eye(4, dtype=np.float32)
    ego2global[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
    ego2global[:3, 3]  = ego2global_translation
    return ego2global @ lidar2ego


def main(data_root, pred_path, vis_path):
    info_dict               = read_data_file(data_root)
    cam_info_dict           = info_dict['cams']

    bboxes, scores, cls_arr = read_txt_to_array(pred_path)

    # 定义绘制BEV视角下框的索引
    draw_boxes_indexes_bev      = [(0, 1), (1, 2), (2, 3), (3, 0)]
    draw_boxes_indexes_img_view = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), 
                                   (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    canva_size   = 1000
    show_range   = 50
    scale_factor = 4
    color_map    = {0: (255, 255, 0), 1: (0, 255, 255)}

    # 定义相机视角列表
    print('start visualizing results')
    pred_boxes        = np.array(bboxes, dtype=np.float32)
    corners_lidar     = LB(pred_boxes, origin=(0.5, 0.5, 0)).corners.numpy().reshape(-1, 3)

    pred_flag = np.ones((corners_lidar.shape[0] // 8, ), dtype=np.bool)
    scores    = np.array(scores, dtype=np.float32)

    # 构建预测框的标志数组
    sort_ids  = np.argsort(scores)

    # 对相机视角进行可视化
    imgs = []
    views = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    for view in views:
            
        img = cv2.imread(cam_info_dict[view]['img_path'])
        

        # 将雷达坐标转换为图像坐标并绘制目标框
        corners_img, valid = lidar2img(corners_lidar, cam_info_dict[view])
        valid = valid.reshape(-1, 8)
        corners_img = corners_img.reshape(-1, 8, 2).astype(np.int)
        for aid in range(valid.shape[0]):
            for index in draw_boxes_indexes_img_view:
                if valid[aid, index[0]] and valid[aid, index[1]]:
                    cv2.line(img, corners_img[aid, index[0]], corners_img[aid, index[1]],
                            color=color_map[int(pred_flag[aid])], thickness=scale_factor)
        imgs.append(img)

    # 构建BEV视图的画布
    canvas = np.zeros((int(canva_size), int(canva_size), 3), dtype=np.uint8)


    ## 绘制中心点和距离
    center_ego = (0, 0)
    center_canvas = int((center_ego[0] + show_range) / show_range / 2.0 * canva_size)
    cv2.circle(canvas, center=(center_canvas, center_canvas), radius=1, color=(255, 255, 255), thickness=0)
    dis = 10
    for r in range(dis, 100, dis):
        r_canvas = int(r / show_range / 2.0 * canva_size)
        cv2.circle(canvas, center=(center_canvas, center_canvas), radius=r_canvas, color=depth2color(r), thickness=0)
        
    
    # 绘制BEV视角下的预测框
    corners_lidar          = corners_lidar.reshape(-1, 8, 3)
    corners_lidar[:, :, 1] = -corners_lidar[:, :, 1]
    bottom_corners_bev     = corners_lidar[:, [0, 3, 7, 4], :2]
    bottom_corners_bev     = (bottom_corners_bev + show_range) / show_range / 2.0 * canva_size
    bottom_corners_bev     = np.round(bottom_corners_bev).astype(np.int32)
    center_bev             = corners_lidar[:, [0, 3, 7, 4], :2].mean(axis=1)
    head_bev               = corners_lidar[:, [0, 4], :2].mean(axis=1)
    canter_canvas          = (center_bev + show_range) / show_range / 2.0 * canva_size
    center_canvas          = canter_canvas.astype(np.int32)
    head_canvas            = (head_bev + show_range) / show_range / 2.0 * canva_size
    head_canvas            = head_canvas.astype(np.int32)

    # 在BEV视角下绘制预测框
    for rid in sort_ids:
        score = scores[rid]
        if score < 0.2 and pred_flag[rid]:
            continue
        score = min(score * 2.0, 1.0) if pred_flag[rid] else 1.0
        color = color_map[int(pred_flag[rid])]
        for index in draw_boxes_indexes_bev:
            cv2.line(canvas, bottom_corners_bev[rid, index[0]], bottom_corners_bev[rid, index[1]],
                    [color[0] * score, color[1] * score, color[2] * score], thickness=1)
        cv2.line(canvas, center_canvas[rid], head_canvas[rid], [color[0] * score, color[1] * score, color[2] * score], 1, lineType=8)

    # 融合图像视角和BEV视角的结果
    img = np.zeros((900 * 2 + canva_size * scale_factor, 1600 * 3, 3), dtype=np.uint8)
    img[:900, :, :] = np.concatenate(imgs[:3], axis=1)
    img_back = np.concatenate([imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]], axis=1)
    img[900 + canva_size * scale_factor:, :, :] = img_back
    img = cv2.resize(img, (int(1600 / scale_factor * 3), int(900 / scale_factor * 2 + canva_size)))
    w_begin = int((1600 * 3 / scale_factor - canva_size) // 2)
    img[int(900 / scale_factor):int(900 / scale_factor) + canva_size, w_begin:w_begin + canva_size, :] = canvas

    # 保存可视化结果
    cv2.imwrite(vis_path, img)
    print(f'Saved visual result to {vis_path}')


if __name__ == '__main__':

    data_root  = 'example-data'
    pred_path  = 'CUDA-FastBEV/model/resnet18/result.txt'
    vis_path   = os.path.join(data_root, "sample0_vis.png")

    main(data_root, pred_path, vis_path)