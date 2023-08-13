/*
 * @Author: Mandy 
 * @Date: 2023-08-13 13:19:29 
 * @Last Modified by: Mandy
 * @Last Modified time: 2023-08-13 17:16:23
 */

#include <cuda_fp16.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <numeric>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"
#include "postprecess.hpp"

namespace fastbev {
namespace post {
namespace transbbox {

#define MAX_DETECTION_BOX_SIZE 1024
const float ThresHold = 1e-8;

static std::vector<std::vector<float>> xywhr2xyxyr(std::vector<std::vector<float>> boxes_xywhr){
    std::vector<std::vector<float>> boxes_xyxyr;
    std::vector<float> box_xyxyr(5, 0.0f);
    for(size_t i=0;i< boxes_xywhr.size();i++){
        float half_w = boxes_xywhr[i][2] / 2;
        float half_h = boxes_xywhr[i][3] / 2;
        box_xyxyr[0] = boxes_xywhr[i][0] - half_w;
        box_xyxyr[1] = boxes_xywhr[i][1] - half_h;
        box_xyxyr[2] = boxes_xywhr[i][0] + half_w;
        box_xyxyr[3] = boxes_xywhr[i][1] + half_h;
        box_xyxyr[4] = boxes_xywhr[i][4];
        boxes_xyxyr.push_back(box_xyxyr);

    }
    return boxes_xyxyr;
}

static float cross(const float2 p1, const float2 p2, const float2 p0) {
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

static int check_box2d(const std::vector<float> box, const float2 p) {
    const float MARGIN = 1e-5;
    float center_x = (box[0]+box[2])/2;
    float center_y = (box[1]+box[3])/2;
    float angle_cos = cos(-box[4]);
    float angle_sin = sin(-box[4]);
    float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (angle_sin) + center_x;
    float rot_y = -(p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos + center_y;

    return (rot_x > (box[0]- MARGIN) && rot_x < (box[2] + MARGIN) && rot_y > (box[1] - MARGIN) && rot_y < (box[3] + MARGIN));
}

bool intersection(const float2 p1, const float2 p0, const float2 q1, const float2 q0, float2 &ans) {

    if (( std::min(p0.x, p1.x) <= std::max(q0.x, q1.x) &&
          std::min(q0.x, q1.x) <= std::max(p0.x, p1.x) &&
          std::min(p0.y, p1.y) <= std::max(q0.y, q1.y) &&
          std::min(q0.y, q1.y) <= std::max(p0.y, p1.y) ) == 0)
        return false;


    float s1 = cross(q0, p1, p0);
    float s2 = cross(p1, q1, p0);
    float s3 = cross(p0, q1, q0);
    float s4 = cross(q1, p1, q0);

    if (!(s1 * s2 > 0 && s3 * s4 > 0))
        return false;

    float s5 = cross(q1, p1, p0);
    if (fabs(s5 - s1) > ThresHold) {
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);

    } else {
        float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
        float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
        float D = a0 * b1 - a1 * b0;

        ans.x = (b0 * c1 - b1 * c0) / D;
        ans.y = (a1 * c0 - a0 * c1) / D;
    }

    return true;
}

static void rotate_around_center(const float2 &center, const float angle_cos, const float angle_sin, float2 &p) {
    float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * (angle_sin) + center.x;
    float new_y = -(p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p = float2 {new_x, new_y};
    return;
}

static float box_overlap(const std::vector<float> &box_a, const std::vector<float> &box_b) {
    float a_angle = box_a[4], b_angle = box_b[4];
    float a_x1 = box_a[0], a_y1 = box_a[1];
    float a_x2 = box_a[2], a_y2 = box_a[3];
    float b_x1 = box_b[0], b_y1 = box_b[1];
    float b_x2 = box_b[2], b_y2 = box_b[3];
    float2 box_a_corners[5];
    float2 box_b_corners[5];

    float2 center_a = float2 {(box_a[0]+box_a[2])/ 2, (box_a[1]+box_a[3])/ 2};
    float2 center_b = float2 {(box_b[0]+box_b[2])/ 2, (box_b[1]+box_b[3])/ 2};

    float2 cross_points[16];
    float2 poly_center = {0, 0};
    int cnt = 0;
    bool flag = false;

    box_a_corners[0] = float2 {a_x1, a_y1};
    box_a_corners[1] = float2 {a_x2, a_y1};
    box_a_corners[2] = float2 {a_x2, a_y2};
    box_a_corners[3] = float2 {a_x1, a_y2};

    box_b_corners[0] = float2 {b_x1, b_y1};
    box_b_corners[1] = float2 {b_x2, b_y1};
    box_b_corners[2] = float2 {b_x2, b_y2};
    box_b_corners[3] = float2 {b_x1, b_y2};

    float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    for (int k = 0; k < 4; k++) {
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
    }

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            flag = intersection(box_a_corners[i + 1], box_a_corners[i],
                                box_b_corners[j + 1], box_b_corners[j],
                                cross_points[cnt]);
            if (flag) {
                poly_center = {poly_center.x + cross_points[cnt].x, poly_center.y + cross_points[cnt].y};
                cnt++;
            }
        }
    }

    for (int k = 0; k < 4; k++) {
        if (check_box2d(box_a, box_b_corners[k])) {
            poly_center = {poly_center.x + box_b_corners[k].x, poly_center.y + box_b_corners[k].y};
            cross_points[cnt] = box_b_corners[k];
            cnt++;
        }
        if (check_box2d(box_b, box_a_corners[k])) {
            poly_center = {poly_center.x + box_a_corners[k].x, poly_center.y + box_a_corners[k].y};
            cross_points[cnt] = box_a_corners[k];
            cnt++;
        }
    }

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    float2 temp;
    for (int j = 0; j < cnt - 1; j++) {
        for (int i = 0; i < cnt - j - 1; i++) {
            if (atan2(cross_points[i].y - poly_center.y, cross_points[i].x - poly_center.x) >
                atan2(cross_points[i+1].y - poly_center.y, cross_points[i+1].x - poly_center.x)
                ) {
                temp = cross_points[i];
                cross_points[i] = cross_points[i + 1];
                cross_points[i + 1] = temp;
            }
        }
    }

    float area = 0;
    for (int k = 0; k < cnt - 1; k++) {
        float2 a = {cross_points[k].x - cross_points[0].x,
                    cross_points[k].y - cross_points[0].y};
        float2 b = {cross_points[k + 1].x - cross_points[0].x,
                    cross_points[k + 1].y - cross_points[0].y};
        area += (a.x * b.y - a.y * b.x);
    }
    return fabs(area) / 2.0;
}

static std::vector<int> nms_cpu(std::vector<std::vector<float>> boxes, std::vector<float> score, const float nms_thresh)
{
    std::vector<int> order = argsort(score);
    std::vector<int> keep(boxes.size(), 0);
    std::vector<int> org_idx;

    for (size_t i = 0; i < boxes.size(); i++) {
        if (keep[i] == 1) {
            continue;
        }
        for (size_t j = i + 1; j < boxes.size(); j++) {
            if (keep[j] == 1) {
                continue;
            }

            float sa = (boxes[order[i]][2] - boxes[order[i]][0]) * (boxes[order[i]][3] - boxes[order[i]][1]);
            float sb = (boxes[order[j]][2] - boxes[order[j]][0]) * (boxes[order[j]][3] - boxes[order[j]][1]);
            float s_overlap = box_overlap(boxes[order[i]], boxes[order[j]]);
            float iou = s_overlap / fmaxf(sa + sb - s_overlap, ThresHold);
            if (iou >= nms_thresh) {
                keep[j] = 1;
            }
        }
    }

    for(size_t i =0; i<keep.size();i++){
        if (keep[i] == 1){
            continue;
        }
        org_idx.push_back(order[i]);
    }

    return org_idx;
}

static std::vector<int> circle_nms_cpu(std::vector<std::vector<float>> boxes, std::vector<float> score, const float nms_thresh)
{
    std::vector<int> order = argsort(score);
    std::vector<int> keep(boxes.size(), 0);
    std::vector<int> org_idx;

    for (size_t i = 0; i < boxes.size(); i++) {
        if (keep[i] == 1) {
            continue;
        }
        float da_x1 = boxes[order[i]][0];
        float da_y1 = boxes[order[i]][1];
        for (size_t j = i + 1; j < boxes.size(); j++) {
            if (keep[j] == 1) {
                continue;
            }
            
            float db_x1 = boxes[order[j]][0];
            float db_y1 = boxes[order[j]][1];
            float dist = pow(da_x1 - db_x1, 2) + pow(da_y1 - db_y1, 2);
            if (dist <= nms_thresh) {
                keep[j] = 1;
            }
        }
    }

    for(size_t i =0; i<keep.size();i++){
        if (keep[i] == 1){
            continue;
        }
        org_idx.push_back(order[i]);
    }

    return org_idx;
}


static float limit_period(float val, float offset){
    return (val - floor(val / M_PI + offset) * M_PI);
}

static std::vector<BoundingBox> box3d_multiclass_scale_nms(const float* mlvl_bboxes, const float* mlvl_bboxes_for_nms, const float* mlvl_scores, TransBBoxParameter param, const int32_t* mlvl_dir_scores, const int num_classes, const int points_length){
    
    std::vector<BoundingBox> result;

    for(int i = 0;i< num_classes;i++){
        std::vector<std::vector<float>> _bboxes_for_nms_class;
        std::vector<std::vector<float>> _mlvl_bboxes_class;
        std::vector<int32_t> _mlvl_dir_scores_class;
        std::vector<float> _scores_class;

        for (int j = 0;j< points_length;j++){
            if (mlvl_scores[j * num_classes + i] <= param.score_thr){
                continue;
            }

            float _scores = mlvl_scores[j * num_classes + i];
            float _dir_scores = mlvl_dir_scores[j];
            std::vector<float> _bboxes_for_nms;
            for(int k = 0;k< 5;k++){
                _bboxes_for_nms.push_back(mlvl_bboxes_for_nms[j * 5 + k]);
            }
            std::vector<float> _mlvl_bboxes;
            for(int k = 0;k< 9;k++){
                _mlvl_bboxes.push_back(mlvl_bboxes[j * 9 + k]);
            }
            _bboxes_for_nms_class.push_back(_bboxes_for_nms);
            _mlvl_bboxes_class.push_back(_mlvl_bboxes);
            _scores_class.push_back(_scores);
            _mlvl_dir_scores_class.push_back(_dir_scores);
            _bboxes_for_nms.clear();
            _mlvl_bboxes.clear();

        }

        float nms_rescale = param.nms_rescale_factor[i];
        for(size_t q=0; q < _bboxes_for_nms_class.size();q++){
            _bboxes_for_nms_class[q][2] = _bboxes_for_nms_class[q][2] * nms_rescale;
            _bboxes_for_nms_class[q][3] = _bboxes_for_nms_class[q][3] * nms_rescale;
        }

        std::vector<int> selected;
        if (i == 9){
            float nms_thre = param.nms_radius_thr_list[i];
            selected = circle_nms_cpu(_bboxes_for_nms_class, _scores_class, nms_thre);
        } else {
            float nms_thre = param.nms_thr_list[i];
            auto boxes_xyxyr = xywhr2xyxyr(_bboxes_for_nms_class);
            selected = nms_cpu(boxes_xyxyr, _scores_class, nms_thre);
        }

        BoundingBox BBox;

        for (size_t v = 0;v< selected.size(); v++){
            BBox.position.x = _mlvl_bboxes_class[selected[v]][0];
            BBox.position.y = _mlvl_bboxes_class[selected[v]][1];
            BBox.position.z = _mlvl_bboxes_class[selected[v]][2];
            BBox.size.w = _mlvl_bboxes_class[selected[v]][3];
            BBox.size.l = _mlvl_bboxes_class[selected[v]][4];
            BBox.size.h = _mlvl_bboxes_class[selected[v]][5];
            float dir_rot = limit_period(_mlvl_bboxes_class[selected[v]][6] - param.dir_offset, param.dir_limit_offset);
            BBox.z_rotation = dir_rot + param.dir_offset + M_PI * _mlvl_dir_scores_class[selected[v]];
            BBox.velocity.vx = _mlvl_bboxes_class[selected[v]][7];
            BBox.velocity.vy = _mlvl_bboxes_class[selected[v]][8];
            BBox.score =  _scores_class[selected[v]];    
            BBox.id = i;
            result.push_back(BBox);
        }

        _bboxes_for_nms_class.clear();
        _mlvl_bboxes_class.clear();
        _scores_class.clear();
        _mlvl_dir_scores_class.clear();
    }

    if(result.size() > (unsigned int)param.max_num){
        std::vector<BoundingBox> slice(result.begin(), result.begin() + param.max_num);
        return slice;
    } else{
        return result;
    }

}


class TransBBoxImplement : public TransBBox {
 public:
  virtual ~TransBBoxImplement() {
    if (cls_scores_host_) checkRuntime(cudaFreeHost(cls_scores_host_));
    if (bbox_preds_host_) checkRuntime(cudaFreeHost(bbox_preds_host_));
    if (dir_cls_scores_host_) checkRuntime(cudaFreeHost(dir_cls_scores_host_));
  }

  virtual bool init(const TransBBoxParameter& param, std::vector<std::vector<int>> bingingshape) {
    param_ = param;
    num_classes_ = bingingshape[0][1];

    std::vector<int> cls_score_shape = bingingshape[0];
    std::vector<int> bbox_preds_shape = bingingshape[2];
    std::vector<int> dir_cls_scores_shape = bingingshape[1];


    volumn_cls_scores_ = std::accumulate(cls_score_shape.begin(), cls_score_shape.end(), 1, std::multiplies<int32_t>());
    checkRuntime(cudaMallocHost(&cls_scores_host_, volumn_cls_scores_ * sizeof(float)));
    volumn_bbox_preds_ = std::accumulate(bbox_preds_shape.begin(), bbox_preds_shape.end(), 1, std::multiplies<int32_t>());
    checkRuntime(cudaMallocHost(&bbox_preds_host_, volumn_bbox_preds_ * sizeof(float)));
    volumn_dir_cls_scores_ = std::accumulate(dir_cls_scores_shape.begin(), dir_cls_scores_shape.end(), 1, std::multiplies<int32_t>());
    checkRuntime(cudaMallocHost(&dir_cls_scores_host_, volumn_dir_cls_scores_ * sizeof(int32_t)));
    checkRuntime(cudaMallocHost(&mlvl_bboxes_for_nms_, volumn_dir_cls_scores_ * 5 *sizeof(float)));

    return true;
  }

  virtual std::vector<BoundingBox> forward(const BindingOut bindings, void* stream,
                                           bool sorted) override {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    float* cls_scores = bindings.cls_scores;
    int32_t* dir_cls_scores = bindings.dir_cls_scores;
    float* bbox_preds = bindings.bbox_preds;

    checkRuntime(cudaMemcpyAsync(dir_cls_scores_host_, dir_cls_scores, volumn_dir_cls_scores_ * sizeof(int32_t), cudaMemcpyDeviceToHost, _stream));
    checkRuntime(cudaMemcpyAsync(cls_scores_host_, cls_scores, volumn_cls_scores_ * sizeof(float), cudaMemcpyDeviceToHost, _stream));
    checkRuntime(cudaMemcpyAsync(bbox_preds_host_, bbox_preds, volumn_bbox_preds_ * sizeof(float), cudaMemcpyDeviceToHost, _stream));
    checkRuntime(cudaStreamSynchronize(_stream));

 
    for(unsigned int i=0;i<volumn_dir_cls_scores_;i++){
        mlvl_bboxes_for_nms_[i * 5] = bbox_preds_host_[i * 9];
        mlvl_bboxes_for_nms_[i * 5 + 1] = bbox_preds_host_[i * 9 + 1];
        mlvl_bboxes_for_nms_[i * 5 + 2] = bbox_preds_host_[i * 9 + 3];
        mlvl_bboxes_for_nms_[i * 5 + 3] = bbox_preds_host_[i * 9 + 4];
        mlvl_bboxes_for_nms_[i * 5 + 4] = bbox_preds_host_[i * 9 + 6];
    }

    auto output = box3d_multiclass_scale_nms(bbox_preds_host_,mlvl_bboxes_for_nms_, cls_scores_host_, param_, dir_cls_scores_host_, num_classes_, volumn_dir_cls_scores_);
    if (sorted) {
      std::sort(output.begin(), output.end(), [](BoundingBox& a, BoundingBox& b) { return a.score > b.score; });
    }
    return output;
  }

 private:
  TransBBoxParameter param_;
  int num_classes_ = 0;
  float* cls_scores_host_ = nullptr;
  float* bbox_preds_host_ = nullptr;
  int32_t* dir_cls_scores_host_ = nullptr;
  unsigned int volumn_cls_scores_ = 0;
  unsigned int volumn_bbox_preds_ = 0;
  unsigned int volumn_dir_cls_scores_ = 0;
  float* mlvl_bboxes_for_nms_ = nullptr;

};

std::shared_ptr<TransBBox> create_transbbox(const TransBBoxParameter& param, std::vector<std::vector<int>> bingingshape) {
  std::shared_ptr<TransBBoxImplement> instance(new TransBBoxImplement());
  if (!instance->init(param, bingingshape)) {
    instance.reset();
  }
  return instance;
}

};  // namespace transbbox
};  // namespace post
};  // namespace fastbev