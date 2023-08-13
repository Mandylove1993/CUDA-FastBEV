/*
 * @Author: Mandy 
 * @Date: 2023-08-13 13:19:34 
 * @Last Modified by:   Mandy 
 * @Last Modified time: 2023-08-13 13:19:34 
 */
#ifndef __HEAD_TRANSBBOX_HPP__
#define __HEAD_TRANSBBOX_HPP__

#include <memory>
#include <string>
#include <vector>

#include "common/dtype.hpp"
#include "fastbev_post.hpp"

namespace fastbev {
namespace post {
namespace transbbox {

template<typename T> std::vector<int> argsort(const std::vector<T>& array)
{
    const int array_len(array.size());
    std::vector<int> array_index(array_len, 0);
    for (int i = 0; i < array_len; ++i)
        array_index[i] = i;

    std::sort(array_index.begin(), array_index.end(),
        [&array](int pos1, int pos2) {return (array[pos1] > array[pos2]); });

    return array_index;
}

struct TransBBoxParameter {
  bool sorted_bboxes = true;
  float confidence_threshold = 0.0f;
  float score_thr = 0.5f;
  int max_num = 500;
  float dir_offset = 0.7854;
  float dir_limit_offset = 0.0f;
  float nms_radius_thr_list[10] = {4.0f, 12.0f, 10.0f, 10.0f, 12.0f, 0.85f, 0.85f, 0.175f, 0.175f, 1.0f};
  float nms_thr_list[10] = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.5f, 0.5f, 0.2f};
  float nms_rescale_factor[10] = {1.0f, 0.7f, 0.55f, 0.4f, 0.7f, 1.0f, 1.0f, 4.5f, 9.0f, 1.0f};
};

struct Position {
  float x, y, z;
};

struct Size {
  float w, l, h;  // x, y, z
};

struct Velocity {
  float vx, vy;
};

struct BoundingBox {
  Position position;
  Size size;
  Velocity velocity;
  float z_rotation;
  float score;
  int id;
};

class TransBBox {
 public:
  virtual std::vector<BoundingBox> forward(const BindingOut bindings, void* stream,
                                           bool sorted_by_conf = false) = 0;
};

std::shared_ptr<TransBBox> create_transbbox(const TransBBoxParameter& param, std::vector<std::vector<int>> bingingshape);

};  // namespace transbbox
};  // namespace post
};  // namespace fastbev

#endif  // __HEAD_TRANSBBOX_HPP__