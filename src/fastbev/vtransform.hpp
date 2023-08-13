/*
 * @Author: Mandy 
 * @Date: 2023-08-13 13:19:43 
 * @Last Modified by: Mandy
 * @Last Modified time: 2023-08-13 17:45:23
 */
#ifndef __CAMERA_VTRANSFORM_HPP__
#define __CAMERA_VTRANSFORM_HPP__

#include <memory>
#include <string>
#include <vector>
#include "common/tensor.hpp"

#include "common/dtype.hpp"

namespace fastbev {
namespace pre {

struct GeometryParameter {
  int feat_width;
  int feat_height;
  int volum_z;
  int volum_x;
  int volum_y;
  int num_camera;
  int32_t valid_points;
};

class VTransform {
 public:
  virtual nvtype::half* forward(const nvtype::half* camera_feature, void* stream = nullptr) = 0;
  virtual void update(const float* valid_c_idx, const int64_t* valid_x, const int64_t* valid_y, void* stream = nullptr) = 0;
  virtual std::vector<int> shape() = 0;
};

std::shared_ptr<VTransform> create_vtrans(const std::vector<int>& feature_shape, GeometryParameter param);

};  // namespace pre
};  // namespace fastbev

#endif  // __CAMERA_VTRANSFORM_HPP__