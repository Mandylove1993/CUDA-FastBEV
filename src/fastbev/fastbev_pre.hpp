/*
 * @Author: Mandy 
 * @Date: 2023-08-13 13:19:07 
 * @Last Modified by:   Mandy 
 * @Last Modified time: 2023-08-13 13:19:07 
 */
#ifndef __CAMERA_BACKBONE_HPP__
#define __CAMERA_BACKBONE_HPP__

#include <memory>
#include <string>
#include <vector>
#include "common/tensor.hpp"

#include "common/dtype.hpp"
#include <opencv2/opencv.hpp>

namespace fastbev {
namespace pre {

class Backbone {
 public:
  virtual void forward(const nvtype::half* images, void* stream = nullptr) = 0;
  virtual nvtype::half* feature() = 0;
  virtual std::vector<int> feature_shape() = 0;
  virtual void print() = 0;
};

std::shared_ptr<Backbone> create_backbone(const std::string& model);

};  // namespace pre
};  // namespace fastbev

#endif  // __CAMERA_BACKBONE_HPP__