/*
 * @Author: Mandy 
 * @Date: 2023-08-13 13:10:57 
 * @Last Modified by:   Mandy 
 * @Last Modified time: 2023-08-13 13:10:57 
 */
#ifndef __FUSEHEAD_HPP__
#define __FUSEHEAD_HPP__

#include <memory>
#include <string>
#include <vector>

#include "common/dtype.hpp"

namespace fastbev {
namespace post {

struct BindingOut{
  float* cls_scores;
  int32_t* dir_cls_scores;
  float* bbox_preds;
};

class Transfusion {
 public:
  virtual BindingOut forward(const nvtype::half* camera_bev, void* stream) = 0;
  virtual void print() = 0;
  virtual std::vector<std::vector<int>> output_shape() = 0;
};

std::shared_ptr<Transfusion> create_transfusion(const std::string& model);

};  // namespace post
};  // namespace fastbev

#endif  // __FUSEHEAD_HPP__