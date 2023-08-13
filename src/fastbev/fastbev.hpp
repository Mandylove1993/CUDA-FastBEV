/*
 * @Author: Mandy 
 * @Date: 2023-08-13 13:19:20 
 * @Last Modified by: Mandy
 * @Last Modified time: 2023-08-13 17:46:43
 */
#ifndef __FASTBEV_HPP__
#define __FASTBEV_HPP__

#include "fastbev_pre.hpp"
#include "vtransform.hpp"
#include "normalization.hpp"
#include "postprecess.hpp"

namespace fastbev {

struct CoreParameter {
  std::string pre_model;
  std::string post_model;
  pre::NormalizationParameter normalize;
  pre::GeometryParameter geo_param;
  post::transbbox::TransBBoxParameter transbbox;
};

class Core {
 public:
  virtual std::vector<post::transbbox::BoundingBox> forward(const unsigned char **camera_images, void *stream ) = 0;

  virtual void print() = 0;
  virtual void set_timer(bool enable) = 0;

  virtual void update(const float *valid_c_idx, const int64_t *valid_x, const int64_t *valid_y, void* stream = nullptr) = 0;

};

std::shared_ptr<Core> create_core(const CoreParameter &param);

};  // namespace fastbev

#endif  // __FASTBEV_HPP__