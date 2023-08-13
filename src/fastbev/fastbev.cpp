/*
 * @Author: Mandy 
 * @Date: 2023-08-13 13:19:16 
 * @Last Modified by: Mandy
 * @Last Modified time: 2023-08-13 17:46:59
 */
#include "fastbev.hpp"

#include <numeric>

#include "common/check.hpp"
#include "common/timer.hpp"


namespace fastbev {

class CoreImplement : public Core {
 public:
  virtual ~CoreImplement() {
  }

  bool init(const CoreParameter& param) {
    camera_backbone_ = pre::create_backbone(param.pre_model);
    if (camera_backbone_ == nullptr) {
      printf("Failed to create camera backbone.\n");
      return false;
    }

    normalizer_ = pre::create_normalization(param.normalize);
    if (normalizer_ == nullptr) {
      printf("Failed to create normalizer.\n");
      return false;
    }

    vtransform_ =
        pre::create_vtrans(camera_backbone_->feature_shape(), param.geo_param);
    if (vtransform_ == nullptr) {
      printf("Failed to create camera bevpool.\n");
      return false;
    }

    fuse_head_ = post::create_transfusion(param.post_model);
    if (fuse_head_ == nullptr) {
      printf("Failed to create lidar scn.\n");
      return false;
    }

    transbbox_ = post::transbbox::create_transbbox(param.transbbox, fuse_head_->output_shape());
    if (transbbox_ == nullptr) {
      printf("Failed to create lidar scn.\n");
      return false;
    }

    param_ = param;
    return true;
  }

 std::vector<post::transbbox::BoundingBox> forward_only(const void* camera_images, void* stream, bool do_normalization) {
    nvtype::half* normed_images = (nvtype::half*)camera_images;
    if (do_normalization) {
      normed_images = (nvtype::half*)this->normalizer_->forward((const unsigned char**)(camera_images), stream);
    }
    this->camera_backbone_->forward(normed_images, stream);
    nvtype::half* camera_bev = this->vtransform_->forward(this->camera_backbone_->feature(), stream);
    auto fusion_feature = this->fuse_head_->forward(camera_bev, stream);
    return this->transbbox_->forward(fusion_feature, stream, param_.transbbox.sorted_bboxes);
 }

  std::vector<post::transbbox::BoundingBox> forward_timer(const void* camera_images, void* stream, bool do_normalization) {
    printf("==================FastBEV===================\n");
    std::vector<float> times;
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    nvtype::half* normed_images = (nvtype::half*)camera_images;
    if (do_normalization) {
      timer_.start(_stream);
      normed_images = (nvtype::half*)this->normalizer_->forward((const unsigned char**)(camera_images), stream);
      timer_.stop("[NoSt] ImageNrom");
    }

    timer_.start(_stream);
    this->camera_backbone_->forward(normed_images, stream);
    times.emplace_back(timer_.stop("Camera Backbone"));

    timer_.start(_stream);
    nvtype::half* camera_bev = this->vtransform_->forward(this->camera_backbone_->feature(), stream);
    times.emplace_back(timer_.stop("BEV vision transform"));

    timer_.start(_stream);
    auto fusion_feature = this->fuse_head_->forward(camera_bev, stream);
    times.emplace_back(timer_.stop("BEV fuse head"));

    timer_.start(_stream);
    auto bbox = this->transbbox_->forward(fusion_feature, stream, param_.transbbox.sorted_bboxes);
    times.emplace_back(timer_.stop("BEV postprocess"));

    float total_time = std::accumulate(times.begin(), times.end(), 0.0f, std::plus<float>{});
    printf("Total: %.3f ms\n", total_time);
    printf("=============================================\n");
    return bbox;
  }

  virtual std::vector<post::transbbox::BoundingBox> forward(const unsigned char** camera_images, void* stream) override {
    if (enable_timer_) {
      return this->forward_timer(camera_images, stream, true);
    } else {
      return this->forward_only(camera_images, stream, true);
    }
  }

  virtual void set_timer(bool enable) override { enable_timer_ = enable; }

  virtual void print() override {
    camera_backbone_->print();
  }

  virtual void update(const float *valid_c_idx, const int64_t *valid_x, const int64_t *valid_y, void* stream = nullptr) override {
    vtransform_->update(valid_c_idx, valid_x, valid_y, stream);
  }

 private:
  CoreParameter param_;
  nv::EventTimer timer_;
  std::shared_ptr<pre::Normalization> normalizer_;
  std::shared_ptr<pre::Backbone> camera_backbone_;
  std::shared_ptr<pre::VTransform> vtransform_;
  std::shared_ptr<post::Transfusion> fuse_head_;
  std::shared_ptr<post::transbbox::TransBBox> transbbox_;
  float confidence_threshold_ = 0;
  bool enable_timer_ = false;
};

std::shared_ptr<Core> create_core(const CoreParameter& param) {
  std::shared_ptr<CoreImplement> instance(new CoreImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

}; // namespace fastbev