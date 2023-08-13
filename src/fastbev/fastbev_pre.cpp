/*
 * @Author: Mandy 
 * @Date: 2023-08-13 13:18:55 
 * @Last Modified by:   Mandy 
 * @Last Modified time: 2023-08-13 13:18:55 
 */
#include <cuda_fp16.h>

#include <numeric>

#include "fastbev_pre.hpp"
#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"

namespace fastbev {
namespace pre {

class BackboneImplement : public Backbone {
 public:
  virtual ~BackboneImplement() {
    if (feature_) checkRuntime(cudaFree(feature_));
  }

  bool init(const std::string& model) {
    engine_ = TensorRT::load(model);
    if (engine_ == nullptr) return false;

    feature_dims_ = engine_->static_dims(1);
    int32_t volumn = std::accumulate(feature_dims_.begin(), feature_dims_.end(), 1, std::multiplies<int32_t>());
    checkRuntime(cudaMalloc(&feature_, volumn * sizeof(nvtype::half)));

    return true;
  }
  
  virtual void print() override { engine_->print("Camerea Backbone"); }

  virtual void forward(const nvtype::half* images, void* stream = nullptr) override {


    engine_->forward({images, feature_}, static_cast<cudaStream_t>(stream));
  }

  virtual nvtype::half* feature() override { return feature_; }
  virtual std::vector<int> feature_shape() override { return feature_dims_; }
 
 private:
  std::shared_ptr<TensorRT::Engine> engine_;
  nvtype::half* feature_ = nullptr;
  std::vector<int> feature_dims_;
};

std::shared_ptr<Backbone> create_backbone(const std::string& model) {
  std::shared_ptr<BackboneImplement> instance(new BackboneImplement());
  if (!instance->init(model)) {
    instance.reset();
  }
  return instance;
}

};  // namespace pre
};  // namespace fastbev