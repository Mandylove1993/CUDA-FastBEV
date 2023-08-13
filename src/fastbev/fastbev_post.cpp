/*
 * @Author: Mandy 
 * @Date: 2023-08-13 13:10:31 
 * @Last Modified by: Mandy
 * @Last Modified time: 2023-08-13 17:17:42
 */
#include <cuda_fp16.h>

#include <algorithm>
#include <numeric>

#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"
#include "fastbev_post.hpp"

namespace fastbev {
namespace post {

class TransfusionImplement : public Transfusion {
 public:
  virtual ~TransfusionImplement() {
    if(bindings_.cls_scores){checkRuntime(cudaFree(bindings_.cls_scores));}
    if(bindings_.dir_cls_scores){checkRuntime(cudaFree(bindings_.dir_cls_scores));}
    if(bindings_.bbox_preds){checkRuntime(cudaFree(bindings_.bbox_preds));}
  }

  virtual bool init(const std::string& model) {
    engine_ = TensorRT::load(model);
    if (engine_ == nullptr) return false;

    if (engine_->has_dynamic_dim()) {
      printf("Dynamic shapes are not supported.\n");
      return false;
    }

    create_binding_memory();
    return true;
  }


  void create_binding_memory() {
    for (int ibinding = 0; ibinding < engine_->num_bindings(); ++ibinding) {
      if (engine_->is_input(ibinding)) continue;

      auto shape = engine_->static_dims(ibinding);
      
      size_t volumn = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
      if(ibinding == 1){
        float* pdata = nullptr;
        checkRuntime(cudaMalloc(&pdata, volumn * sizeof(float)));
        bindings_.cls_scores = pdata;
      }
      if(ibinding == 2){
        int32_t* pdata = nullptr;
        checkRuntime(cudaMalloc(&pdata, volumn * sizeof(int32_t)));
        bindings_.dir_cls_scores = pdata;
      }
      if(ibinding == 3){
        float* pdata = nullptr;
        checkRuntime(cudaMalloc(&pdata, volumn * sizeof(float)));
        bindings_.bbox_preds = pdata;
      }

      bindshape_.push_back(shape);
    }
    Assertf(bindshape_.size() == 3, "Invalid output num of bindings[%d]", static_cast<int>(bindshape_.size()));
  }

  virtual void print() override { engine_->print("Transfusion"); }

  virtual std::vector<std::vector<int>> output_shape() override { return bindshape_; }

  virtual BindingOut forward(const nvtype::half* camera_bev, void* stream) override {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    
    engine_->forward({/* input  */ camera_bev,
                      /* output */ bindings_.cls_scores, bindings_.dir_cls_scores, bindings_.bbox_preds},
                     _stream);

    return bindings_;
  }

 private:
  std::shared_ptr<TensorRT::Engine> engine_;
  std::vector<std::vector<int>> bindshape_;
  BindingOut bindings_;
};

std::shared_ptr<Transfusion> create_transfusion(const std::string& param) {
  std::shared_ptr<TransfusionImplement> instance(new TransfusionImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace post
};  // namespace fastbev