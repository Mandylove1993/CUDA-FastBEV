/*
 * @Author: Mandy 
 * @Date: 2023-08-13 13:19:39 
 * @Last Modified by: Mandy
 * @Last Modified time: 2023-08-13 19:02:55
 */
#include <cuda_fp16.h>
#include <numeric>

#include "vtransform.hpp"
#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"


namespace fastbev {
namespace pre {

static __global__ void compute_volum_kernel(int num_valid, const half* camera_feature, const float* valid_index, const int64_t* valid_y, const int64_t* valid_x, int num_camera, int feat_height, int feat_width, half* output_feature) {
  int tid = cuda_linear_index;
  if (tid >= num_valid) return;

  for (int icamera = 0; icamera < num_camera; ++icamera) {
    int index = icamera * num_valid + tid;
    if(valid_index[index] == 1.0){
      int64_t x = valid_x[index];
      int64_t y = valid_y[index];
      for(int c=0; c< 64; c++){
        output_feature[c*num_valid+tid] = camera_feature[icamera*64*feat_height*feat_width+c*feat_height*feat_width +feat_width*y+x];
      }
    }
  }
}

class VtransformImplement : public VTransform {
 public:
  virtual ~VtransformImplement() {
    if (output_feature_) checkRuntime(cudaFree(output_feature_));
    if (input_feature_) checkRuntime(cudaFreeHost(input_feature_));
    if (valid_index_device_) checkRuntime(cudaFree(valid_index_device_));
    if (valid_x_device_) checkRuntime(cudaFree(valid_x_device_));
    if (valid_y_device_) checkRuntime(cudaFree(valid_y_device_));
  }


  bool init(const std::vector<int>& feature_shape, GeometryParameter param) {
    this->param_ = param;
    this->feature_shape_ = feature_shape;

    volumn_ = std::accumulate(feature_shape_.begin(), feature_shape_.end(), 1, std::multiplies<int32_t>());
    checkRuntime(cudaMallocHost(&input_feature_, volumn_ * sizeof(half)));

    unsigned int C = feature_shape_[1];
    volumn_output_ = C * param_.volum_x * param_.volum_y * param_.volum_z;
    output_dims_ = {1, (int)C, (int)param_.volum_x, (int)param_.volum_y, (int)param_.volum_z};
    checkRuntime(cudaMalloc(&output_feature_, volumn_output_ * sizeof(half)));
    
    bytes_of_valid_index_ = param_.valid_points * param_.num_camera * sizeof(float);
    checkRuntime(cudaMalloc(&valid_index_device_, bytes_of_valid_index_));
    bytes_of_x_or_y_index_ = param_.valid_points * param_.num_camera * sizeof(int64_t);
    checkRuntime(cudaMalloc(&valid_y_device_, bytes_of_x_or_y_index_));
    checkRuntime(cudaMalloc(&valid_x_device_, bytes_of_x_or_y_index_));
    
    return true;
  }


  virtual std::vector<int> shape() override { return output_dims_; }

  virtual void update(const float* valid_c_idx, const int64_t* valid_x, const int64_t* valid_y, void* stream = nullptr) override {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    checkRuntime(cudaMemcpyAsync(valid_index_device_,valid_c_idx, bytes_of_valid_index_, cudaMemcpyHostToDevice, _stream));
    checkRuntime(cudaMemcpyAsync(valid_y_device_,valid_y, bytes_of_x_or_y_index_, cudaMemcpyHostToDevice, _stream));
    checkRuntime(cudaMemcpyAsync(valid_x_device_,valid_x, bytes_of_x_or_y_index_, cudaMemcpyHostToDevice, _stream));

  }


  virtual nvtype::half* forward(const nvtype::half* camera_feature, void* stream = nullptr) override {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    checkRuntime(cudaMemsetAsync(output_feature_, 0.0, volumn_output_ * sizeof(half), _stream));
    cuda_linear_launch(compute_volum_kernel, _stream, param_.valid_points, reinterpret_cast<const half*>(camera_feature), valid_index_device_, valid_y_device_, valid_x_device_, param_.num_camera, param_.feat_height, param_.feat_width, output_feature_);
    return reinterpret_cast<nvtype::half*>(output_feature_);
  }
  

 private:
  size_t bytes_of_valid_index_ = 0;
  size_t bytes_of_x_or_y_index_ = 0;
  std::vector<int> feature_shape_; 
  half* output_feature_ = nullptr;
  half* input_feature_ = nullptr;
  std::vector<int> output_dims_;
  unsigned int volumn_output_ = 0;
  unsigned int volumn_ = 0;
  GeometryParameter param_;
  float* valid_index_device_ = nullptr;
  int64_t* valid_x_device_ = nullptr;
  int64_t* valid_y_device_ = nullptr;
};

std::shared_ptr<VTransform> create_vtrans(const std::vector<int>& feature_shape, GeometryParameter param) {
  std::shared_ptr<VtransformImplement> instance(new VtransformImplement());
  if (!instance->init(feature_shape, param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace pre
};  // namespace fastbev