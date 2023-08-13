/*
 * @Author: Mandy 
 * @Date: 2023-08-13 13:19:50 
 * @Last Modified by: Mandy
 * @Last Modified time: 2023-08-13 18:51:21
 */
#include <cuda_runtime.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <algorithm>

#include "fastbev/fastbev.hpp"
#include "common/check.hpp"
#include "common/tensor.hpp"
#include "common/timer.hpp"
#include "common/visualize.hpp"

const int final_height = 256;
const int final_weith = 704;

static std::vector<unsigned char*> load_images(const std::string& root) {
  const char* file_names[] = {"0-FRONT.jpg", "1-FRONT_RIGHT.jpg", "2-FRONT_LEFT.jpg",
                              "3-BACK.jpg",  "4-BACK_LEFT.jpg",   "5-BACK_RIGHT.jpg"};

  std::vector<unsigned char*> images;
  for (int i = 0; i < 6; ++i) {
    char path[200];
    sprintf(path, "%s/%s", root.c_str(), file_names[i]);

    int width, height, channels;
    images.push_back(stbi_load(path, &width, &height, &channels, 0));
  }
  return images;
}


static void free_images(std::vector<unsigned char*>& images) {
  for (size_t i = 0; i < images.size(); ++i) stbi_image_free(images[i]);

  images.clear();
}


std::shared_ptr<fastbev::Core> create_core(const std::string& model, const std::string& precision) {
  printf("Create by %s, %s\n", model.c_str(), precision.c_str());
  fastbev::pre::NormalizationParameter normalization;
  normalization.image_width = 1600;
  normalization.image_height = 900;
  normalization.output_width = 704;
  normalization.output_height = 256;
  normalization.num_camera = 6;
  normalization.resize_lim = 0.44f;
  normalization.interpolation = fastbev::pre::Interpolation::Nearest;

  float mean[3] = {123.675, 116.28, 103.53};
  float std[3] = {58.395, 57.12, 57.375};
  normalization.method = fastbev::pre::NormMethod::mean_std(mean, std, 1.0f, 0.0f);
  fastbev::pre::GeometryParameter geo_param;
  geo_param.feat_height = 64;
  geo_param.feat_width = 176;
  geo_param.num_camera =6;
  geo_param.valid_points = 160000;
  geo_param.volum_x = 200;
  geo_param.volum_y = 200;
  geo_param.volum_z = 4;

  fastbev::CoreParameter param;
  param.pre_model = nv::format("model/%s/build/fastbev_pre_trt.plan", model.c_str());
  param.normalize = normalization;
  param.post_model = nv::format("model/%s/build/fastbev_post_trt_decode.plan", model.c_str());
  param.geo_param = geo_param;
  return fastbev::create_core(param);
}


void SaveBoxPred(std::vector<fastbev::post::transbbox::BoundingBox> boxes, std::string file_name)
{
    std::ofstream ofs;
    ofs.open(file_name, std::ios::out);
    if (ofs.is_open()) {
        for (const auto box : boxes) {
          ofs << box.position.x << " ";
          ofs << box.position.y << " ";
          ofs << box.position.z << " ";
          ofs << box.size.w << " ";
          ofs << box.size.l << " ";
          ofs << box.size.h << " ";
          ofs << box.z_rotation << " ";
          ofs << box.id << " ";
          ofs << box.score << " ";
          ofs << "\n";
        }
    }
    else {
      std::cerr << "Output file cannot be opened!" << std::endl;
    }
    ofs.close();
    std::cout << "Saved prediction in: " << file_name << std::endl;
    return;
};

int main(int argc, char** argv){


    const char* data = "example-data";
    const char* model = "resnet18";
    const char* precision = "fp16";

    if (argc > 1) data      = argv[1];
    if (argc > 2) model     = argv[2];
    if (argc > 3) precision = argv[3];

    std::string Save_Dir = nv::format("model/%s/result", model);
    auto core = create_core(model, precision);
    if (core == nullptr) {
      printf("Core has been failed.\n");
      return -1;
    }
 
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    core->print();
    core->set_timer(true);

    auto images = load_images(data);

    auto valid_c_idx = nv::Tensor::load(nv::format("%s/valid_c_idx.tensor", data), false);
    auto valid_x = nv::Tensor::load(nv::format("%s/x.tensor", data), false);
    auto valid_y = nv::Tensor::load(nv::format("%s/y.tensor", data), false);
    core->update(valid_c_idx.ptr<float>(), valid_x.ptr<int64_t>(), valid_y.ptr<int64_t>(), stream);
    // warmup
    auto bboxes = core->forward((const unsigned char**)images.data(), stream);
    
    // evaluate inference time
    for (int i = 0; i < 5; ++i) {
      core->forward((const unsigned char**)images.data(), stream);
    }

    std::string save_file_name = Save_Dir + ".txt";
    SaveBoxPred(bboxes, save_file_name);

    // destroy memory
    free_images(images);
    checkRuntime(cudaStreamDestroy(stream));
    return 0;
}