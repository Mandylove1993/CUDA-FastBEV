#!/bin/bash
cmake_minimum_required(VERSION 2.8.12)
project(fastbev)

set(CMAKE_BUILD_PYTHON $ENV{USE_Python})
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})


find_package(CUDA REQUIRED)
find_package(Protobuf REQUIRED)
find_package(OpenCV)

set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB")
set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++14 -O0 -g -DENABLE_TEXT_BACKEND_STB")

set(CUDA_NVCC_FLAGS_RELEASE "-Werror=all-warnings -Xcompiler -std=c++14,-Wextra,-Wall,-Wno-deprecated-declarations,-O3 -DENABLE_TEXT_BACKEND_STB")
set(CUDA_NVCC_FLAGS_DEBUG   "-std=c++14 -O0 -g -DENABLE_TEXT_BACKEND_STB")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
  -gencode arch=compute_$ENV{CUDASM},code=compute_$ENV{CUDASM}
)

include_directories(
  ${Protobuf_INCLUDE_DIRS}
  $ENV{CUDA_Inc}
  $ENV{TensorRT_Inc}
  $ENV{Python_Inc}
  # ${OpenCV_INCLUDE_DIRS}
  ../dependencies/stb
  ../dependencies/pybind11/include
  ../libraries/cuOSD/src
  ../libraries/3DSparseConvolution/libspconv/include
  src/common
  src/
)

link_directories(
  $ENV{CUDA_Lib}
  $ENV{TensorRT_Lib}
  ../libraries/3DSparseConvolution/libspconv/lib/${arch}
  build
  $ENV{Python_Lib}
)

file(GLOB_RECURSE CORE_FILES 
  src/fastbev/*.cu 
  src/fastbev/*.cpp
  src/common/tensor.cu
  src/common/tensorrt.cpp
)

file(GLOB_RECURSE VISUALIZE_FILES 
  ../libraries/cuOSD/src/*.cpp
  ../libraries/cuOSD/src/*.cu
  src/common/visualize.cu
)

####################### libfastbev_core.so ##########################
cuda_add_library(fastbev_core SHARED 
  ${CORE_FILES}
)

target_link_libraries(fastbev_core
  libcudart.so
  libcublasLt.so
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
  ${OpenCV_LIBS}
)
#######################################################################


####################### fastbev program #############################
cuda_add_executable(fastbev 
  src/main.cpp
  ${VISUALIZE_FILES}
)

target_link_libraries(fastbev
  fastbev_core
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
  ${OpenCV_LIBS}
)
#######################################################################


########################## libpybev.so ################################
if(${CMAKE_BUILD_PYTHON} STREQUAL "ON")
  cuda_add_library(pybev SHARED
    src/python.cpp
  )

  target_link_libraries(pybev
    fastbev_core
    libnvinfer.so
    libspconv.so
    ${Protobuf_LIBRARIES}
    libnvinfer_plugin.so
    $ENV{Python_Soname}
    ${OpenCV_LIBS}
  )
endif()
#######################################################################