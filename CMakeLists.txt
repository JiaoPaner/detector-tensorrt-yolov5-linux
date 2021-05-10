cmake_minimum_required(VERSION 2.6)

project(detector)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
#include_directories(/home/adminpc/jiaopan/app/tensorRT/TensorRT-7.0.0.11/include)
#link_directories(/home/adminpc/jiaopan/app/tensorRT/TensorRT-7.0.0.11/lib)
include_directories(include/tensorrt)
link_directories(${PROJECT_SOURCE_DIR}/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(plugins SHARED include/yololayer.cu)
target_link_libraries(plugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB src src/*.cpp src/*.c include/*.h include/*.hpp include/tensorrt/*.h)
set(SRC ${src})

add_executable(detector ${SRC})
#add_library(detector SHARED ${SRC})

target_link_libraries(detector ${OpenCV_LIBS} plugins nvinfer cudart)

add_definitions(-O2 -pthread)

