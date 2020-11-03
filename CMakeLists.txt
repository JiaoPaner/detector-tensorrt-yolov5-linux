cmake_minimum_required(VERSION 3.14)
project(detector)

set(CMAKE_CXX_STANDARD 11)
add_definitions(-std=c++11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include include/common include/tensorrt)

link_directories(lib)

#CUDA
find_package(CUDA 10.2 REQUIRED)
message(“CUDA_LIBRARIES:${CUDA_LIBRARIES}”)
message(“CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}”)
include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
cuda_add_library(layer SHARED ${PROJECT_SOURCE_DIR}/src/yolo_layer.cu)

file(GLOB src src/*.cpp src/*.c src/*.cu include/*.h include/*.hpp include/tensorrt/*.h)
set(SRC ${src})

add_executable(detector ${SRC})

target_link_libraries(detector ${OpenCV_LIBS} ${CUDA_LIBRARIES} layer nvinfer nvparsers nvinfer_plugin nvonnxparser)