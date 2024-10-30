cmake_minimum_required(VERSION 3.0.2)
project(yolo11 LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE release)

# CUDA
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES OFF)
find_package(CUDA REQUIRED)
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")

# OpenCV
set(OpenCV_DIR /usr/local/lib/cmake/opencv4)  # Set OpenCV installation path
find_package(OpenCV 4.9.0 REQUIRED)

# TensorRT
set(TensorRT_INCLUDE_DIRS /home/cll/TensorRT/include)
set(TensorRT_LIBRARIES
        /home/cll/TensorRT/lib/libnvinfer.so
        /home/cll/TensorRT/lib/libnvinfer_plugin.so
        /home/cll/TensorRT/lib/libnvparsers.so
        /home/cll/TensorRT/lib/libnvonnxparser.so)

# cv_bridge
set(cv_bridge_DIR /usr/local/lib/cv_bridge/cmake)

find_package(catkin REQUIRED COMPONENTS
        cv_bridge
        roscpp
        rospy
        sensor_msgs
        std_msgs
)

catkin_package()

list(APPEND ALL_INCLUDE_DIRS
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
)

list(APPEND ALL_LIBS
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${TensorRT_LIBRARIES}
        ${catkin_LIBRARIES}
)


include_directories(include ${ALL_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}_ros
        src/main.cpp
        src/infer.cpp
        src/calibrator.cpp
        src/preprocess.cu
        src/postprocess.cu
)

target_include_directories(
        ${PROJECT_NAME}_ros
        PUBLIC
        ${ALL_INCLUDE_DIRS}
)

target_link_libraries(
        ${PROJECT_NAME}_ros
        PRIVATE
        ${ALL_LIBS}

)
