cmake_minimum_required(VERSION 3.10)
project(yolov8_openvino)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic -O3)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)

add_library(${PROJECT_NAME} SHARED src/yolov8.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} openvino::frontend::onnx openvino::runtime ${OpenCV_LIBS})

add_executable(yolov8 src/example.cpp)
target_include_directories(yolov8 PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolov8 ${PROJECT_NAME} ${OpenCV_LIBS})