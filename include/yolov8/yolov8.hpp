// Copyright 2024 Chengfu Zou
#ifndef YOLOV8_HPP_
#define YOLOV8_HPP_

// std
#include <filesystem>
#include <string>
#include <utility>
#include <vector>
// thidr party
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

namespace yolov8 {

struct Params {
  float conf_thresh = 0.25;
  float nms_thresh = 0.45;
};

struct Object {
  std::string class_name;
  float class_score;
  cv::Rect bbox;
  std::vector<cv::Point2f> pts;
};

class YOLOv8 {
 public:
  YOLOv8(const std::filesystem::path &model_path,
         const std::vector<std::string> &labels, const std::string &device,
         const Params params = Params());

  // @brief Detect object keypoints
  // @param src Input image
  // @return std::vector<Object>, detected objects
  auto detect(const cv::Mat &src) -> std::vector<Object>;

 private:
  // @brief Process input image and do inference
  // @param src Input image
  // @return float, scale factor
  auto forward(const cv::Mat &src) -> float;

  // @brief Decode the output tensor
  // @param scale Scale factor
  // @return std::vector<Object>, detected objects
  auto decode(float scale) -> std::vector<Object>;

  std::vector<std::string> labels_;
  Params params_;
  ov::InferRequest infer_request_;
  ov::CompiledModel compiled_model_;
};

}  // namespace yolov8
#endif  // YOLOV8_HPP_
