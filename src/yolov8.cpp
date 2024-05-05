// Copyright 2024 Chengfu Zou

#include "yolov8/yolov8.hpp"

// std
#include <chrono>
#include <filesystem>
#include <iostream>
// third party
#include <opencv2/opencv.hpp>

#include "opencv2/dnn/dnn.hpp"

namespace yolov8 {

constexpr int INPUT_SIZE = 480;
constexpr int NUM_KEYPOINTS = 9;

auto letterbox(const cv::Mat &src) -> cv::Mat {
  int col = src.cols;
  int row = src.rows;
  int max = std::max(col, row);
  cv::Mat result = cv::Mat::zeros(max, max, CV_8UC3);
  src.copyTo(result(cv::Rect(0, 0, col, row)));
  return result;
}

YOLOv8::YOLOv8(const std::filesystem::path &model_path,
               const std::vector<std::string> &labels,
               const std::string &device, const Params params)
    : labels_(labels), params_(std::move(params)) {
  if (!std::filesystem::exists(model_path)) {
    throw std::runtime_error("Model path does not exist.");
  }
  auto core = ov::Core();
  std::shared_ptr<ov::Model> model = core.read_model(model_path.string());

  compiled_model_ = core.compile_model(model, device);
  infer_request_ = compiled_model_.create_infer_request();
  std::cout << "Model loaded successfully." << std::endl;
}

auto YOLOv8::forward(const cv::Mat &src) -> float {
  // Preprocess
  cv::Mat letterboxed = letterbox(src);
  float scale = static_cast<float>(letterboxed.size[0]) / INPUT_SIZE;
  cv::Mat blob = cv::dnn::blobFromImage(letterboxed, 1.0 / 255.0,
                                        cv::Size(INPUT_SIZE, INPUT_SIZE),
                                        cv::Scalar(), true);
  auto input_port = compiled_model_.input();

  // Create tensor from external memory
  ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(),
                          blob.ptr<float>(0));
  infer_request_.set_input_tensor(input_tensor);

  // Inference
  infer_request_.infer();
  return scale;
}

auto YOLOv8::detect(const cv::Mat &src) -> std::vector<Object> {
  auto t1 = std::chrono::high_resolution_clock::now();

  float scale = forward(src);

  auto result = decode(scale);

  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  std::cout << "Latency: " << duration << "ms" << std::endl;
  return result;
}

auto YOLOv8::decode(float scale) -> std::vector<Object> {
  std::vector<Object> result;

  const ov::Tensor &output = infer_request_.get_output_tensor();
  auto output_shape = output.get_shape();
  cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F,
                        output.data<float>());
  // [4725, 23] -> [23, 4725]
  cv::transpose(output_buffer, output_buffer);

  std::vector<int> class_ids;
  std::vector<float> class_scores;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<float>> objects_keypoints;
  for (int i = 0; i < output_buffer.rows; i++) {
    float class_score = output_buffer.at<float>(i, 4);

    if (class_score > params_.conf_thresh) {
      class_scores.push_back(class_score);
      class_ids.push_back(0);  // {0:"person"}
      float cx = output_buffer.at<float>(i, 0);
      float cy = output_buffer.at<float>(i, 1);
      float w = output_buffer.at<float>(i, 2);
      float h = output_buffer.at<float>(i, 3);
      // Get the box
      int left = static_cast<int>((cx - 0.5 * w) * scale);
      int top = static_cast<int>((cy - 0.5 * h) * scale);
      int width = static_cast<int>(w * scale);
      int height = static_cast<int>(h * scale);
      // Get the keypoints
      std::vector<float> keypoints;
      cv::Mat kpts = output_buffer.row(i).colRange(5, 23);
      for (int i = 0; i < NUM_KEYPOINTS; i++) {
        float x = kpts.at<float>(0, i * 2 + 0) * scale;
        float y = kpts.at<float>(0, i * 2 + 1) * scale;
        keypoints.push_back(x);
        keypoints.push_back(y);
      }

      boxes.push_back(cv::Rect(left, top, width, height));
      objects_keypoints.push_back(keypoints);
    }
  }
  // NMS
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, class_scores, params_.conf_thresh,
                    params_.nms_thresh, indices);

  for (int i : indices) {
    std::vector<cv::Point2f> keypoints;
    for (int j = 0; j < NUM_KEYPOINTS; j++) {
      keypoints.emplace_back(cv::Point2f(objects_keypoints[i][j * 2 + 0],
                                         objects_keypoints[i][j * 2 + 1]));
    }
    result.emplace_back(
        Object{labels_[class_ids[i]], class_scores[i], boxes[i], keypoints});
  }

  return result;
}
}  // namespace yolov8
