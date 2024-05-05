#include "yolov8/yolov8.hpp"

#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>"
              << std::endl;
    return 1;
  }
  // Load image
  cv::Mat src = cv::imread(argv[2]);
  if (src.empty()) {
    std::cerr << "Failed to load image." << std::endl;
    return 1;
  }

  // Load model
  std::filesystem::path model_path = argv[1];
  std::vector<std::string> labels = {"person"};
  yolov8::YOLOv8 detector(model_path, labels, "CPU");

  // Detect
  auto objects = detector.detect(src);
  for (const auto &obj : objects) {
    cv::rectangle(src, obj.bbox, cv::Scalar(50, 50, 255), 2);
    cv::putText(src, obj.class_name, cv::Point(obj.bbox.x, obj.bbox.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(50, 50, 255), 2);
    for (const auto &point : obj.pts) {
      cv::circle(src, cv::Point(point), 3, cv::Scalar(0, 255, 0), -1);
    }
  }
  cv::imshow("Result", src);
  cv::waitKey(0);

  return 0;
}
