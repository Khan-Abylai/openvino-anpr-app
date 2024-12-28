#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <array>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

namespace fs = std::filesystem;

class Detector {
public:
    Detector();
    ~Detector();

    cv::Mat detect(const cv::Mat& image);
private:
    ov::CompiledModel model_;
    ov::InferRequest infer_request_;    
    float threshold = 0.8f;
    float calculateIOU(const std::vector<float>& box1, const std::vector<float>& box2, bool x1y1x2y2 = false);

    std::vector<std::vector<float>> nms(const std::vector<std::vector<float>>& predictions, 
                                  float conf_thres = 0.7f, 
                                  float nms_thres = 0.4f);

    std::vector<std::vector<float>> postprocess(const std::vector<std::vector<float>>& predictions);

    std::vector<float> preprocess(const cv::Mat& image);

    const int IMG_WIDTH=512, IMG_HEIGHT=512, IMG_CHANNELS=3;
};