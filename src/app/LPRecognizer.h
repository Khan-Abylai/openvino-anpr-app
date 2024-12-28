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

class Recognizer {
public:
    Recognizer();
    ~Recognizer();

    std::vector<std::pair<std::string, float>> predict(const std::vector<cv::Mat> &frames);

private:
    const int
            SEQUENCE_SIZE = 30,
            ALPHABET_SIZE = 37,
            BLANK_INDEX = 0,
            IMG_WIDTH = 128,
            IMG_HEIGHT = 32,
            IMG_CHANNELS = 3,
            INPUT_SIZE = IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH,
            OUTPUT_SIZE = SEQUENCE_SIZE * ALPHABET_SIZE,
            MAX_BATCH_SIZE = 1,
            MAX_PLATE_SIZE = 12;
    const std::string
            ALPHABET = "-0123456789abcdefghijklmnopqrstuvwxyz",
            WEIGHTS_NAME = "../models/recognizer.xml";
    
    [[nodiscard]] std::vector<float> prepareImage(const std::vector<cv::Mat> &frames) const;
    std::string classIdToString(size_t class_id);
    ov::CompiledModel model_;
    ov::InferRequest infer_request_;

    std::vector<float> softmax(std::vector<float> &score_vec) {
        std::vector<float> softmax_vec(ALPHABET_SIZE);
        double score_max = *(max_element(score_vec.begin(), score_vec.end()));
        double e_sum = 0;
        for (int j = 0; j < ALPHABET_SIZE; j++) {
            softmax_vec[j] = exp((double) score_vec[j] - score_max);
            e_sum += softmax_vec[j];
        }
    for (int k = 0; k < ALPHABET_SIZE; k++)
        softmax_vec[k] /= e_sum;
    return softmax_vec;
}
};