#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>

using namespace std;

std::vector<cv::Mat> preprocess(const std::string& path) {
        std::vector<cv::Mat> processed_images;
        std::vector<std::string> image_paths;

        // Check if path is directory or single file
        if (std::filesystem::is_directory(path)) {
            for (const auto& entry : std::filesystem::directory_iterator(path)) {
                image_paths.push_back(entry.path().string());
            }
        } else {
            image_paths.push_back(path);
        }

        // Process each image
        for (const auto& img_path : image_paths) {
            cv::Mat img = cv::imread(img_path);
            cv::Mat resized;
            cv::resize(img, resized, cv::Size(512, 512));
            
            // Convert to float32 and normalize
            cv::Mat float_img;
            resized.convertTo(float_img, CV_32F);
            float_img = 2 * (float_img / 254.5 - 0.5);
            
            processed_images.push_back(float_img);
        }

        return processed_images;
    };

ov::Tensor prepare_input(const cv::Mat& img) {
    int shape_ = 512;
        // Convert OpenCV Mat to OpenVINO Tensor
        ov::Shape input_shape = {1, 3, static_cast<size_t>(512), static_cast<size_t>(512)};
        ov::Tensor input_tensor(ov::element::f32, input_shape);
        
        float* input_data = input_tensor.data<float>();
        
        // OpenCV BGR to planar format
        std::vector<cv::Mat> channels(3);
        cv::split(img, channels);
        
        // Copy each channel
        size_t channel_size = shape_ * shape_;
        for (int c = 0; c < 3; c++) {
            memcpy(input_data + c * channel_size, 
                   channels[c].data, 
                   channel_size * sizeof(float));
        }
        
        return input_tensor;
    }

int main() {
    try {
        // Create OpenVINO Runtime Core 
        ov::Core core;

        // Print available devices
        std::cout << "\nAvailable devices:" << std::endl;
        for (auto&& device : core.get_available_devices()) {
            std::cout << device << std::endl;
        }

        std::cout << "\nOpenVINO is working correctly!" << std::endl;

        auto model = core.read_model("../models/detection.xml");
        ov::CompiledModel compiled_model = core.compile_model(model, "CPU");

        auto infer_request = compiled_model.create_infer_request();

        cv::Mat image = cv::imread("../images/test.jpeg");
        if (image.empty()) {
            throw std::runtime_error("Failed to read image");
        }

        std::vector<cv::Mat> processed_images = preprocess("../images/test.jpeg");
        const ov::Tensor input_tensor = prepare_input(processed_images[0]);
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();
        auto results = infer_request.get_output_tensor();
        auto output_data = results.data<float>();
        std::cout << output_data[12] << std::endl;

        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}