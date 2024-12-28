#include "LPDetector.h"

using namespace std;

Detector::Detector() {
    try {
        ov::Core core;
        
        std::string model_path = "./models/detection.xml";  // Adjust path as needed
        auto model = core.read_model(model_path);
        model_ = core.compile_model(model, "CPU");
        infer_request_ = model_.create_infer_request(); 
        std::cout << "Detector Model loaded successfully" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error initializing detector: " << e.what() << std::endl;
        throw;
    }
}

Detector::~Detector() {
}

float Detector::calculateIOU(const std::vector<float>& box1, const std::vector<float>& box2, bool x1y1x2y2) {
    float b1_x1, b1_y1, b1_x2, b1_y2;
    float b2_x1, b2_y1, b2_x2, b2_y2;

    if (!x1y1x2y2) {
        b1_x1 = box1[0] - box1[2] / 2;
        b1_y1 = box1[1] - box1[3] / 2;
        b1_x2 = box1[0] + box1[2] / 2;
        b1_y2 = box1[1] + box1[3] / 2;
        
        b2_x1 = box2[0] - box2[2] / 2;
        b2_y1 = box2[1] - box2[3] / 2;
        b2_x2 = box2[0] + box2[2] / 2;
        b2_y2 = box2[1] + box2[3] / 2;
    } else {
        b1_x1 = box1[0]; b1_y1 = box1[1];
        b1_x2 = box1[2]; b1_y2 = box1[3];
        b2_x1 = box2[0]; b2_y1 = box2[1];
        b2_x2 = box2[2]; b2_y2 = box2[3];
    }

    float inter_rect_x1 = std::max(b1_x1, b2_x1);
    float inter_rect_y1 = std::max(b1_y1, b2_y1);
    float inter_rect_x2 = std::min(b1_x2, b2_x2);
    float inter_rect_y2 = std::min(b1_y2, b2_y2);

    float inter_area = std::max(inter_rect_x2 - inter_rect_x1 + 1, 0.0f) * 
                      std::max(inter_rect_y2 - inter_rect_y1 + 1, 0.0f);

    float b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1);
    float b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1);

    return inter_area / (b1_area + b2_area - inter_area);
}

std::vector<std::vector<float>> Detector::nms(const std::vector<std::vector<float>>& predictions, 
                                  float conf_thres, 
                                  float nms_thres) {
    std::vector<std::vector<float>> output;
    std::vector<std::vector<float>> filtered_predictions;

    for (const auto& pred : predictions) {
        if (pred.back() >= conf_thres) {
            filtered_predictions.push_back(pred);
        }
    }

    if (filtered_predictions.empty()) {
        return output;
    }

    while (!filtered_predictions.empty()) {
        auto max_conf_it = std::max_element(
            filtered_predictions.begin(), 
            filtered_predictions.end(),
            [](const auto& a, const auto& b) { return a.back() < b.back(); }
        );

        output.push_back(*max_conf_it);
        std::vector<std::vector<float>> temp_preds;

        for (const auto& pred : filtered_predictions) {
            if (pred != *max_conf_it) {
                std::vector<float> box1(max_conf_it->begin(), max_conf_it->begin() + 4);
                std::vector<float> box2(pred.begin(), pred.begin() + 4);
                if (calculateIOU(box1, box2) < nms_thres) {
                    temp_preds.push_back(pred);
                }
            }
        }
        filtered_predictions = std::move(temp_preds);
    }

    return output;
}

std::vector<float> Detector::preprocess(const cv::Mat& image) {
    cv::Mat resized;
    resize(image, resized, cv::Size(IMG_WIDTH, IMG_HEIGHT));
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32F, 1.0 / 255.0);
    float_img = 2 * (float_img - 0.5);

    std::vector<float> input_data(IMG_CHANNELS * IMG_WIDTH * IMG_HEIGHT);
    for (int c = 0; c < IMG_CHANNELS; c++) {
        for (int h = 0; h < IMG_HEIGHT; h++) {
            for (int w = 0; w < IMG_WIDTH; w++) {
                input_data[c * IMG_WIDTH * IMG_HEIGHT + h * IMG_WIDTH + w] = 
                    float_img.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    return input_data;
}

cv::Mat Detector::detect(const cv::Mat& image) {
    std::vector<float> input_data = preprocess(image);
    ov::Tensor input_tensor(model_.input().get_element_type(),
                                  model_.input().get_shape(),
                                  input_data.data());
    infer_request_.set_input_tensor(input_tensor);
    infer_request_.infer();
    auto results = infer_request_.get_output_tensor();

    float* output_data = results.data<float>();
    std::vector<std::vector<float>> predictions;
    size_t pred_size = 13; // 12 coordinates + 1 confidence
    size_t num_predictions = results.get_size() / pred_size;
    for (size_t i = 0; i < num_predictions; i++) {
        std::vector<float> pred(output_data + i * pred_size, 
                            output_data + (i + 1) * pred_size);
        predictions.push_back(pred);
    }
    auto filtered_predictions = nms(predictions, 0.2f, 0.2f);
    
    float rx = static_cast<float>(image.cols) / IMG_WIDTH;
    float ry = static_cast<float>(image.rows) / IMG_HEIGHT;

    if (filtered_predictions.empty()) {
        return cv::Mat();
    }else{
        for (auto& pred : filtered_predictions) {
            if (pred.back() < threshold) continue;

            pred[4] += pred[0];  // lt_x
            pred[6] += pred[0];  // lb_x
            pred[8] += pred[0];  // rt_x
            pred[10] += pred[0]; // rb_x
            
            pred[5] += pred[1];  // lt_y
            pred[7] += pred[1];  // lb_y
            pred[9] += pred[1];  // rt_y
            pred[11] += pred[1]; // rb_y

            for (size_t i = 0; i < pred.size(); i += 2) {
                pred[i] *= rx;
            }
            for (size_t i = 1; i < pred.size(); i += 2) {
                pred[i] *= ry;
            }

            cv::Point2f src_points[4] = {
                cv::Point2f(pred[4], pred[5]),   // Left Top
                cv::Point2f(pred[8], pred[9]),   // Right Top
                cv::Point2f(pred[6], pred[7]),   // Left Bottom
                cv::Point2f(pred[10], pred[11])  // Right Bottom
            };

            float width = std::max(
                cv::norm(src_points[0] - src_points[1]),  // top edge
                cv::norm(src_points[2] - src_points[3])   // bottom edge
            );
            float height = std::max(
                cv::norm(src_points[0] - src_points[2]),  // left edge
                cv::norm(src_points[1] - src_points[3])   // right edge
            );

            cv::Point2f dst_points[4] = {
                cv::Point2f(0, 0),           // Left Top
                cv::Point2f(width, 0),       // Right Top
                cv::Point2f(0, height),      // Left Bottom
                cv::Point2f(width, height)   // Right Bottom
            };

            cv::Mat perspective_matrix = cv::getPerspectiveTransform(src_points, dst_points);
            cv::Mat result_image_;
            cv::warpPerspective(image, result_image_, perspective_matrix, cv::Size(width, height));
            return result_image_;
        }

        return cv::Mat();
    }
}