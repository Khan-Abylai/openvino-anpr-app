#include "LPRecognizer.h"

using namespace std;

Recognizer::Recognizer() {
    try {
        ov::Core core;
        
        std::string model_path = "./models/recognition.xml";  // Adjust path as needed
        auto model = core.read_model(model_path);
        model_ = core.compile_model(model, "CPU");
        infer_request_ = model_.create_infer_request(); 
        std::cout << "Recognizer Model loaded successfully" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error initializing recognizer: " << e.what() << std::endl;
        throw;
    }
}

Recognizer::~Recognizer() {

}

vector<float> Recognizer::prepareImage(const vector<cv::Mat> &frames) const {
    std::vector<float> input_data;
    input_data.reserve(frames.size() * INPUT_SIZE);

    for (const auto& frame : frames) {
        // Resize and convert to float in one step
        cv::Mat resized_float;
        cv::resize(frame, resized_float, cv::Size(IMG_WIDTH, IMG_HEIGHT));
        resized_float.convertTo(resized_float, CV_32F, 1.0/255.0);

        // Split and flatten channels
        std::vector<cv::Mat> channels;
        cv::split(resized_float, channels);
        
        // Copy each channel's data
        for (const auto& channel : channels) {
            input_data.insert(input_data.end(), 
                            (float*)channel.data, 
                            (float*)channel.data + IMG_WIDTH * IMG_HEIGHT);
        }
    }

    return input_data;
}

std::vector<std::pair<std::string, float>> Recognizer::predict(const std::vector<cv::Mat> &frames){
        auto start_total = std::chrono::high_resolution_clock::now();

    auto start_prep = std::chrono::high_resolution_clock::now();
    auto preparedImages = prepareImage(frames);
    auto end_prep = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<std::string, float>> labels;

    try {
        auto start_infer = std::chrono::high_resolution_clock::now();

        ov::Tensor input_tensor(
            ov::element::f32, 
            model_.input().get_shape(), 
            preparedImages.data()
        );
        
        infer_request_.set_input_tensor(input_tensor);
        infer_request_.infer();
        auto end_infer = std::chrono::high_resolution_clock::now();
        auto start_postproc = std::chrono::high_resolution_clock::now();
        auto output = infer_request_.get_output_tensor();
        const float* output_buffer = output.data<const float>();
        size_t output_size = output.get_size();
        cout << output_size << endl;
        std::vector<float> output_vector(output_buffer, output_buffer + output_size);

        vector<float> predictions;
        predictions.resize(OUTPUT_SIZE);
        int batchSize = 1;
        for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {

            vector<float> preds(OUTPUT_SIZE);

            for (int i = 0; i < SEQUENCE_SIZE; i++) {
                vector<float> temp_out(ALPHABET_SIZE);
                for (int j = 0; j < ALPHABET_SIZE; j++) {
                    temp_out[j] = output_vector[ALPHABET_SIZE * i + j];
                }

                vector<float> result = softmax(temp_out);
                temp_out.clear();
                int count = 0;
                for (int j = i * ALPHABET_SIZE; j < ALPHABET_SIZE * (i + 1); j++) {
                    preds[j] = result[count];
                    count++;
                }
                result.clear();
            }
            output_vector.clear();
            int c = 0;
            for (int i = batchIndex * SEQUENCE_SIZE * ALPHABET_SIZE;
                 i < (batchIndex + 1) * ALPHABET_SIZE * SEQUENCE_SIZE; i++) {
                predictions[i] = preds[c];
                c++;
            }
            preds.clear();
        }


        for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {
            float prob = 1.0;
            string currentLabel;
            currentLabel.reserve(MAX_PLATE_SIZE);

            int currentChar = BLANK_INDEX;
            float currentProb = 1.0;

            for (int i = 0; i < SEQUENCE_SIZE; i++) {
                float maxProb = 0.0;
                int maxIndex = 0;

                for (int j = 0; j < ALPHABET_SIZE; j++) {
                    if (maxProb < predictions[batchIndex * ALPHABET_SIZE * SEQUENCE_SIZE + i * ALPHABET_SIZE + j]) {
                        maxIndex = j;
                        maxProb = predictions[batchIndex * ALPHABET_SIZE * SEQUENCE_SIZE + i * ALPHABET_SIZE + j];
                    }
                }

                if (maxIndex == currentChar) {
                    currentProb = max(maxProb, currentProb);
                } else {
                    if (currentChar != BLANK_INDEX) {
                        currentLabel += ALPHABET[currentChar];
                        prob *= currentProb;
                    }
                    currentProb = maxProb;
                    currentChar = maxIndex;
                }
            }

            if (currentChar != BLANK_INDEX) {
                currentLabel += ALPHABET[currentChar];
                prob *= currentProb;
            }

            if (currentLabel.empty()) {
                currentLabel += ALPHABET[BLANK_INDEX];
                prob = 0.0;
            }

            labels.emplace_back(make_pair(currentLabel, prob));
        }

        auto end_postproc = std::chrono::high_resolution_clock::now();
        auto end_total = std::chrono::high_resolution_clock::now();


        double prep_time = std::chrono::duration<double>(end_prep - start_prep).count();
        double infer_time = std::chrono::duration<double>(end_infer - start_infer).count();
        double postproc_time = std::chrono::duration<double>(end_postproc - start_postproc).count();
        double total_time = std::chrono::duration<double>(end_total - start_total).count();

        // Print timing information
        std::cout << "\nRecognizer Timing Information:" << std::endl;
        std::cout << "Preprocessing time: " << prep_time * 1000 << " ms" << std::endl;
        std::cout << "Inference time: " << infer_time * 1000 << " ms" << std::endl;
        std::cout << "Postprocessing time: " << postproc_time * 1000 << " ms" << std::endl;
        std::cout << "Total time: " << total_time * 1000 << " ms\n" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
    }

    return labels;
}