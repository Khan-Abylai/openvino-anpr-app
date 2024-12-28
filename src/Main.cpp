#include <thread>
#include <chrono>
#include <csignal>
#include <condition_variable>
#include <mutex>

#include "client/CameraClientLauncher.h"
#include "Config.h"
#include "SharedQueue.h"
#include "app/LPRecognizerService.h"
#include "app/DetectionService.h"
#include "package_sending/PackageSender.h"
#include "package_sending/Package.h"
#include "package_sending/Snapshot.h"
#include "package_sending/SnapshotSender.h"
#include "Template.h"

using namespace std;

atomic<bool> shutdownFlag = false;
condition_variable shutdownEvent;
mutex shutdownMutex;

void signalHandler(int signum) {
    cout << "signal is to shutdown" << endl;
    shutdownFlag = true;
    shutdownEvent.notify_all();
}
namespace fs = std::filesystem;


int main(int argc, char *argv[]) {

    char env[] = "CUDA_MODULE_LOADING=LAZY";
    putenv(env);

    string configFileName, templateFileName;
    if (argc <= 1)
        configFileName = "config.json";
    else
        configFileName = argv[1];

    templateFileName = "template.json";

    if (!Config::parseJson(configFileName))
        return -1;

    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    signal(SIGKILL, signalHandler);
    signal(SIGHUP, signalHandler);
    signal(SIGABRT, signalHandler);
    auto packageQueue = make_shared < SharedQueue < shared_ptr < Package>>>();
    auto snapshotQueue = make_shared < SharedQueue < shared_ptr < Snapshot>>>();
    vector <shared_ptr<IThreadLauncher>> services;
    auto camerasVector = Config::getCameras();
    auto allCameras = Config::getAllCameras();
    vector < shared_ptr < SharedQueue < unique_ptr < FrameData >> >> frameQueues;

    for (const auto &cameras: camerasVector) {
        std::unordered_map<std::string, Constants::CountryCode> templates;
        if (Template::parseJson(templateFileName)) {
            templates = Template::getTemplateMap();
        }
        auto lpRecognizerService = make_shared<LPRecognizerService>(packageQueue, cameras,
                                                                    Config::getRecognizerThreshold(),
                                                                    Config::getCalibrationSizes(),
                                                                    Config::getStatInterval(), templates);
        services.emplace_back(lpRecognizerService);
        for (const auto &camera: cameras) {
            auto frameQueue = make_shared < SharedQueue < unique_ptr < FrameData>>>();
            auto detectionService = make_shared<DetectionService>(frameQueue, snapshotQueue,
                                                                  lpRecognizerService,
                                                                  camera, Config::getTimeSentSnapshots(),
                                                                  Config::getCalibrationSizes());
            services.emplace_back(detectionService);
            frameQueues.push_back(std::move(frameQueue));
        }
    }


    shared_ptr <IThreadLauncher> clientStarter;
    clientStarter = make_shared<CameraClientLauncher>(Config::getAllCameras(), frameQueues,
                                                      Config::useGPUDecode());
    services.emplace_back(clientStarter);

    auto packageSender = make_shared<PackageSender>(packageQueue, Config::getAllCameras(), Config::getBaseFolder(),
                                                    Config::useImageWriter());
    services.emplace_back(packageSender);


    auto snapshotSender = make_shared<SnapshotSender>(snapshotQueue, Config::getAllCameras(), Config::useImageWriter(),
                                                      Config::getBaseFolder());
    services.emplace_back(snapshotSender);

    vector <thread> threads;
    for (const auto &service: services) {
        threads.emplace_back(&IThreadLauncher::run, service);
    }

    unique_lock <mutex> shutdownLock(shutdownMutex);
    while (!shutdownFlag) {
        auto timeout = chrono::hours(24);
        if (shutdownEvent.wait_for(shutdownLock, timeout, [] { return shutdownFlag.load(); })) {
            cout << "shutting all services" << endl;
        }
    }

    for (int i = 0; i < services.size(); i++) {
        services[i]->shutdown();
        if (threads[i].joinable())
            threads[i].join();
    }
}


// int main() {
//     cv::Mat result_image;
//     std::shared_ptr<Recognizer> recognizer = std::make_shared<Recognizer>();
//     std::shared_ptr<Detector> detector = std::make_shared<Detector>();
//     const std::string image_path = "../images/test.jpeg";
//
//     cv::Mat myImage = cv::imread(image_path);
//     auto start_postproc = std::chrono::high_resolution_clock::now();
//     auto detector_result = detector->detect(myImage);
//     auto end_postproc = std::chrono::high_resolution_clock::now();
//     auto start_recognizer = std::chrono::high_resolution_clock::now();
//     auto recognizer_result = recognizer->predict({detector_result});
//     auto end_recognizer = std::chrono::high_resolution_clock::now();
//     cout << "Recognition result: " << recognizer_result[0].first << endl;
//     cout << "Postprocessing time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_postproc - start_postproc).count() << " ms" << endl;
//     cout << "Recognizer time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_recognizer - start_recognizer).count() << " ms" << endl;
//     return 0;
// }
