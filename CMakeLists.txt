cmake_minimum_required(VERSION 3.15)

set(CMAKE_C_COMPILER /usr/bin/gcc-9)
set(CMAKE_CXX_COMPILER /usr/bin/g++-9)

project(cpp-app)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
add_compile_options(-w)  # Disable all warnings for GCC/Clang
find_package(PkgConfig)
find_package(nlohmann_json REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)

pkg_check_modules(GST REQUIRED gstreamer-1.0)
pkg_check_modules(GST_RTP REQUIRED gstreamer-rtp-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)

include_directories(${CURL_INCLUDE_DIR})
include_directories(${GST_INCLUDE_DIRS} ${GST_APP_INCLUDE_DIRS} ${GST_RTP_INCLUDE_DIRS})

# Create executable
add_executable(${PROJECT_NAME} 
        src/Main.cpp
        src/LPRecognizer.cpp
        src/LPRecognizer.h
        src/LPDetector.cpp
        src/LPDetector.h
        src/Config.cpp
        src/Config.h
        src/ILogger.cpp
        src/ILogger.h
        src/IThreadLauncher.h
        src/ITimer.h
        src/Profile.h
        src/RandomStringGenerator.cpp
        src/RandomStringGenerator.h
        src/SharedQueue.h
        src/Template.cpp
        src/Template.h
        src/client/CameraClientLauncher.cpp
        src/client/CameraClientLauncher.h
        src/client/FrameData.cpp
        src/client/FrameData.h
        src/client/GstreamerReader.cpp
        src/client/GstreamerReader.h
        src/package_sending/ImageWriterService.cpp
        src/package_sending/ImageWriterService.h
        src/package_sending/Package.cpp
        src/package_sending/Package.h
        src/package_sending/PackageSender.cpp
        src/package_sending/PackageSender.h
        src/package_sending/Snapshot.cpp
        src/package_sending/Snapshot.h
        src/package_sending/SnapshotSender.cpp
        src/package_sending/SnapshotSender.h
        src/app/BaseCarTracker.cpp
        src/app/BaseCarTracker.h
        src/app/CalibParams.cpp
        src/app/CalibParams.h
        src/app/CalibParamsUpdater.cpp
        src/app/CalibParamsUpdater.h
        src/app/CameraScope.cpp
        src/app/CameraScope.h
        src/app/Car.cpp
        src/app/Car.h
        src/app/Constants.h
        src/app/Counter.h
        src/app/DetectionService.cpp
        src/app/DetectionService.h
        src/app/LicensePlate.cpp
        src/app/LicensePlate.h
        src/app/LPRecognizerService.cpp
        src/app/LPRecognizerService.h
        src/app/MaskCarTracker.cpp
        src/app/MaskCarTracker.h
        src/app/TemplateMatching.cpp
        src/app/TemplateMatching.h
        src/app/Utils.cpp
        src/app/Utils.h
)

set(MODELS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models")
set(MODELS_DESTINATION_DIR "${CMAKE_BINARY_DIR}/models")

message(STATUS "Models source dir: ${MODELS_SOURCE_DIR}")
message(STATUS "Models destination dir: ${MODELS_DESTINATION_DIR}")

add_custom_command(
        TARGET cpp-app POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${MODELS_SOURCE_DIR}
        ${MODELS_DESTINATION_DIR}
        COMMENT "Copying models directory to build directory")

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    openvino::runtime ${GST_LIBRARIES} ${GST_APP_LIBRARIES} ${GST_RTP_LIBRARIES}
    ${OpenCV_LIBS}  nlohmann_json::nlohmann_json  ${CURL_LIBRARIES} /usr/local/lib/libcpr.so
)