cmake_minimum_required(VERSION 3.10)
project(openvino_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)

# Create executable
add_executable(${PROJECT_NAME} 
    main.cpp
)
# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    openvino::runtime
    ${OpenCV_LIBS}
)