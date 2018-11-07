cmake_minimum_required(VERSION 3.12)
project(football_track)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

add_executable(football_track
        src/main.cpp src/image.cpp src/image.h)

target_link_libraries(football_track ${OpenCV_LIBS})