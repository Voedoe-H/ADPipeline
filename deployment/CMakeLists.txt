cmake_minimum_required(VERSION 3.14)
project(InferenceADPipeline)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_libraries(${OpenCV_LIBS})
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "OpenCV not found")
endif()