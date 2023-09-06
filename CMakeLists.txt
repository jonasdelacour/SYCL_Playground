cmake_minimum_required(VERSION 3.5)
project(device_info)

message(STATUS "USE_DPCPP is set to $ENV{USE_DPCPP}")


if("$ENV{USE_DPCPP}" STREQUAL "true")
    message(STATUS "Using DPCPP")
    set(CMAKE_CXX_COMPILER icpx)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xcuda-ptxas --maxrregcount=80 -Xcuda-ptxas --verbose -funsafe-math-optimizations -fno-associative-math -fp-model=fast")
else()
    message(STATUS "Using hipSYCL")
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/../cmake)
    set(HIPSYCL_TARGETS "cuda-nvcxx")
    find_package(hipSYCL CONFIG REQUIRED)
    include_directories(${HIPSYCL_INCLUDE_DIR})
endif()

add_subdirectory(programs)
add_subdirectory(sycl-tests)
# flags for cxx compiler
