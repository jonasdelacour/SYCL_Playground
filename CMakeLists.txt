cmake_minimum_required(VERSION 3.5)

project(device_info )

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/../cmake)

#set(HIPSYCL_TARGETS "cuda" CACHE STRING "List of hipSYCL targets to build for")
# Find OpenSYCL
set(HIPSYCL_TARGETS "cuda-nvcxx")
find_package(hipSYCL CONFIG REQUIRED)

# Add the OpenSYCL CMake module
#list(APPEND CMAKE_MODULE_PATH ${HIPSYCL_TOOLCHAIN_DIR}/cmake)

# Set the OpenSYCL device selector
#set(HIPSYCL_DEVICE_SELECTOR "omp")

# Add the executable
include_directories(${HIPSYCL_INCLUDE_DIR})
add_subdirectory(programs)
add_subdirectory(sycl-tests)
# flags for cxx compiler



# Link with OpenSYCL
#target_link_libraries(device_info  hipSYCL::hipSYCL)