set(EXECUTABLES
  device_info
  kernel_properties
  isomer-batch-test
  buffer-test
)
foreach(EXECUTABLE ${EXECUTABLES})
  add_executable(${EXECUTABLE} ${EXECUTABLE}.cpp)
  add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${EXECUTABLE}.cpp)
  target_compile_options(${EXECUTABLE} PRIVATE -O3)
endforeach()