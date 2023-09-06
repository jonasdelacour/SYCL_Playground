
#Create list of all targets

set(EXECUTABLES
  dualise
  forcefield-opt
)
foreach(EXECUTABLE ${EXECUTABLES})
  add_executable(${EXECUTABLE} ${EXECUTABLE}.cpp)
  if("$ENV{USE_DPCPP}" STREQUAL "false")
    add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${EXECUTABLE}.cpp)
    target_compile_options(${EXECUTABLE} PRIVATE -O4 -cuda -gpu=fastmath,maxregcount:80,lineinfo,cc86, -Mvect)
  endif()
  
endforeach()

#set(HIPSYCL_SYCLCC_EXTRA_COMPILE_O "-O3 --opensycl-targets=cuda-nvcxx")
#set(HIPSYCL_SYCLCC_EXTRA_ARGS "--hipsycl-targets=cuda-nvcxx")