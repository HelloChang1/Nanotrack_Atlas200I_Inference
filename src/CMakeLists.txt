# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(ACL_NAnotrack)

set(CURRENT_DIR ${CMAKE_CURRENT_LIST_DIR})


# Compile options
add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_LIST_DIR}/../out)
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

set(INC_PATH $ENV{DDK_PATH})

set(INC_PATH $ENV{DDK_PATH})
if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else()
    message(STATUS "set INC_PATH: ${INC_PATH}")
endif ()


set(LIB_PATH $ENV{NPU_HOST_LIB})

set(LIB_PATH $ENV{NPU_HOST_LIB})
if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else()
    message(STATUS "set LIB_PATH: ${LIB_PATH}")
endif ()

# 自动查找 OpenCV
# find_package   是 CMake 提供的一个功能，可以自动查找和配置 OpenCV。
find_package(OpenCV REQUIRED)

# 包含 OpenCV 的头文件路径
include_directories(${OpenCV_INCLUDE_DIRS})


include_directories(
   ${INC_PATH}/runtime/include/
   ${CURRENT_DIR}/../inc
)

link_directories(
    ${LIB_PATH}
)


add_executable(main
        ${CURRENT_DIR}/ACNNModel_B.cpp
        ${CURRENT_DIR}/ACNNModel_N.cpp
        ${CURRENT_DIR}/nanotrack.cpp
        ${CURRENT_DIR}/main.cpp)


# target_link_libraries(main ascendcl stdc++ opencv_core opencv_imgproc opencv_imgcodecs dl rt)
target_link_libraries(main ${OpenCV_LIBS} ascendcl stdc++ dl rt)


install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})