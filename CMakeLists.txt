cmake_minimum_required(VERSION 3.15)
project(BM1684X_det)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_VERBOSE_MAKEFILE ON)

message(STATUS "CMAKE_SYSTEM_NAME      : " ${CMAKE_SYSTEM_NAME})
message(STATUS "CMAKE_SYSTEM_PROCESSOR : " ${CMAKE_SYSTEM_PROCESSOR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fvisibility=hidden -fvisibility-inlines-hidden")
#set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--allow-multiple-definition")

if(NOT BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
    set(CMAKE_CXX_FLAGS " -fPIC ${CMAKE_CXX_FLAGS}")
endif()

# Select Device
if(NOT DEVICE_LOWER)
    set(DEVICE_LOWER cpu) # cpu tpu
endif()
string(TOUPPER ${DEVICE_LOWER} DEVICE_UPPER)
set(USE_${DEVICE_UPPER} ON)
message(STATUS "Device type is : " ${DEVICE_LOWER})

# Select Engine
if(NOT ENGINE_LOWER)
    set(ENGINE_LOWER bm1684x) # bm1684x bm1684
endif()
string(TOUPPER ${ENGINE_LOWER} ENGINE_UPPER)
set(USE_${ENGINE_UPPER} ON)
message(STATUS "Engine type is : " ${ENGINE_LOWER})


if(NOT OUTPUT_DIRECTORY)
    set(OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/${CMAKE_SYSTEM_NAME}/${ENGINE_UPPER})
endif()

if(NOT LIB_MAJOR_VERSION)
    set(LIB_MAJOR_VERSION 1)
endif()

if(NOT LIB_MINOR_VERSION)
    set(LIB_MINOR_VERSION 0)
endif()

include(cmake/cmake_utils.cmake)
string(TIMESTAMP COMPILE_TIME %Y%m%d)
#git号
get_git_hash(GIT_HASH)
#git分支
get_git_branch(GIT_BRANCH)

##############Set output directory##############
file(MAKE_DIRECTORY ${OUTPUT_DIRECTORY})
set(EXECUTABLE_OUTPUT_PATH ${OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${EXECUTABLE_OUTPUT_PATH})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${EXECUTABLE_OUTPUT_PATH})
set(LIBRARY_OUTPUT_PATH ${EXECUTABLE_OUTPUT_PATH})

set(THIRD_PARTY_ROOT ${PROJECT_SOURCE_DIR}/../third_party)
##############SOPHON_SDK##############
include(${THIRD_PARTY_ROOT}/sophon_sdk.cmake)
set(SOPHON_SDK_ROOT ${SDK})
set(SOPHON_SDK_INCLUDE_DIR
        ${SOPHON_SDK_ROOT}/include
        ${SOPHON_SDK_ROOT}/include/opencv4
)
set(SOPHON_SDK_LIB_DIR
        ${SOPHON_SDK_ROOT}/lib
)
set(BM_LIBS bmlib bmrt bmcv yuv avcodec avutil avformat)
set(OPENCV_LIBS opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio)
message("SOPHON_SDK: " ${SDK})
aux_source_directory(${PROJECT_SOURCE_DIR}/../dependencies/src/ dependencies_SRC_FILES)
set(src_all ${src_all} ${dependencies_SRC_FILES})

set(INCLUDE_PATH ${INCLUDE_PATH}
        ../dependencies/include
        ${SOPHON_SDK_INCLUDE_DIR}
        )
set(LIB_PATH ${LIB_PATH}
        ${SOPHON_SDK_LIB_DIR}
        )

include_directories(${INCLUDE_PATH})
link_directories(${LIB_PATH})

set(DET_LIB_NAME ${ENGINE_LOWER}_det)
add_subdirectory(det)

add_subdirectory(examples)

