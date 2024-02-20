cmake_minimum_required(VERSION 3.15)
project(RK35XX_det)

set(CMAKE_CXX_STANDARD 14)
if(NOT CMAKE_SYSTEM_NAME)
    set(CMAKE_SYSTEM_NAME Android)
endif()

if("Android" STREQUAL ${CMAKE_SYSTEM_NAME})
    set(CMAKE_ANDROID_STL_TYPE gnustl_static)
    set(SYSTEM_ABI ${ANDROID_ABI})
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

message(STATUS "CMAKE_SYSTEM_NAME      : " ${CMAKE_SYSTEM_NAME})
message(STATUS "CMAKE_SYSTEM_PROCESSOR : " ${CMAKE_SYSTEM_PROCESSOR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--allow-multiple-definition")

if(NOT BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
    set(CMAKE_CXX_FLAGS " -fPIC ${CMAKE_CXX_FLAGS}")
endif()

# Select Device
if(NOT DEVICE_LOWER)
    set(DEVICE_LOWER cpu) # cpu cuda
endif()
string(TOUPPER ${DEVICE_LOWER} DEVICE_UPPER)
set(USE_${DEVICE_UPPER} ON)
message(STATUS "Device type is : " ${DEVICE_LOWER})

# Select Engine
if(NOT ENGINE_LOWER)
    set(ENGINE_LOWER rk3588) # rk356x rk3588
endif()
string(TOUPPER ${ENGINE_LOWER} ENGINE_UPPER)
set(USE_${ENGINE_UPPER} ON)
message(STATUS "Engine type is : " ${ENGINE_LOWER})

if(NOT USE_RGA)
    set(USE_RGA OFF)
endif()
if(USE_RGA)
     add_compile_definitions(USE_RGA)
endif()

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

set(THIRD_PARTY_ROOT ${PROJECT_SOURCE_DIR}/../third_party/${CMAKE_SYSTEM_NAME})
##############opencv##############
if("Android" STREQUAL ${CMAKE_SYSTEM_NAME})
    set(OpenCV_DIR_ROOT ${THIRD_PARTY_ROOT}/OpenCV-android-sdk/sdk/native)
    set(OpenCV_DIR ${OpenCV_DIR_ROOT}/jni/abi-${ANDROID_ABI})
    find_package(OpenCV REQUIRED)
    set(OpenCV_INCLUDE_PATH ${OpenCV_INCLUDE_DIRS})
    set(OpenCV_LIB_PATH ${OpenCV_DIR_ROOT}/libs/${ANDROID_ABI})
else()
    set(OpenCV_DIR_ROOT ${THIRD_PARTY_ROOT}/opencv451)
    set(OpenCV_INCLUDE_PATH ${OpenCV_DIR_ROOT}/include
                            ${OpenCV_DIR_ROOT}/include/opencv4
                            ${OpenCV_DIR_ROOT}/include/opencv4/opencv2)
    set(OpenCV_LIB_PATH ${OpenCV_DIR_ROOT}/lib)
endif()

##############RK35XX##############
set(RKNN_DIR ${THIRD_PARTY_ROOT}/${ENGINE_UPPER}/librknn_api/)
set(RKNN_INCLUDE ${RKNN_DIR}/include)
set(RKNN_LIB_PATH ${RKNN_DIR}/lib/${SYSTEM_ABI})

##############RGA##############
if(USE_RGA)
    set(RGA_DIR ${THIRD_PARTY_ROOT}/${ENGINE_UPPER}/rga191/) # rga132 rga191
    set(RGA_INCLUDE ${RGA_DIR}/include)
    set(RGA_LIB_PATH ${RGA_DIR}/lib/${SYSTEM_ABI})
endif()

##############MPP##############
set(MPP_DIR ${THIRD_PARTY_ROOT}/mpp)
set(MPP_INCLUDE ${MPP_DIR}/include ${MPP_DIR}/include/rockchip)
set(MPP_LIB_PATH ${MPP_DIR}/lib/${SYSTEM_ABI})

set(INCLUDE_PATH ${INCLUDE_PATH}
        ${RKNN_INCLUDE}
        ${RGA_INCLUDE}
        ${MPP_INCLUDE}
        ${OpenCV_INCLUDE_PATH}
        )
set(LIB_PATH ${LIB_PATH}
        ${RKNN_LIB_PATH}
        ${RGA_LIB_PATH}
        ${MPP_LIB_PATH}
        ${OpenCV_LIB_PATH}
        )

include_directories(${INCLUDE_PATH})
link_directories(${LIB_PATH})

set(DET_LIB_NAME rk35xx_det)
add_subdirectory(det)

add_subdirectory(examples)

