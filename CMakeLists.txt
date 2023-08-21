cmake_minimum_required(VERSION 3.15)
project(AlgRK35XX)

set(CMAKE_ANDROID_STL_TYPE gnustl_static)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_SYSTEM_NAME Android)
#set(CMAKE_SYSTEM_VERSION 28)
set(CMAKE_VERBOSE_MAKEFILE ON)

message(STATUS "CMAKE_SYSTEM_NAME      : " ${CMAKE_SYSTEM_NAME})
message(STATUS "CMAKE_SYSTEM_PROCESSOR : " ${CMAKE_SYSTEM_PROCESSOR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--allow-multiple-definition")

if(NOT SUB_LIB_BUILD_SHARED_LIBS)
    set(SUB_LIB_BUILD_SHARED_LIBS OFF)
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
    set(ENGINE_LOWER rk356x)
endif()
string(TOUPPER ${ENGINE_LOWER} ENGINE_UPPER)
set(USE_${ENGINE_UPPER} ON)
message(STATUS "Engine type is : " ${ENGINE_LOWER})

if(NOT OUTPUT_DIRECTORY)
    set(OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/${CMAKE_SYSTEM_NAME})
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
set(OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/${CMAKE_SYSTEM_NAME})
file(MAKE_DIRECTORY ${OUTPUT_DIRECTORY})
set(EXECUTABLE_OUTPUT_PATH ${OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${EXECUTABLE_OUTPUT_PATH})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${EXECUTABLE_OUTPUT_PATH})
set(LIBRARY_OUTPUT_PATH ${EXECUTABLE_OUTPUT_PATH})

##############opencv##############
set(OpenCV_DIR ${PROJECT_SOURCE_DIR}/third_party/${CMAKE_SYSTEM_NAME}/OpenCV-android-sdk/sdk/native/jni/abi-${ANDROID_ABI})
find_package(OpenCV REQUIRED)

##############RK35XX##############
set(RKNN_DIR ${PROJECT_SOURCE_DIR}/third_party/${CMAKE_SYSTEM_NAME}/${ENGINE_UPPER}/librknn_api/)
set(RKNN_INCLUDE ${RKNN_DIR}/include)
set(RKNN_LIB_PATH ${RKNN_DIR}/lib/${ANDROID_ABI})

##############RGA##############
set(RGA_DIR ${PROJECT_SOURCE_DIR}/third_party/${CMAKE_SYSTEM_NAME}/${ENGINE_UPPER}/rga/)
set(RGA_INCLUDE ${RGA_DIR}/include)
set(RGA_LIB_PATH ${RGA_DIR}/lib/${ANDROID_ABI})

set(INCLUDE_PATH ${INCLUDE_PATH}
        ${RKNN_INCLUDE}
        ${RGA_INCLUDE}
        ${OpenCV_INCLUDE_DIRS}
        )
set(LIB_PATH ${LIB_PATH}
        ${RKNN_LIB_PATH}
        ${RGA_LIB_PATH}
        )

include_directories(${INCLUDE_PATH})
link_directories(${LIB_PATH})

set(CLS_LIB_NAME ${ENGINE_LOWER}_cls)

add_subdirectory(cls)
add_subdirectory(examples)

