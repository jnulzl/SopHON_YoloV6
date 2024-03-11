file(GLOB _tmp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(src_all ${src_all} ${_tmp})

file(GLOB _tmp ${CMAKE_CURRENT_SOURCE_DIR}/rk35xx/*.cpp)
set(src_all ${src_all} ${_tmp})

if (BUILD_SHARED_LIBS)
    add_library(${DET_LIB_NAME} SHARED ${src_all})
    if("Linux" STREQUAL ${CMAKE_SYSTEM_NAME})
        set_target_properties(${DET_LIB_NAME} PROPERTIES VERSION ${LIB_MAJOR_VERSION}.${LIB_MINOR_VERSION}.${GIT_HASH} SOVERSION ${LIB_MAJOR_VERSION})
    endif()
else()
    set(src_all ${src_all})
    add_library(${DET_LIB_NAME} STATIC ${src_all})
endif ()

target_include_directories(${DET_LIB_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/rk35xx
        )

target_compile_definitions(${DET_LIB_NAME} PRIVATE
        $<$<BOOL:${USE_${ENGINE_UPPER}}>:USE_${ENGINE_UPPER}>
        $<$<BOOL:${USE_${DEVICE_UPPER}}>:USE_${DEVICE_UPPER}>
        )

target_link_options(${DET_LIB_NAME} PRIVATE
        -Wl,--warn-unresolved-symbols
        -Wl,--no-undefined
        -Wl,--no-allow-shlib-undefined
        )

target_link_libraries(${DET_LIB_NAME} PRIVATE
        rknnrt
#        -Wl,--whole-archive ${OpenCV_LIBS} -Wl,--no-whole-archive
        )

if("Android" STREQUAL ${CMAKE_SYSTEM_NAME})
    target_link_libraries(${DET_LIB_NAME} PRIVATE
            opencv_java4
            )
else()
    target_link_libraries(${DET_LIB_NAME} PRIVATE
            opencv_core
            opencv_highgui
            opencv_imgproc
            opencv_imgcodecs
            opencv_video
            opencv_videoio
            )
endif()

if(USE_RGA)
    target_link_libraries(${DET_LIB_NAME} PRIVATE rga)
endif()