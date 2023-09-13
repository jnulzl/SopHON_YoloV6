set(demo_src demo_det_images.cpp)
get_filename_component(exe_name ${demo_src} NAME_WLE)

add_executable(${exe_name} ${demo_src})
target_include_directories(${exe_name} PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/common
        )

target_compile_options(${exe_name} PRIVATE
        -DPROJECT_ROOT="${PROJECT_SOURCE_DIR}"
        )
target_compile_definitions(${exe_name} PRIVATE
        $<$<BOOL:${USE_${ENGINE_UPPER}}>:USE_${ENGINE_UPPER}>
        $<$<BOOL:${USE_${DEVICE_UPPER}}>:USE_${DEVICE_UPPER}>
        )

target_link_options(${exe_name} PRIVATE
        -Wl,--no-undefined
        -Wl,--no-allow-shlib-undefined
        -Wl,--warn-unresolved-symbols
        )

if(BUILD_SHARED_LIBS)
    target_link_libraries(${exe_name} PRIVATE
            ${DET_LIB_NAME}
            )
else()
    target_link_libraries(${exe_name} PRIVATE
            -Wl,--whole-archive ${DET_LIB_NAME} -Wl,--no-whole-archive
            )
    target_link_options(${exe_name} PRIVATE
            -Wl,--allow-multiple-definition
            )
endif()

target_link_libraries(${exe_name} PRIVATE
        rknnrt
#        ${OpenCV_LIBS}
#        mpp
        opencv_java4
)

if(USE_RGA)
    target_link_libraries(${exe_name} PRIVATE rga)
endif()