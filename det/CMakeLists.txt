file(GLOB _tmp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(src_all ${src_all} ${_tmp})

set(ENGINE_LOWER rk35xx)

file(GLOB _tmp ${CMAKE_CURRENT_SOURCE_DIR}/${ENGINE_LOWER}/*.cpp)
set(src_all ${src_all} ${_tmp})

if(USE_JNI)
    file(GLOB _tmp ${CMAKE_CURRENT_SOURCE_DIR}/jni/*.cpp)
    set(src_all ${src_all} ${_tmp})
endif()

if (BUILD_SHARED_LIBS)
    add_library(${DET_LIB_NAME} SHARED ${src_all})
    set_target_properties(${DET_LIB_NAME} PROPERTIES SUFFIX .so.${GIT_HASH})
else()
    set(src_all ${src_all})
    add_library(${DET_LIB_NAME} STATIC ${src_all})
endif ()

target_include_directories(${DET_LIB_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        $<$<BOOL:${USE_JNI}>:${CMAKE_CURRENT_SOURCE_DIR}/jni>
        $<$<BOOL:${USE_${ENGINE_UPPER}}>:${CMAKE_CURRENT_SOURCE_DIR}/${ENGINE_LOWER}>
        )

target_compile_definitions(${DET_LIB_NAME} PRIVATE
        $<$<BOOL:${USE_${ENGINE_UPPER}}>:USE_${ENGINE_UPPER}>
        $<$<BOOL:${USE_${DEVICE_UPPER}}>:USE_${DEVICE_UPPER}>
        )

target_link_options(${DET_LIB_NAME} PRIVATE
#        -Wl,--warn-unresolved-symbols
        -Wl,--no-undefined
        -Wl,--no-allow-shlib-undefined
        )

target_link_libraries(${DET_LIB_NAME} PRIVATE
        rknnrt
#        -Wl,--whole-archive ${OpenCV_LIBS} -Wl,--no-whole-archive
        opencv_java4
        )

if(USE_JNI)
    target_link_libraries(${DET_LIB_NAME} PRIVATE
            log
            jnigraphics)
endif()

#target_link_libraries(${DET_LIB_NAME} PRIVATE rga)

if(USE_RGA)
    target_link_libraries(${DET_LIB_NAME} PRIVATE rga)
endif()