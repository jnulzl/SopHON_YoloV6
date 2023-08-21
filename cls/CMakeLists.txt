file(GLOB _tmp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(src_all ${src_all} ${_tmp})

file(GLOB _tmp ${CMAKE_CURRENT_SOURCE_DIR}/${ENGINE_LOWER}/*.cpp)
set(src_all ${src_all} ${_tmp})

file(GLOB _tmp ${CMAKE_CURRENT_SOURCE_DIR}/jni/*.cpp)
set(src_all ${src_all} ${_tmp})

if (SUB_LIB_BUILD_SHARED_LIBS)
    add_library(${CLS_LIB_NAME} SHARED ${src_all})
else()
    set(src_all ${src_all})
    add_library(${CLS_LIB_NAME} STATIC ${src_all})
endif ()

target_include_directories(${CLS_LIB_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/jni
        $<$<BOOL:${USE_${ENGINE_UPPER}}>:${CMAKE_CURRENT_SOURCE_DIR}/${ENGINE_LOWER}>
        )

target_compile_definitions(${CLS_LIB_NAME} PRIVATE
        $<$<BOOL:${USE_${ENGINE_UPPER}}>:USE_${ENGINE_UPPER}>
        $<$<BOOL:${USE_${DEVICE_UPPER}}>:USE_${DEVICE_UPPER}>
        )

target_link_options(${CLS_LIB_NAME} PRIVATE
#        -Wl,--warn-unresolved-symbols
        -Wl,--no-undefined
        -Wl,--no-allow-shlib-undefined
        )

target_link_libraries(${CLS_LIB_NAME} PRIVATE
        log
        jnigraphics
        rknnrt
        rga
        -Wl,--whole-archive ${OpenCV_LIBS} -Wl,--no-whole-archive
        )