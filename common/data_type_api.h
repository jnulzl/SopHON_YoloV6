//
// Created by jnulzl on 2020/5/24.
//

#ifndef ALG_DATA_TYPE_C_API_H
#define ALG_DATA_TYPE_C_API_H

#if defined(_MSC_VER) || defined(WIN32)
#if defined(BUILDING_ALG_DLL)
        #define ALG_PUBLIC __declspec(dllexport)
    #elif defined(USING_ALG_DLL)
        #define ALG_PUBLIC __declspec(dllimport)
    #else
        #define ALG_PUBLIC
    #endif
#else
#define ALG_PUBLIC __attribute__((visibility("default")))
#endif

typedef void* Handle;

typedef struct net_config_tag_c
{
    const char** input_names;
    const char** output_names;
    const char* weights_path;
    const char* deploy_path;
    float means[3];
    float scales[3];
    int mean_length;
    int net_inp_channels;
    int net_inp_width;
    int net_inp_height;
    int net_inp_num = 1;
    int net_out_num = 1;
    int num_threads = 2;
}net_config_tag_c;

typedef struct box_info_tag_c
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} box_info_tag_c;

typedef struct cls_info_tag_c
{
    int label;
    float score;
    int frame_id;
} cls_info_tag_c;

typedef enum data_type_tag_c
{
    PIXEL_BGR = 0,
    PIXEL_RGB = 1,
    PIXEL_GRAY = 2,
}data_type_tag_c;

typedef struct img_info_tag_c
{
    void* data;
    int width;
    int height;
    int channels;
    int stride;
    data_type_tag_c data_type = data_type_tag_c::PIXEL_BGR;
} img_info_tag_c;

#endif //ALG_DATA_TYPE_C_API_H
