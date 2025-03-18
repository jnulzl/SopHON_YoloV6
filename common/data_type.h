﻿//
// Created by jnulzl on 2020/5/24.
//

#ifndef ALG_DATA_TYPE_H
#define ALG_DATA_TYPE_H

#include <iostream>
#include <string>
#include <vector>
#include <cmath>


#define ANY_POINTER_CAST(impl, T) reinterpret_cast<T*>(impl)
typedef void* AW_ANY_POINTER;

struct BaseConfig
{
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::string weights_path;
    std::string deploy_path;
    float means[3];
    float scales[3];
    int mean_length;
    int net_inp_channels;
    int net_inp_width;
    int net_inp_height;
    int num_threads = 2;
#if defined(USE_CUDA) || defined(USE_RK3588) || defined(USE_BM1684X) || defined(USE_BM1684)
    int batch_size = 1;
    int device_id = 0;
#ifdef USE_TENSORRT
    int dlaCore = -1;
    bool fp16 = true;
    bool int8 = false;
#endif
#endif
    int model_include_preprocess = 0;
};

struct YoloConfig : public BaseConfig
{
#if defined(USE_BM1684X) || defined(USE_BM1684)
    void* handle;
#endif
    int num_cls = 1;
    float conf_thres;
    float nms_thresh;
    std::vector<int> strides;
    std::vector<std::vector<float>> anchor_grids;
};


typedef struct
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    float area;
    int label;
} BoxInfo;

typedef struct
{
    BoxInfo *boxes;
    int size;
    int capacity;
    int frame_id;
} BoxInfos;

typedef struct
{
    int label;
    float score;
    int frame_id;
} ClsInfo;

typedef struct
{
    ClsInfo *boxes;
    int size;
    int capacity;
    int frame_id;
} ClsInfos;

typedef struct
{
    uint8_t *cls;
    float *probs;
    int height;
    int width;
    char reserve[8];
} SegmentResult;

typedef enum : int
{
    IMG_BGR = 0,
    IMG_RGB = 1,
    IMG_GRAY = 2
} InputDataType;


template<typename DATATYPE>
struct _Rect
{
    DATATYPE xmin;
    DATATYPE ymin;
    DATATYPE xmax;
    DATATYPE ymax;

    DATATYPE width;
    DATATYPE height;
};
typedef _Rect<float> RectFloat;
typedef _Rect<int> RectInt;

template<typename DATATYPE>
struct _Point
{
    DATATYPE x;
    DATATYPE y;
    float score;
};
typedef _Point<float> PointFloat;
typedef _Point<int> PointInt;

typedef struct
{
    PointFloat *points;
    int size;
    int capacity;
    int frame_id;
} PointFloats;


typedef struct
{
    RectFloat rect;
    int id;
    char reserve[8];
} RectWithID;

template<typename DATATYPE>
struct _ImageInfo
{
    DATATYPE *data = nullptr;
    int img_height = 0;
    int img_width = 0;
    int is_device_data = 0;
    int stride = 0;
    int frame_id = 0;
    InputDataType img_data_type = InputDataType::IMG_BGR;
    char reserve[8];
};
typedef _ImageInfo<uint8_t> ImageInfoUint8;
typedef _ImageInfo<float> ImageInfoFloat32;

typedef struct
{
    int frame_id;
    int attribute;
    PointFloat pointFloat[4];
    char reserve[8];
} frame_attribute_t;
#endif //ALG_DATA_TYPE_H
