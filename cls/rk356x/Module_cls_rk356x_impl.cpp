//
// Created by lizhaoliang-os on 2020/6/9.
//

#include "opencv2/opencv.hpp"
#include "rk356x/Module_cls_rk356x_impl.h"
#include "debug.h"

#ifdef ALG_DEBUG
static void printRKNNTensor(rknn_tensor_attr *attr)
{
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0],
           attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}
#endif

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr)
    {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char *) malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if (model_len != fread(model, 1, model_len, fp))
    {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }
    *model_size = model_len;
    if (fp)
    {
        fclose(fp);
    }
    return model;
}

CModule_cls_rk356x_impl::CModule_cls_rk356x_impl()
{

}

CModule_cls_rk356x_impl::~CModule_cls_rk356x_impl()
{

}

void CModule_cls_rk356x_impl::engine_deinit()
{
    // Destroy rknn memory
    for (uint32_t i = 0; i < input_mems_.size(); ++i)
    {
        rknn_destroy_mem(ctx_, input_mems_[0]);
    }

    for (uint32_t i = 0; i < output_mems_.size(); ++i)
    {
        rknn_destroy_mem(ctx_, output_mems_[i]);
    }

    // Release
    if (ctx_ >= 0)
    {
        rknn_destroy(ctx_);
    }

    if(model_buffer_)
    {
        free(model_buffer_);
    }
#ifdef ALG_DEBUG
    std::printf("%d,%s\n", __LINE__, __FUNCTION__);
#endif
}

void CModule_cls_rk356x_impl::engine_init()
{
    int            model_len = 0;
    model_buffer_    = load_model(config_.weights_path.c_str(), &model_len);
    int ret = rknn_init(&ctx_, model_buffer_, model_len, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        exit(-1);
    }

    // Get sdk and driver version
    rknn_sdk_version sdk_ver;
    ret = rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        exit(-1);
    }
    printf("rknn_api/rknnrt version: %s, driver version: %s\n", sdk_ver.api_version, sdk_ver.drv_version);

    // Get Model Input Output Info
    rknn_input_output_num io_num;
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        exit(-1);
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors: %d\n", io_num.n_input);
    input_attrs_.resize(io_num.n_input);
    input_mems_.resize(io_num.n_input);
    memset(input_attrs_.data(), 0, io_num.n_input * sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs_[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            exit(-1);
        }
        input_mems_[i] = rknn_create_mem(ctx_, input_attrs_[0].size_with_stride);
#ifdef ALG_DEBUG
        printRKNNTensor(&(input_attrs_[i]));
#endif
    }
    printf("output tensors:\n");
    output_attrs_.resize(io_num.n_output);
    output_mems_.resize(io_num.n_output);
    memset(output_attrs_.data(), 0, io_num.n_output * sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            exit(-1);
        }
        output_mems_[i] = rknn_create_mem(ctx_, output_attrs_[i].n_elems * sizeof(float));

        // default output type is depend on model, this require float32 to compute top5
        output_attrs_[i].type = RKNN_TENSOR_FLOAT32;
        // set output memory and attribute
        ret = rknn_set_io_mem(ctx_, output_mems_[i], &output_attrs_[i]);
        if (ret < 0)
        {
            printf("rknn_set_io_mem fail! ret=%d\n", ret);
            exit(-1);
        }
#ifdef ALG_DEBUG
        printRKNNTensor(&(output_attrs_[i]));
#endif
    }
}

void CModule_cls_rk356x_impl::engine_run()
{
#ifdef ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> begin_time = std::chrono::system_clock::now();
#endif

    // Set Input Data
    input_attrs_[0].type = RKNN_TENSOR_UINT8;
    input_attrs_[0].fmt = RKNN_TENSOR_NHWC;

    // Copy input data to input tensor memory
    int width = input_attrs_[0].dims[2];
    int stride = input_attrs_[0].w_stride;

    if (width == stride)
    {
        memcpy(input_mems_[0]->virt_addr, src_resize_ptr_, width * input_attrs_[0].dims[1] * input_attrs_[0].dims[3]);
    }
    else
    {
        int height = input_attrs_[0].dims[1];
        int channel = input_attrs_[0].dims[3];
        // copy from src to dst with stride
        uint8_t *src_ptr = src_resize_ptr_;
        uint8_t *dst_ptr = (uint8_t *) input_mems_[0]->virt_addr;
        // width-channel elements
        int src_wc_elems = width * channel;
        int dst_wc_elems = stride * channel;
        for (int h = 0; h < height; ++h)
        {
            memcpy(dst_ptr, src_ptr, src_wc_elems);
            src_ptr += src_wc_elems;
            dst_ptr += dst_wc_elems;
        }
    }

#ifdef ALG_DEBUG
//    cv::Mat src_tmp(config_.net_inp_height, config_.net_inp_width, CV_8UC3, src_mat_.data());
//    cv::imwrite("src_tmp.jpg", src_tmp);
#endif

    // Set input tensor memory
    int ret = rknn_set_io_mem(ctx_, input_mems_[0], &input_attrs_[0]);
    if (ret < 0)
    {
        printf("rknn_set_io_mem fail! ret=%d\n", ret);
        exit(-1);
    }

    // Run
    ret = rknn_run(ctx_, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        exit(-1);
    }

#ifdef ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();
    std::printf("rv1126 inference time %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count());
#endif

    // Get Output
    int net_output_num = config_.output_names.size();
    if (data_out_.empty())
    {
        int output_num = 0;
        for (size_t idx = 0; idx < net_output_num; idx++)
        {
            output_num += output_attrs_[idx].n_elems;
        }
        data_out_.resize(output_num);
    }

    // post process
    memcpy(data_out_.data(), output_mems_[0]->virt_addr, sizeof(float) *  output_attrs_[0].n_elems);
#ifdef ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> begin_time_nms = std::chrono::system_clock::now();
    std::printf("postprocess0 time %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(begin_time_nms - end_time).count());
#endif
}
