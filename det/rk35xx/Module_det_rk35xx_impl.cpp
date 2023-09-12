//
// Created by lizhaoliang-os on 2020/6/9.
//
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "rk35xx/Module_det_rk35xx_impl.h"
#include "alg_define.h"
#include "debug.h"

namespace rk35xx_det
{

#ifdef ALG_DEBUG
    static void printRKNNTensor(rknn_tensor_attr *attr)
    {
        printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
               attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
               attr->n_elems, attr->size, attr->fmt, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
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

#ifndef DET_USE_FLOAT_OUTPUT
    inline static int32_t __clip(float val, float min, float max)
    {
        float f = val <= min ? min : (val >= max ? max : val);
        return f;
    }

    static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
    {
        float dst_val = (f32 / scale) + zp;
        int8_t res = (int8_t)__clip(dst_val, -128, 127);
        return res;
    }

    static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale)
    {
        return ((float)qnt - (float)zp) * scale;
    }

    static void getTopKBoxesFromInt8Output(const int8_t* boxes, const int8_t* indexs, const int8_t* scores,
                             const std::vector<float>& out_scales_,
                             const std::vector<int32_t>& out_zps_,
                             int obj_num, int topk,
                             std::vector<float>& topK_boxes_scores_labels)
    {
        std::vector<bool> flag(obj_num, false);
        for (int idx = 0; idx < topk; ++idx)
        {
            float max_score = -1.0f;
            int max_index = -1;
            for (int idy = 0; idy < obj_num; ++idy)
            {
                if(flag[idy])
                {
                    continue;
                }
                float score_tmp = deqnt_affine_to_f32(scores[idy], out_zps_[2], out_scales_[2]);
                if(score_tmp > max_score)
                {
                    max_score = score_tmp;
                    max_index = idy;
                }
            }
            topK_boxes_scores_labels[6 * idx + 0] = deqnt_affine_to_f32(boxes[4 * max_index + 0],out_zps_[0], out_scales_[0]);
            topK_boxes_scores_labels[6 * idx + 1] = deqnt_affine_to_f32(boxes[4 * max_index + 1],out_zps_[0], out_scales_[0]);
            topK_boxes_scores_labels[6 * idx + 2] = deqnt_affine_to_f32(boxes[4 * max_index + 2],out_zps_[0], out_scales_[0]);
            topK_boxes_scores_labels[6 * idx + 3] = deqnt_affine_to_f32(boxes[4 * max_index + 3],out_zps_[0], out_scales_[0]);
            topK_boxes_scores_labels[6 * idx + 4] = max_score;
            topK_boxes_scores_labels[6 * idx + 5] = deqnt_affine_to_f32(indexs[max_index],out_zps_[1], out_scales_[1]);
            flag[max_index] = true;
        }
    }
#else

    static void getMaxValAndIn(const float* data, int data_len, float& max_val, int& max_index)
    {
        max_val = -1.0e6;
        max_index = -1;
        for (int idx = 0; idx < data_len; idx++)
        {
            if (data[idx] > max_val)
            {
                max_val = data[idx];
                max_index = idx;
            }
        }
    }

    static void getTopKBoxesFromFloatOutput(const float *boxes, const float *indexs, const float *scores,
                                            int obj_num, int topk,
                                            std::vector<float> &topK_boxes_scores_labels)
    {
        std::vector<bool> flag(obj_num, false);
        for (int idx = 0; idx < topk; ++idx)
        {
            float max_score = -1.0f;
            int max_index = -1;
            for (int idy = 0; idy < obj_num; ++idy)
            {
                if (flag[idy])
                {
                    continue;
                }
                float score_tmp = scores[idy];
                if (score_tmp > max_score)
                {
                    max_score = score_tmp;
                    max_index = idy;
                }
            }
            topK_boxes_scores_labels[6 * idx + 0] = boxes[4 * max_index + 0];
            topK_boxes_scores_labels[6 * idx + 1] = boxes[4 * max_index + 1];
            topK_boxes_scores_labels[6 * idx + 2] = boxes[4 * max_index + 2];
            topK_boxes_scores_labels[6 * idx + 3] = boxes[4 * max_index + 3];
            topK_boxes_scores_labels[6 * idx + 4] = max_score;
            topK_boxes_scores_labels[6 * idx + 5] = indexs[max_index];
            flag[max_index] = true;
//        scores[max_index] = -1.0f;
        }
    }

#endif

    CModule_det_rk35xx_impl::CModule_det_rk35xx_impl()
    {

    }

    CModule_det_rk35xx_impl::~CModule_det_rk35xx_impl()
    {

    }

    void CModule_det_rk35xx_impl::engine_deinit()
    {
#ifdef ALG_DEBUG
        AIALG_PRINT("release success begin\n");
#endif
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
        rknn_destroy(ctx_);

        if (model_buffer_)
        {
            free(model_buffer_);
        }
#ifdef ALG_DEBUG
        AIALG_PRINT("release success end!\n");
#endif
    }

    void CModule_det_rk35xx_impl::engine_init()
    {
        // init rga context
        memset(&src_rect_, 0, sizeof(src_rect_));
        memset(&dst_rect_, 0, sizeof(dst_rect_));
        memset(&src_rga_buffer_, 0, sizeof(src_rga_buffer_));
        memset(&dst_rga_buffer_, 0, sizeof(dst_rga_buffer_));

        // Load RKNN Model
        int model_len = 0;
        model_buffer_ = load_model(config_.weights_path.c_str(), &model_len);
        int ret = rknn_init(&ctx_, model_buffer_, model_len, 0, nullptr);
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
            input_mems_[i] = rknn_create_mem(ctx_, input_attrs_[i].size_with_stride);
            memset(input_mems_[i]->virt_addr, 0, input_attrs_[i].size_with_stride * sizeof(char));
            // Set Input Data
            input_attrs_[i].type = RKNN_TENSOR_UINT8;
            input_attrs_[i].size =
                    config_.net_inp_height * config_.net_inp_width * config_.net_inp_channels * sizeof(char);
            input_attrs_[i].fmt = RKNN_TENSOR_NHWC;
            // TODO -- The efficiency of pass through will be higher, we need adjust the layout of input to
            //         meet the use condition of pass through.
            input_attrs_[i].pass_through = 0;
            // 4.1.3 Set input buffer
            rknn_set_io_mem(ctx_, input_mems_[i], &(input_attrs_[i]));
#ifdef USE_RGA
            // 4.1.4 bind virtual address to rga virtual address
            dst_rga_buffer_ = wrapbuffer_virtualaddr((void *)input_mems_[i]->virt_addr,
                                                     config_.net_inp_width, config_.net_inp_height,
                                               RK_FORMAT_RGB_888);
#endif

#ifdef ALG_DEBUG
            printRKNNTensor(&(input_attrs_[i]));
#endif
        }
        printf("output tensors: %d\n", io_num.n_output);
        output_attrs_.resize(io_num.n_output);
        output_mems_.resize(io_num.n_output);
        out_scales_.resize(io_num.n_output);
        out_zps_.resize(io_num.n_output);
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
            // https://github.com/rockchip-linux/rknpu2/blob/master/examples/rknn_yolov5_android_apk_demo/app/src/main/cpp/yolo_image.cc#L163
            out_scales_[i] = output_attrs_[i].scale;
            out_zps_[i] = output_attrs_[i].zp;

#ifdef DET_USE_FLOAT_OUTPUT
            output_mems_[i] = rknn_create_mem(ctx_, output_attrs_[i].n_elems * sizeof(float));
            memset(output_mems_[i]->virt_addr, 0, output_attrs_[i].n_elems * sizeof(float));
            output_attrs_[i].type = RKNN_TENSOR_FLOAT32;
#else
            // 4.2.1 Create output tensor memory, output data type is int8, post_process need int8 data.
            output_mems_[i] = rknn_create_mem(ctx_, output_attrs_[i].n_elems * sizeof(char));
            memset(output_mems_[i]->virt_addr, 0, output_attrs_[i].n_elems * sizeof(char));
    //        output_attrs_[i].type = RKNN_TENSOR_INT8;
#endif
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

    void CModule_det_rk35xx_impl::pre_process(const ImageInfoUint8 &imageInfo)
    {
        CModule_det_impl::pre_process(imageInfo);

//#if 1
#ifdef USE_RGA
        /*--------------------------------pre_process with rga-----------------------------------*/
        // https://github.com/rockchip-linux/rknpu2/blob/master/examples/rknn_yolov5_android_apk_demo/app/src/main/cpp/yolo_image.cc#L236
        src_rga_buffer_ = wrapbuffer_virtualaddr((void*)src_resize_ptr_,
                                                 config_.net_inp_width, config_.net_inp_height,
                                                 RK_FORMAT_RGB_888); // wstride, hstride,

        IM_STATUS ret = imresize(src_rga_buffer_, dst_rga_buffer_);
        if (IM_STATUS_SUCCESS != ret) {
            printf("run_yolo: resize image with rga failed: %s\n", imStrError((IM_STATUS)ret));
            exit(-1);
        }
#else
        /*--------------------------------pre_process without rga-----------------------------------*/
        // Copy input data to input tensor memory
        int width = input_attrs_[0].dims[2];
        int stride = input_attrs_[0].w_stride;

        if (width == stride)
        {
            memcpy(input_mems_[0]->virt_addr, src_resize_ptr_,
                   width * input_attrs_[0].dims[1] * input_attrs_[0].dims[3]);
        } else
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
//    #ifdef ALG_DEBUG
//        cv::Mat src_tmp(config_.net_inp_height, config_.net_inp_width, CV_8UC3, src_resize_ptr_);
//        cv::imwrite("src_tmp.jpg", src_tmp);
//    #endif
#endif
    }

    void CModule_det_rk35xx_impl::engine_run()
    {
#ifdef ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> begin_time = std::chrono::system_clock::now();
#endif
        // Inference
        int ret = rknn_run(ctx_, nullptr);
        if (ret < 0)
        {
            printf("rknn_run fail! ret=%d\n", ret);
            exit(-1);
        }
#ifdef ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();
        AIALG_PRINT("RK35XX inference time %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count());
#endif

        size_t elem_num = topK_ * 6;
        if (topK_boxes_scores_labels_.size() < elem_num)
        {
            topK_boxes_scores_labels_.resize(elem_num);
        }

        if (max_scores_.empty())
        {
            max_scores_.resize( output_attrs_[0].dims[1]);
        }

        if (max_indexs_.empty())
        {
            max_indexs_.resize(output_attrs_[0].dims[1]);
        }

#ifdef DET_USE_FLOAT_OUTPUT
        const float *pred_bboxes = reinterpret_cast<const float *>(output_mems_[0]->virt_addr);
        const float *score_index = reinterpret_cast<const float *>(output_mems_[1]->virt_addr);
        for (int idx = 0; idx < output_attrs_[1].dims[1]; ++idx)
        {
            float max_score;
            int max_index;
            getMaxValAndIn(score_index + idx * output_attrs_[1].dims[2], output_attrs_[1].dims[2], max_score, max_index);
            max_scores_[idx] = max_score;
            max_indexs_[idx] = max_index;
        }
        getTopKBoxesFromFloatOutput(pred_bboxes, max_indexs_.data(), max_scores_.data(),
                                    max_indexs_.size(), topK_, topK_boxes_scores_labels_);
#else
        const int8_t * max_socres = reinterpret_cast<const int8_t *>(output_mems_[2]->virt_addr);
        const int8_t * max_index = reinterpret_cast<const int8_t *>(output_mems_[1]->virt_addr);
        const int8_t * pred_bboxes = reinterpret_cast<const int8_t *>(output_mems_[0]->virt_addr);

        getTopKBoxesFromInt8Output(pred_bboxes, max_index, max_socres,
                     out_scales_, out_zps_,
                     output_attrs_[2].n_elems, topK_, topK_boxes_scores_labels_);
#endif

#ifdef ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> begin_time_nms = std::chrono::system_clock::now();
        AIALG_PRINT("Postprocess0 time %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(begin_time_nms - end_time).count());
#endif
    }
}