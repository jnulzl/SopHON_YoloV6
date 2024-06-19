//
// Created by lizhaoliang-os on 2020/6/9.
//
#include <algorithm>
#include <string.h>
#include <vector>
#include <numeric>
#include <iterator>
#include "Module_det_rv1126_impl.h"
#include "alg_define.h"
#include "debug.h"

#ifdef USE_RGA
#include "rga_func.h"
#endif

namespace rk35xx_det
{
#ifdef ALG_DEBUG
    static void printRKNNTensor(rknn_tensor_attr *attr)
    {
        AIALG_PRINT("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
               attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
               attr->n_elems, attr->size, attr->fmt, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
    }
#endif

    static void TopKIndex(const float* vec, int vec_len, int* topk_vec, int topk)
    {
        std::vector<size_t> vec_index(vec_len);
        std::iota(vec_index.begin(), vec_index.end(), 0);

        std::sort(vec_index.begin(), vec_index.end(),
                  [&vec](size_t index_1, size_t index_2) { return vec[index_1] > vec[index_2]; });

        int k_num = std::min<int>(vec_len, topk);

        for (int idx = 0; idx < k_num; ++idx)
        {
            topk_vec[idx] = vec_index[idx];
        }
    }

#ifdef USE_ZERO_COPY
    inline static int32_t __clip(float val, float min, float max)
    {
        float f = val <= min ? min : (val >= max ? max : val);
        return f;
    }

    static int8_t qnt_f32_to_affine(float f32, uint32_t zp, float scale)
    {
        float dst_val = (f32 / scale) + zp;
        uint8_t res = (uint8_t)__clip(dst_val, 0,255);
        return res;
    }

    static float deqnt_affine_to_f32(uint8_t qnt, uint32_t zp, float scale)
    {
        return ((float)qnt - (float)zp) * scale;
    }

    static void getMaxValAndIn(const uint8_t* data, int data_len, float scale, uint32_t zp, float& max_val, int& max_index)
    {
        max_val = -1.0e6;
        max_index = -1;
        for (int idx = 0; idx < data_len; idx++)
        {
            float score_tmp = deqnt_affine_to_f32(data[idx], zp, scale);
            if (score_tmp > max_val)
            {
                max_val = score_tmp;
                max_index = idx;
            }
        }
    }

    static void getTopKBoxesFromFloatOutput(const uint8_t* boxes,
                                            const float *indexs, const float *scores,
                                            float scale, uint32_t zp,
                                            int obj_num, int topk,
                                            std::vector<float> &topK_boxes_scores_labels)
    {
        std::vector<int> topKIndex(topk);
        TopKIndex(scores, obj_num, topKIndex.data(), topk);
        for (int idx = 0; idx < topk; ++idx)
        {
            int max_index = topKIndex[idx];
            float max_score = scores[max_index];
            topK_boxes_scores_labels[6 * idx + 0] = deqnt_affine_to_f32(boxes[4 * max_index + 0], zp, scale);
            topK_boxes_scores_labels[6 * idx + 1] = deqnt_affine_to_f32(boxes[4 * max_index + 1], zp, scale);
            topK_boxes_scores_labels[6 * idx + 2] = deqnt_affine_to_f32(boxes[4 * max_index + 2], zp, scale);
            topK_boxes_scores_labels[6 * idx + 3] = deqnt_affine_to_f32(boxes[4 * max_index + 3], zp, scale);
            topK_boxes_scores_labels[6 * idx + 4] = max_score;
            topK_boxes_scores_labels[6 * idx + 5] = indexs[max_index];
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
        std::vector<int> topKIndex(topk);
        TopKIndex(scores, obj_num, topKIndex.data(), topk);
        for (int idx = 0; idx < topk; ++idx)
        {
            int max_index = topKIndex[idx];
            float max_score = scores[max_index];
            topK_boxes_scores_labels[6 * idx + 0] = boxes[4 * max_index + 0];
            topK_boxes_scores_labels[6 * idx + 1] = boxes[4 * max_index + 1];
            topK_boxes_scores_labels[6 * idx + 2] = boxes[4 * max_index + 2];
            topK_boxes_scores_labels[6 * idx + 3] = boxes[4 * max_index + 3];
            topK_boxes_scores_labels[6 * idx + 4] = max_score;
            topK_boxes_scores_labels[6 * idx + 5] = indexs[max_index];
        }
    }
#endif
    static unsigned char *load_model(const char *filename, int *model_size)
    {
        FILE *fp = fopen(filename, "rb");
        if (fp == nullptr)
        {
            AIALG_PRINT("fopen %s fail!\n", filename);
            return NULL;
        }
        fseek(fp, 0, SEEK_END);
        int model_len = ftell(fp);
        unsigned char *model = (unsigned char *) malloc(model_len);
        fseek(fp, 0, SEEK_SET);
        if (model_len != fread(model, 1, model_len, fp))
        {
            AIALG_PRINT("fread %s fail!\n", filename);
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

    CModule_det_rv1126_impl::CModule_det_rv1126_impl() = default;

    CModule_det_rv1126_impl::~CModule_det_rv1126_impl() = default;

    void CModule_det_rv1126_impl::engine_deinit()
    {
#ifdef ALG_DEBUG
        AIALG_PRINT("release success begin\n");
#endif

#ifdef USE_ZERO_COPY
        for (int i = 0; i < inputs_mem_.size(); i++)
        {
            rknn_destroy_mem(ctx_, inputs_mem_[i]);
        }

        for (int i = 0; i < outputs_mem_.size(); i++)
        {
            rknn_destroy_mem(ctx_, outputs_mem_[i]);
        }
#endif
        if (model_buffer_)
        {
            free(model_buffer_);
        }

#ifdef USE_RGA
        RGA_deinit(&rga_ctx_);
#endif

        // Destroy rknn memory
        if(ctx_ >= 0)
        {
            rknn_destroy(ctx_);
        }
#ifdef ALG_DEBUG
        AIALG_PRINT("release success end!\n");
#endif
    }

    void CModule_det_rv1126_impl::engine_init()
    {
        // Load RKNN Model
        int model_len = 0;
        model_buffer_ = load_model(config_.weights_path.c_str(), &model_len);
        int ret = rknn_init(&ctx_, model_buffer_, model_len, 0);
        if (ret < 0)
        {
            AIALG_PRINT("rknn_init fail! ret=%d\n", ret);
            exit(-1);
        }

        // Get sdk and driver version
        rknn_sdk_version sdk_ver;
        ret = rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
        if (ret != RKNN_SUCC)
        {
            AIALG_PRINT("rknn_query fail! ret=%d\n", ret);
            exit(-1);
        }
        AIALG_PRINT("rknn_api/rknnrt version: %s, driver version: %s\n", sdk_ver.api_version, sdk_ver.drv_version);

        // Get Model Input Output Info
        rknn_input_output_num io_num;
        ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
        if (ret != RKNN_SUCC)
        {
            AIALG_PRINT("rknn_query fail! ret=%d\n", ret);
            exit(-1);
        }
        AIALG_PRINT("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

        AIALG_PRINT("input tensors: %d\n", io_num.n_input);
        input_attrs_.resize(io_num.n_input);
        memset(input_attrs_.data(), 0, io_num.n_input * sizeof(rknn_tensor_attr));
        for (int i = 0; i < io_num.n_input; i++)
        {
            input_attrs_[i].index = i;
            ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs_[i]), sizeof(rknn_tensor_attr));
            if (ret != RKNN_SUCC)
            {
                AIALG_PRINT("rknn_query fail! ret=%d\n", ret);
                exit(-1);
            }
#ifdef ALG_DEBUG
            printRKNNTensor(&(input_attrs_[i]));
#endif
        }
        AIALG_PRINT("output tensors: %d\n", io_num.n_output);
        output_attrs_.resize(io_num.n_output);
        memset(output_attrs_.data(), 0, io_num.n_output * sizeof(rknn_tensor_attr));
        for (int i = 0; i < io_num.n_output; i++)
        {
            output_attrs_[i].index = i;
            ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]), sizeof(rknn_tensor_attr));
            if (ret != RKNN_SUCC)
            {
                AIALG_PRINT("rknn_query fail! ret=%d\n", ret);
                exit(-1);
            }
#ifdef ALG_DEBUG
            printRKNNTensor(&(output_attrs_[i]));
#endif
        }

#ifdef USE_ZERO_COPY
        inputs_mem_.resize(io_num.n_input);
        outputs_mem_.resize(io_num.n_output);
        out_scales_.resize(io_num.n_output);
        out_zps_.resize(io_num.n_output);
        for (int i = 0; i < io_num.n_input; i++)
        {
            inputs_mem_[i] = rknn_create_mem(ctx_, input_attrs_[i].size);
            rknn_set_io_mem(ctx_, inputs_mem_[0], &input_attrs_[0]);
        }

        for (int i = 0; i < io_num.n_output; i++)
        {
            out_scales_[i] = output_attrs_[i].scale;
            out_zps_[i] = output_attrs_[i].zp;
            outputs_mem_[i] = rknn_create_mem(ctx_, output_attrs_[i].size);
            rknn_set_io_mem(ctx_, outputs_mem_[i], &output_attrs_[i]);
        }
#endif

#ifdef USE_RGA
        memset(&rga_ctx_, 0, sizeof(rga_context));
        RGA_init(&rga_ctx_);
#endif
        std::printf("Init success!\n");
    }

    void CModule_det_rv1126_impl::pre_process(const ImageInfoUint8 &imageInfo)
    {
#ifdef USE_RGA
#ifdef ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> begin_time_nms = std::chrono::system_clock::now();
#endif
        if (img_width_ != imageInfo.img_width || img_height_ != imageInfo.img_height)
        {
            img_width_ = imageInfo.img_width;
            img_height_ = imageInfo.img_height;
            float scale_wh = 1.0 * std::fmax(1.0 * config_.net_inp_height, 1.0 * config_.net_inp_width) /
                             std::fmax(1.0 * img_height_, 1.0 * img_width_);
            roi_new_width_ = img_width_ * scale_wh;
            roi_new_height_ = img_height_ * scale_wh;
        }
        rga_resize(&rga_ctx_, -1,
                   imageInfo.data, 0, 0, imageInfo.img_width, imageInfo.img_height,imageInfo.img_width, imageInfo.img_height,
                   inputs_mem_[0]->fd, nullptr, 0, 0, roi_new_width_, roi_new_height_, config_.net_inp_width, config_.net_inp_height);
#else // RGA
        CModule_det_impl::pre_process(imageInfo);
#ifdef ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> begin_time_nms = std::chrono::system_clock::now();
#endif
        // Set Input Data
        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].size = sizeof(uint8_t) * config_.net_inp_channels * config_.net_inp_height * config_.net_inp_width;
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].buf = src_resize_ptr_;

        int ret = rknn_inputs_set(ctx_, 1, inputs);
        if (ret < 0)
        {
            printf("rknn_input_set fail! ret=%d\n", ret);
            exit(1);
        }
#endif // RGA

#ifdef ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> end_time_nms = std::chrono::system_clock::now();
        AIALG_PRINT("rknn_inputs_set time %d ms\n", static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(end_time_nms - begin_time_nms).count()));
#endif
    }

    void CModule_det_rv1126_impl::engine_run()
    {
#ifdef ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> begin_time = std::chrono::system_clock::now();
#endif
        // Inference
        int ret = rknn_run(ctx_, nullptr);
        if (ret < 0)
        {
            AIALG_PRINT("rknn_run fail! ret=%d\n", ret);
            exit(-1);
        }
#ifdef ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();
        AIALG_PRINT("rv1126 inference time %d ms\n", static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count()));
        begin_time = std::chrono::system_clock::now();
#endif

#ifndef USE_ZERO_COPY
        // Get Output
        int net_output_num = config_.output_names.size();
        if(outputs_.empty())
        {
            outputs_.resize(net_output_num);
            for (int idx = 0; idx < net_output_num; ++idx)
            {
                outputs_[idx].want_float = 1;
            }
        }
        std::chrono::time_point<std::chrono::system_clock> begin_time_nmsaaa = std::chrono::system_clock::now();
        ret = rknn_outputs_get(ctx_, net_output_num, outputs_.data(), NULL);
        if (ret < 0)
        {
            printf("rknn_outputs_get fail! ret=%d\n", ret);
            exit(1);
        }

#ifdef ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> end_time_nmsaaa = std::chrono::system_clock::now();
        AIALG_PRINT("rknn_outputs_get time %d ms\n", static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(end_time_nmsaaa - begin_time_nmsaaa).count()));
#endif
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

#ifdef ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> begin_time_nms = std::chrono::system_clock::now();
#endif
#ifdef USE_ZERO_COPY
        const uint8_t* pred_bboxes = reinterpret_cast<const uint8_t*>(outputs_mem_[0]->logical_addr);
        const uint8_t* score_index = reinterpret_cast<const uint8_t*>(outputs_mem_[1]->logical_addr);
        for (int idx = 0; idx < output_attrs_[1].dims[1]; ++idx)
        {
            float max_score;
            int max_index;
            getMaxValAndIn(score_index + idx * output_attrs_[1].dims[0], output_attrs_[1].dims[0],
                           out_scales_[1], out_zps_[1],
                           max_score, max_index);
            max_scores_[idx] = max_score;
            max_indexs_[idx] = max_index;
        }
#ifdef ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> end_time_nms = std::chrono::system_clock::now();
        AIALG_PRINT("getMaxValAndIn time %d ms\n", static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(end_time_nms - begin_time_nms).count()));
#endif
        getTopKBoxesFromFloatOutput(pred_bboxes, max_indexs_.data(), max_scores_.data(),
                                    out_scales_[0], out_zps_[0],
                                    max_indexs_.size(), topK_, topK_boxes_scores_labels_);
#ifdef ALG_DEBUG
        end_time = std::chrono::system_clock::now();
        AIALG_PRINT("post_process1111 time %d ms\n", static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count()));
#endif
#else // USE_ZERO_COPY
        const float *pred_bboxes = reinterpret_cast<const float *>(outputs_[0].buf);
        const float *score_index = reinterpret_cast<const float *>(outputs_[1].buf);
        for (int idx = 0; idx < output_attrs_[1].dims[1]; ++idx)
        {
            float max_score;
            int max_index;
            getMaxValAndIn(score_index + idx * output_attrs_[1].dims[0], output_attrs_[1].dims[0], max_score, max_index);
            max_scores_[idx] = max_score;
            max_indexs_[idx] = max_index;
        }
#ifdef ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> end_time_nms = std::chrono::system_clock::now();
        AIALG_PRINT("getMaxValAndIn time %d ms\n", static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(end_time_nms - begin_time_nms).count()));
#endif
        getTopKBoxesFromFloatOutput(pred_bboxes, max_indexs_.data(), max_scores_.data(),
                                    max_indexs_.size(), topK_, topK_boxes_scores_labels_);

#ifdef ALG_DEBUG
        end_time = std::chrono::system_clock::now();
        AIALG_PRINT("post_process1111 time %d ms\n", static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count()));
#endif
#endif // USE_ZERO_COPY


#ifndef USE_ZERO_COPY
        rknn_outputs_release(ctx_, outputs_.size(), outputs_.data());
#endif
    }
}