//
// Created by lizhaoliang-os on 2020/6/9.
//
#include <algorithm>
#include <numeric>
#include <iterator>
#include <memory>
#include <cassert>
#include "opencv2/opencv.hpp"
#include "Module_det_bm1684x_impl.h"
#include "alg_define.h"
#include "debug.h"

#include <omp.h>

#define USE_ASPECT_RATIO 1

namespace bm1684x_det
{
    int arg_max(const float* vec, int vec_len) {
        return static_cast<int>(std::distance(vec, std::max_element(vec, vec + vec_len)));
    }

    int arg_min(const float* vec, int vec_len) {
        return static_cast<int>(std::distance(vec, std::min_element(vec, vec + vec_len)));
    }

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

    static void getMaxValAndIn(const float* data, int data_len, float& max_val, int& max_index)
    {
        max_index = arg_max(data, data_len);
        max_val = data[max_index];
    }

    static void getTopKBoxesFromFloatOutput(const float *boxes, const float *indexs, const float *scores,
                                            int obj_num, int topk,
                                            float* topK_boxes_scores_labels)
    {
        std::vector<int> topKIndex(topk);
        TopKIndex(scores, obj_num, topKIndex.data(), topk);
        for (int idx = 0; idx < topk; ++idx)
        {
            int max_index = topKIndex[idx];
            topK_boxes_scores_labels[6 * idx + 0] = boxes[4 * max_index + 0];
            topK_boxes_scores_labels[6 * idx + 1] = boxes[4 * max_index + 1];
            topK_boxes_scores_labels[6 * idx + 2] = boxes[4 * max_index + 2];
            topK_boxes_scores_labels[6 * idx + 3] = boxes[4 * max_index + 3];
            topK_boxes_scores_labels[6 * idx + 4] = scores[max_index];
            topK_boxes_scores_labels[6 * idx + 5] = indexs[max_index];
        }
    }

    float CModule_det_bm1684x_impl::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *pIsAligWidth)
    {
        float ratio;
        float r_w = (float)dst_w / src_w;
        float r_h = (float)dst_h / src_h;
        if (r_h > r_w){
            *pIsAligWidth = true;
            ratio = r_w;
        }
        else{
            *pIsAligWidth = false;
            ratio = r_h;
        }
        return ratio;
    }

    void CModule_det_bm1684x_impl::convert_image_info_to_bm_image(const ImageInfoUint8 *img_info, bm_image *bm_img)
    {
        bm_img->image_private = reinterpret_cast<bm_image_private*>(img_info->data);
        bm_img->data_type = DATA_TYPE_EXT_1N_BYTE; // only support uint8_t
        bm_img->image_format = FORMAT_BGR_PACKED; // only support BGR_PACKED
        bm_img->width = img_info->img_width;
        bm_img->height = img_info->img_height;
    }

    CModule_det_bm1684x_impl::CModule_det_bm1684x_impl() = default;

    CModule_det_bm1684x_impl::~CModule_det_bm1684x_impl() = default;

    void CModule_det_bm1684x_impl::engine_deinit()
    {
#ifdef ALG_DEBUG
        AIALG_PRINT("release success begin\n");
#endif
        bm_image_free_contiguous_mem(m_max_batch_, m_resized_imgs_.data());
        for(int idx = 0; idx < m_max_batch_; idx++)
        {
            bm_image_destroy(m_resized_imgs_[idx]);
        }
#ifdef ALG_DEBUG
        AIALG_PRINT("release success end!\n");
#endif
    }

    void CModule_det_bm1684x_impl::engine_init()
    {
        AIALG_PRINT("set device id: %d\n", config_.device_id);
        m_handle_ = std::make_shared<BMNNHandle>(config_.device_id);

        // Load bmodel
        m_bmContext_ = std::make_shared<BMNNContext>(m_handle_, config_.weights_path.c_str());
        //1. get network
        m_bmNetwork_ = m_bmContext_->network(0);

        //2. get input
        m_max_batch_ = m_bmNetwork_->maxBatch();
        auto tensor = m_bmNetwork_->inputTensor(0);
        m_net_h_ = tensor->get_shape()->dims[2];
        m_net_w_ = tensor->get_shape()->dims[3];

        //3. get output
        m_output_num_ = m_bmNetwork_->outputTensorNum();
        assert(output_num == 1 || output_num == 2);

        //4. initialize bmimages
        m_resized_imgs_.resize(m_max_batch_);
        // some API only accept bm_image whose stride is aligned to 64
        int aligned_net_w = FFALIGN(m_net_w_, 64);
        int strides[3] = {aligned_net_w, aligned_net_w, aligned_net_w};
        for(int idx = 0; idx < m_max_batch_; idx++)
        {
            auto ret= bm_image_create(m_bmContext_->handle(), m_net_h_, m_net_w_, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &m_resized_imgs_[idx], strides);
            assert(BM_SUCCESS == ret);
        }
        bm_image_alloc_contiguous_mem(m_max_batch_, m_resized_imgs_.data());

        // 5.converto
        float input_scale = tensor->get_scale();
        input_scale = input_scale * 1.0 / 255.f;
        m_converto_attr_.alpha_0 = input_scale;
        m_converto_attr_.beta_0 = 0;
        m_converto_attr_.alpha_1 = input_scale;
        m_converto_attr_.beta_1 = 0;
        m_converto_attr_.alpha_2 = input_scale;
        m_converto_attr_.beta_2 = 0;
    }

    void CModule_det_bm1684x_impl::pre_process(const ImageInfoUint8 *imageInfos, int batch_size)
    {
        #pragma omp parallel for // num_threads(batch_size)
        for(int i = 0; i < batch_size; ++i)
        {
#ifdef ALG_DEBUG
            printf("i = %d, I am Thread %d, total thread num : %d\n", i, omp_get_thread_num(), omp_get_num_threads());
#endif
            bm_image image_tmp;
            convert_image_info_to_bm_image(&imageInfos[i], &image_tmp);
            bm_image image_aligned;
            bool need_copy = image_tmp.width & (64-1);
            if(need_copy)
            {
                int stride1[3], stride2[3];
                bm_image_get_stride(image_tmp, stride1);
                stride2[0] = FFALIGN(stride1[0], 64);
                stride2[1] = FFALIGN(stride1[1], 64);
                stride2[2] = FFALIGN(stride1[2], 64);
                bm_image_create(m_bmContext_->handle(), image_tmp.height, image_tmp.width,
                                image_tmp.image_format, image_tmp.data_type, &image_aligned, stride2);
                bm_image_alloc_dev_mem(image_aligned, BMCV_IMAGE_FOR_IN);
                bmcv_copy_to_atrr_t copyToAttr;
                memset(&copyToAttr, 0, sizeof(copyToAttr));
                copyToAttr.start_x = 0;
                copyToAttr.start_y = 0;
                copyToAttr.if_padding = 1;
                bmcv_image_copy_to(m_bmContext_->handle(), copyToAttr, image_tmp, image_aligned);
            }
            else
            {
                image_aligned = image_tmp;
            }
#if USE_ASPECT_RATIO
            bool isAlignWidth = false;
            float ratio = get_aspect_scaled_ratio(image_tmp.width, image_tmp.height, m_net_w_, m_net_h_, &isAlignWidth);
            bmcv_padding_atrr_t padding_attr;
            //memset(&padding_attr, 0, sizeof(padding_attr));
            padding_attr.dst_crop_sty = 0;
            padding_attr.dst_crop_stx = 0;
            padding_attr.padding_b = 114;
            padding_attr.padding_g = 114;
            padding_attr.padding_r = 114;
            padding_attr.if_memset = 1;
            if (isAlignWidth)
            {
                padding_attr.dst_crop_h = image_tmp.height*ratio;
                padding_attr.dst_crop_w = m_net_w_;

//                int ty1 = static_cast<int>((m_net_h_ - padding_attr.dst_crop_h) / 2);
//                int ty1 = static_cast<int>(m_net_h_ - padding_attr.dst_crop_h);
//                padding_attr.dst_crop_sty = 0;
//                padding_attr.dst_crop_stx = 0;
            }
            else
            {
                padding_attr.dst_crop_h = m_net_h_;
                padding_attr.dst_crop_w = image_tmp.width*ratio;

//                int tx1 = static_cast<int>((m_net_w_ - padding_attr.dst_crop_w) / 2);
//                int tx1 = static_cast<int>(m_net_w_ - padding_attr.dst_crop_w);
//                padding_attr.dst_crop_sty = 0;
//                padding_attr.dst_crop_stx = tx1;
            }

            bmcv_rect_t crop_rect{0, 0, image_tmp.width, image_tmp.height};
            auto ret = bmcv_image_vpp_convert_padding(m_bmContext_->handle(), 1, image_aligned, &m_resized_imgs_[i],
                &padding_attr, &crop_rect, BMCV_INTER_NEAREST);
#else
            auto ret = bmcv_image_vpp_convert(m_bmContext_->handle(), 1, image_tmp, &m_resized_imgs_[i]);
#endif
            assert(BM_SUCCESS == ret);

//ifdef ALG_DEBUG
#if 0
            cv::Mat resized_img;
            cv::bmcv::toMAT(&m_resized_imgs_[i], resized_img);
            std::string fname = cv::format("resized_img_%d.jpg", i);
            cv::imwrite(fname, resized_img);
#endif
            if(need_copy)
            {
                bm_image_destroy(image_aligned);
            }
        }

        std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork_->inputTensor(0);
        //3. attach to tensor
        //assert(batch_size == m_max_batch_)
        if(batch_size != m_max_batch_)
        {
            batch_size = m_bmNetwork_->get_nearest_batch(batch_size);
        }
        bm_device_mem_t input_dev_mem;
        bm_image_get_contiguous_device_mem(batch_size, m_resized_imgs_.data(), &input_dev_mem);
        input_tensor->set_device_mem(&input_dev_mem);
        input_tensor->set_shape_by_dim(0, batch_size);  // set real batch number
    }

    void CModule_det_bm1684x_impl::engine_run()
    {
#ifdef ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> begin_time = std::chrono::system_clock::now();
#endif
        // Inference
        int ret = m_bmNetwork_->forward();
        if (ret < 0)
        {
            AIALG_PRINT("forward fail! ret=%d\n", ret);
        }
#ifdef ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();
        AIALG_PRINT("BM1684 inference time %ld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count());
#endif

        std::vector<std::shared_ptr<BMNNTensor>> outputTensors(m_output_num_);
        for(int idx = 0; idx < m_output_num_; idx++)
        {
            outputTensors[idx] = m_bmNetwork_->outputTensor(idx);
        }
        int batch_size = outputTensors[1]->get_shape()->dims[0];
        int middle_dim = outputTensors[1]->get_shape()->dims[1];
        int num_cls = outputTensors[1]->get_shape()->dims[2];

        size_t elem_num = batch_size * topK_ * 6;
        if (topK_boxes_scores_labels_.size() < elem_num)
        {
            topK_boxes_scores_labels_.resize(elem_num);
        }
//        AIALG_PRINT("batch_size=%d, middle_dim=%d, num_cls=%d\n", batch_size, middle_dim, num_cls);
        size_t elem_num2 = batch_size * middle_dim; // batch * 8400(for 640 x 640 input)
        if (max_scores_.empty())
        {
            max_scores_.resize( elem_num2);
        }

        if (max_indexs_.empty())
        {
            max_indexs_.resize(elem_num2);
        }
        const float *pred_bboxes = reinterpret_cast<const float *>(outputTensors[0]->get_cpu_data());
        const float *score_index = reinterpret_cast<const float *>(outputTensors[1]->get_cpu_data());
        #pragma omp parallel for // num_threads(batch_size)
        for (int bs = 0; bs < batch_size; ++bs)
        {
            for (int idx = 0; idx < middle_dim; ++idx)
            {
                float max_score;
                int max_index;
                getMaxValAndIn(score_index + bs * (middle_dim * num_cls) + idx * num_cls, num_cls, max_score, max_index);
                max_scores_[bs * middle_dim + idx] = max_score;
                max_indexs_[bs * middle_dim + idx] = max_index;
            }
            getTopKBoxesFromFloatOutput(pred_bboxes + bs * (middle_dim * 4),
                                        max_indexs_.data() + bs * middle_dim,
                                        max_scores_.data() + bs * middle_dim,
                                        middle_dim, topK_, topK_boxes_scores_labels_.data() + bs * topK_ * 6);
        }

#ifdef ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> begin_time_nms = std::chrono::system_clock::now();
        AIALG_PRINT("Postprocess0 time %ld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(begin_time_nms - end_time).count());
#endif
    }
}