#include "opencv2/opencv.hpp"
#include "Module_det_impl.h"
#include "post_process.h"
#include "alg_define.h"
#include "debug.h"

namespace bm1684x_det
{
    CModule_det_impl::CModule_det_impl() = default;

    CModule_det_impl::~CModule_det_impl() = default;

    void CModule_det_impl::init(const YoloConfig &config)
    {
        config_ = config;
        engine_init();
        topK_ = 100;

        img_heights_.resize(config_.batch_size);
        img_widths_.resize(config_.batch_size);
        frame_ids_.resize(config_.batch_size);
        boxs_batch_.resize(config_.batch_size);
        for (int idx = 0; idx < config_.batch_size; ++idx)
        {
            boxs_batch_[idx].boxes = new BoxInfo[MAX_DET_NUM];
            boxs_batch_[idx].capacity = MAX_DET_NUM;
            boxs_batch_[idx].size = 0;
        }
    }

    void CModule_det_impl::deinit()
    {
#ifdef ALG_DEBUG
        AIALG_PRINT("release success begin\n");
#endif
        engine_deinit();
        for (int idx = 0; idx < config_.batch_size; ++idx)
        {
            if(boxs_batch_[idx].boxes)
            {
                delete[] boxs_batch_[idx].boxes;
            }
        }
#ifdef AI_ALG_DEBUG
        AIALG_PRINT("release success!");
#endif
    }

    void CModule_det_impl::pre_process(const ImageInfoUint8 *imageInfos, int batch_size)
    {

    }

    void CModule_det_impl::process(const ImageInfoUint8 *imageInfos, int batch_size)
    {
        for (int bs = 0; bs < config_.batch_size; ++bs)
        {
            img_heights_[bs] = imageInfos[bs].img_height;
            img_widths_[bs] = imageInfos[bs].img_width;
            frame_ids_[bs] = imageInfos[bs].frame_id;
        }

#ifdef ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> begin_time_nms = std::chrono::system_clock::now();
#endif

        pre_process(imageInfos, batch_size);
#ifdef ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> end_time_nms = std::chrono::system_clock::now();
        AIALG_PRINT("preprocess time %d ms\n", static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(end_time_nms - begin_time_nms).count()));
#endif

        engine_run();

#ifdef ALG_DEBUG
        begin_time_nms = std::chrono::system_clock::now();
        AIALG_PRINT("engine_run time %d ms\n", static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(begin_time_nms - end_time_nms).count()));
#endif

        post_process();

#ifdef ALG_DEBUG
        end_time_nms = std::chrono::system_clock::now();
        AIALG_PRINT("post_process time %d ms\n", static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(end_time_nms - begin_time_nms).count()));
#endif
    }

    void CModule_det_impl::post_process()
    {
        if (keep_indexs_.size() < topK_)
        {
            keep_indexs_.resize(topK_);
            boxs_tmp_.resize(topK_);
        }
        for (int bs = 0; bs < config_.batch_size; ++bs)
        {
            int det_obj = 0;
            non_max_suppression_opt(topK_boxes_scores_labels_.data() + bs * topK_ * 6, topK_, config_.num_cls,
                                    config_.conf_thres, config_.nms_thresh,
                                    config_.net_inp_height, config_.net_inp_width, img_heights_[bs], img_widths_[bs],
                                    boxs_tmp_.data(), keep_indexs_.data(), &det_obj);
            if (boxs_batch_[bs].capacity < det_obj)
            {
                if(boxs_batch_[bs].boxes)
                {
                    delete[] boxs_batch_[bs].boxes;
                }
                boxs_batch_[bs].boxes = new BoxInfo[det_obj];
                boxs_batch_[bs].capacity = det_obj;
                boxs_batch_[bs].size = 0;
            }
            boxs_batch_[bs].size = det_obj;
            boxs_batch_[bs].frame_id = frame_ids_[bs];
            for (int idx = 0; idx < det_obj; ++idx)
            {
                boxs_batch_[bs].boxes[idx] = boxs_tmp_[keep_indexs_[idx]];
            }
        }
    }

    const BoxInfos *CModule_det_impl::get_result()
    {
        return boxs_batch_.data();
    }

    const YoloConfig* CModule_det_impl::get_config() const
    {
        return &config_;
    }
}