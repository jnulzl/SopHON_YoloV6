#include "opencv2/opencv.hpp"
#include "Module_det_impl.h"
#include "post_process.h"
#include "alg_define.h"
#include "debug.h"

namespace rk35xx_det
{
    CModule_det_impl::CModule_det_impl()
    {}

    CModule_det_impl::~CModule_det_impl()
    {}

    void CModule_det_impl::init(const YoloConfig &config)
    {
        config_ = config;
        des_mat_ = new cv::Mat(config_.net_inp_height, config_.net_inp_width, CV_8UC3);
        engine_init();
        topK_ = 100;

        boxs_res_.boxes = new BoxInfo[MAX_DET_NUM];
        boxs_res_.capacity = MAX_DET_NUM;
        boxs_res_.size = 0;
    }

    void CModule_det_impl::deinit()
    {
#ifdef ALG_DEBUG
        AIALG_PRINT("release success begin\n");
#endif
        engine_deinit();
        if (boxs_res_.boxes)
        {
            delete[] boxs_res_.boxes;
            boxs_res_.boxes = nullptr;
        }
        reinterpret_cast<cv::Mat *>(des_mat_)->release();
        delete reinterpret_cast<cv::Mat *>(des_mat_);
#ifdef ALG_DEBUG
        AIALG_PRINT("release success end!\n");
#endif
    }

    void CModule_det_impl::pre_process(const ImageInfoUint8 &imageInfo)
    {
        if (img_width_ != imageInfo.img_width || img_height_ != imageInfo.img_height)
        {
            img_width_ = imageInfo.img_width;
            img_height_ = imageInfo.img_height;
            float scale_wh = 1.0 * std::fmax(1.0 * config_.net_inp_height, 1.0 * config_.net_inp_width) /
                             std::fmax(1.0 * img_height_, 1.0 * img_width_);
            roi_new_width_ = img_width_ * scale_wh;
            roi_new_height_ = img_height_ * scale_wh;
        }
        cv::Rect roi;
        roi.x = 0;
        roi.y = 0;
        roi.width = roi_new_width_;
        roi.height = roi_new_height_;

        cv::Mat mat = cv::Mat(img_height_, img_width_, CV_8UC3, imageInfo.data, imageInfo.stride);
        cv::Mat des_tmp = *reinterpret_cast<cv::Mat *>(des_mat_);
        cv::resize(mat, des_tmp(roi), cv::Size(roi_new_width_, roi_new_height_), 0, 0, cv::INTER_NEAREST);
        cv::cvtColor(des_tmp(roi), des_tmp(roi), cv::COLOR_BGR2RGB);
        src_resize_ptr_ = reinterpret_cast<cv::Mat *>(des_mat_)->data; //rgb
    }

    void CModule_det_impl::process(const ImageInfoUint8 *imageInfos, int batch_size)
    {
#ifdef ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> begin_time_nms = std::chrono::system_clock::now();
#endif
        boxs_res_.frame_id = imageInfos[0].frame_id;
        pre_process(imageInfos[0]);
#ifdef ALG_DEBUG
        std::chrono::time_point<std::chrono::system_clock> end_time_nms = std::chrono::system_clock::now();
        AIALG_PRINT("preprocess time %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time_nms - begin_time_nms).count());
#endif

        engine_run();

#ifdef ALG_DEBUG
        begin_time_nms = std::chrono::system_clock::now();
        AIALG_PRINT("engine_run time %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(begin_time_nms - end_time_nms).count());
#endif

        post_process();

#ifdef ALG_DEBUG
        end_time_nms = std::chrono::system_clock::now();
        AIALG_PRINT("post_process time %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time_nms - begin_time_nms).count());
#endif
    }

    void CModule_det_impl::post_process()
    {
        int num_obj = topK_;
        //int batch_size = topK_boxes_scores_labels_.size() / num_obj / 6;
        if (keep_indexs_.size() < num_obj)
        {
            keep_indexs_.resize(num_obj);
            boxs_tmp_.resize(num_obj);
        }

        int det_obj = 0;
        non_max_suppression_opt(topK_boxes_scores_labels_.data(), num_obj, config_.num_cls,
                                config_.conf_thres, config_.nms_thresh,
                                config_.net_inp_height, config_.net_inp_width,
                                img_height_, img_width_,
                                boxs_tmp_.data(), keep_indexs_.data(), &det_obj);
        if (boxs_res_.capacity < det_obj)
        {
            if (boxs_res_.boxes)
            {
                delete[] boxs_res_.boxes;
                boxs_res_.boxes = nullptr;
            }
            boxs_res_.boxes = new BoxInfo[det_obj];
            boxs_res_.capacity = det_obj;
            boxs_res_.size = 0;
        }
        boxs_res_.size = det_obj;
        for (int idx = 0; idx < det_obj; ++idx)
        {
            boxs_res_.boxes[idx] = boxs_tmp_[keep_indexs_[idx]];
        }
    }

    const BoxInfos *CModule_det_impl::get_result()
    {
        return &boxs_res_;
    }
}