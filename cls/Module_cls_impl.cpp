#include "opencv2/opencv.hpp"
#include "Module_cls_impl.h"
#include "debug.h"

static void softmax(float* vec, size_t len)
{
    float sum = 0.0f;
    for (size_t i = 0; i < len; i++)
    {
        sum += std::exp(vec[i]);
    }
    for (size_t i = 0; i < len; i++)
    {
        vec[i] = exp(vec[i]) / sum;
    }
}

CModule_cls_impl::CModule_cls_impl()
{}

CModule_cls_impl::~CModule_cls_impl()
{}

void CModule_cls_impl::init(const BaseConfig &config)
{
    config_ = config;
    des_mat_ = new cv::Mat(config_.net_inp_height, config_.net_inp_width, CV_8UC3);
    engine_init();
}

void CModule_cls_impl::deinit()
{
    engine_deinit();

    reinterpret_cast<cv::Mat *>(des_mat_)->release();
    delete reinterpret_cast<cv::Mat *>(des_mat_);
#ifdef ALG_DEBUG
    std::printf("%d,%s\n", __LINE__, __FUNCTION__);
#endif
}

void CModule_cls_impl::pre_process()
{
//    std::cout << "This is default preprocess!!!!! " << std::endl;
}

void CModule_cls_impl::process(const cv::Mat &mat)
{
#ifdef ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> begin_time_nms = std::chrono::system_clock::now();
#endif
    cv::resize(mat, *reinterpret_cast<cv::Mat *>(des_mat_),
               cv::Size(config_.net_inp_width, config_.net_inp_height), 0, 0, cv::INTER_NEAREST);
    src_resize_ptr_ = reinterpret_cast<cv::Mat *>(des_mat_)->data; //bgr

#ifdef ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> end_time_nms = std::chrono::system_clock::now();
    std::printf("preprocess time %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time_nms - begin_time_nms).count());
    begin_time_nms = std::chrono::system_clock::now();
#endif

    engine_run();

#ifdef ALG_DEBUG
    end_time_nms = std::chrono::system_clock::now();
    std::printf("engine_run time %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time_nms - begin_time_nms).count());
#endif

    post_process();
}

void CModule_cls_impl::post_process()
{
    float *score_bs = data_out_.data();
    softmax(score_bs, data_out_.size());
    int max_index = 0;
    float max_score = score_bs[0];
    for (int idx = 1; idx < data_out_.size(); ++idx)
    {
        if (score_bs[idx] > max_score)
        {
            max_score = score_bs[idx];
            max_index = idx;
        }
    }
    clsInfo_.label = max_index;
    clsInfo_.score = max_score;
}

const ClsInfo &CModule_cls_impl::get_result()
{
    return clsInfo_;
}
