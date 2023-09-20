#include "Module_det_impl.h"
#include "Module_det.h"
#include "Module_det_c_api.h"

#include "Module_det_rk35xx_impl.h"
#include "debug.h"

namespace rk35xx_det
{
    CModule_det::CModule_det()
    {
        impl_ = new CModule_det_rk35xx_impl();
    }

    CModule_det::~CModule_det()
    {
        if (ANY_POINTER_CAST(impl_, CModule_det_impl))
        {
            delete ANY_POINTER_CAST(impl_, CModule_det_impl);
        }
    }

    void CModule_det::init(const YoloConfig &config)
    {
        ANY_POINTER_CAST(impl_, CModule_det_impl)->init(config);
    }

    void CModule_det::deinit()
    {
#ifdef ALG_DEBUG
        AIALG_PRINT("release success begin\n");
#endif
        ANY_POINTER_CAST(impl_, CModule_det_impl)->deinit();
#ifdef ALG_DEBUG
        AIALG_PRINT("release success end!\n");
#endif
    }

    void CModule_det::process_batch(const ImageInfoUint8 *imageInfos, int batch_size)
    {
        ANY_POINTER_CAST(impl_, CModule_det_impl)->process(imageInfos, batch_size);
    }

    const BoxInfos *CModule_det::get_result()
    {
        return ANY_POINTER_CAST(impl_, CModule_det_impl)->get_result();
    }
}
///************************* c api *************************/
//void alg_det_init(Handle* handle, const net_config_tag_c* config)
//{
//    *handle = new CModule_det();
//
//    YoloConfig baseConfig;
//    baseConfig.input_names.resize(config->net_inp_num);
//    for (int idx = 0; idx < config->net_inp_num; ++idx)
//    {
//        baseConfig.input_names[idx] =  std::string(config->input_names[idx]);
//    }
//    baseConfig.output_names.resize(config->net_out_num);
//    for (int idx = 0; idx < config->net_out_num; ++idx)
//    {
//        baseConfig.output_names[idx] =  std::string(config->output_names[idx]);
//    }
//
//    baseConfig.weights_path = !config->weights_path ? "" : config->weights_path;
//    baseConfig.deploy_path = !config->deploy_path ? "" : config->deploy_path;
//    baseConfig.means[0] = config->means[0];
//    baseConfig.means[1] = config->means[1];
//    baseConfig.means[2] = config->means[2];
//
//    baseConfig.scales[0] = config->scales[0];
//    baseConfig.scales[1] = config->scales[1];
//    baseConfig.scales[2] = config->scales[2];
//
//    baseConfig.mean_length = config->mean_length;
//    baseConfig.net_inp_width = config->net_inp_width;
//    baseConfig.net_inp_height = config->net_inp_height;
//    baseConfig.net_inp_channels = config->net_inp_channels;
//
//    baseConfig.num_threads = config->num_threads;
//    ANY_POINTER_CAST(*handle, CModule_det)->init(baseConfig);
//}
//
//void alg_det_release(Handle handle)
//{
//    if(handle)
//    {
//        ANY_POINTER_CAST(handle, CModule_det)->deinit();
//        delete ANY_POINTER_CAST(handle, CModule_det);
//#if defined(ALG_DEBUG) || defined(ALPHAPOSE_DEBUG)
//        AIALG_PRINT("%d,%s\n", __LINE__, __FUNCTION__);
//#endif
//    }
//}
//
//void ALG_PUBLIC alg_det_run(Handle handle, const img_info_tag_c* img_info)
//{
//    if(3 == img_info->channels)
//    {
//        cv::Mat img = cv::Mat(img_info->height, img_info->width, CV_8UC3, img_info->data, img_info->stride);
//        ANY_POINTER_CAST(handle, CModule_det)->process(img);
//    }
//    else if(1 == img_info->channels)
//    {
//        cv::Mat img = cv::Mat(img_info->height, img_info->width, CV_8UC1, img_info->data, img_info->stride);
//        ANY_POINTER_CAST(handle, CModule_det)->process(img);
//    }
//    else
//    {
//        AIALG_PRINT("unsupported image format\n");
//    }
//}
//
//void ALG_PUBLIC alg_det_get_result(Handle handle, box_info_tag_c* det_info)
//{
//    const BoxInfo* det = ANY_POINTER_CAST(handle, CModule_det)->get_result();
//    det_info->x1 = det->x1;
//    det_info->y1 = det->y1;
//    det_info->x2 = det->x2;
//    det_info->y2 = det->y2;
//    det_info->label = det->label;
//    det_info->score = det->score;
//    det_info->frame_id = det->frame_id;
//}
