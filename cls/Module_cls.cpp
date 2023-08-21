#include "Module_cls_impl.h"
#include "Module_cls.h"
#include "Module_cls_c_api.h"

#include "Module_cls_rk356x_impl.h"
#include "debug.h"

CModule_cls::CModule_cls()
{
    impl_ = new CModule_cls_rk356x_impl();
}

CModule_cls::~CModule_cls()
{
}

void CModule_cls::init(const BaseConfig& config)
{
    ANY_POINTER_CAST(impl_, CModule_cls_impl)->init(config);
}

void CModule_cls::deinit()
{
    ANY_POINTER_CAST(impl_, CModule_cls_impl)->deinit();
    
    delete ANY_POINTER_CAST(impl_, CModule_cls_impl);
#if defined(ALG_DEBUG) || defined(ALPHAPOSE_DEBUG)
    std::printf("%d,%s\n", __LINE__, __FUNCTION__);
#endif
}

void CModule_cls::process(const cv::Mat& mat)
{
    ANY_POINTER_CAST(impl_, CModule_cls_impl)->process(mat);
}

const ClsInfo& CModule_cls::get_result()
{
    return ANY_POINTER_CAST(impl_, CModule_cls_impl)->get_result();
}

/************************* c api *************************/
void alg_cls_init(Handle* handle, const net_config_tag_c* config)
{
    *handle = new CModule_cls();

    BaseConfig baseConfig;
    baseConfig.input_names.resize(config->net_inp_num);
    for (int idx = 0; idx < config->net_inp_num; ++idx)
    {
        baseConfig.input_names[idx] =  std::string(config->input_names[idx]);
    }
    baseConfig.output_names.resize(config->net_out_num);
    for (int idx = 0; idx < config->net_out_num; ++idx)
    {
        baseConfig.output_names[idx] =  std::string(config->output_names[idx]);
    }

    baseConfig.weights_path = !config->weights_path ? "" : config->weights_path;
    baseConfig.deploy_path = !config->deploy_path ? "" : config->deploy_path;
    baseConfig.means[0] = config->means[0];
    baseConfig.means[1] = config->means[1];
    baseConfig.means[2] = config->means[2];

    baseConfig.scales[0] = config->scales[0];
    baseConfig.scales[1] = config->scales[1];
    baseConfig.scales[2] = config->scales[2];

    baseConfig.mean_length = config->mean_length;
    baseConfig.net_inp_width = config->net_inp_width;
    baseConfig.net_inp_height = config->net_inp_height;
    baseConfig.net_inp_channels = config->net_inp_channels;

    baseConfig.num_threads = config->num_threads;
    ANY_POINTER_CAST(*handle, CModule_cls)->init(baseConfig);
}

void alg_cls_release(Handle handle)
{
    if(handle)
    {
        ANY_POINTER_CAST(handle, CModule_cls)->deinit();
        delete ANY_POINTER_CAST(handle, CModule_cls);
#if defined(ALG_DEBUG) || defined(ALPHAPOSE_DEBUG)
        std::printf("%d,%s\n", __LINE__, __FUNCTION__);
#endif
    }
}

void ALG_PUBLIC alg_cls_run(Handle handle, const img_info_tag_c* img_info)
{
    if(3 == img_info->channels)
    {
        cv::Mat img = cv::Mat(img_info->height, img_info->width, CV_8UC3, img_info->data, img_info->stride);
        ANY_POINTER_CAST(handle, CModule_cls)->process(img);
    }
    else if(1 == img_info->channels)
    {
        cv::Mat img = cv::Mat(img_info->height, img_info->width, CV_8UC1, img_info->data, img_info->stride);
        ANY_POINTER_CAST(handle, CModule_cls)->process(img);
    }
    else
    {
        std::printf("unsupported image format\n");
    }
}

void ALG_PUBLIC alg_cls_get_result(Handle handle, cls_info_tag_c* cls_info)
{
    const ClsInfo& cls = ANY_POINTER_CAST(handle, CModule_cls)->get_result();
    cls_info->label = cls.label;
    cls_info->score = cls.score;
    cls_info->frame_id = cls.frame_id;
}
