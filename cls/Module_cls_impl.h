#ifndef MODULE_CLS_IMPL_H
#define MODULE_CLS_IMPL_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "data_type.h"

class CModule_cls_impl
{
public:
	CModule_cls_impl();
	virtual ~CModule_cls_impl() ;

    void init(const BaseConfig &config);

    void deinit();

    void process(const cv::Mat& mat);

	const ClsInfo& get_result();

protected:
    virtual void engine_init() = 0;
    virtual void engine_deinit() = 0;
    virtual void engine_run() = 0;
	virtual void pre_process();
    virtual void post_process();

protected:
    BaseConfig config_;
    std::vector<float> data_out_;
    void* des_mat_;
    uint8_t* src_resize_ptr_;
    ClsInfo clsInfo_;
};

#endif // MODULE_CLS_IMPL_H

