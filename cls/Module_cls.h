#ifndef MODULE_CLS_H
#define MODULE_CLS_H

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "data_type.h"
#include "alg_define.h"

class ALG_PUBLIC CModule_cls
{
public:
	CModule_cls();

	~CModule_cls();

	void init(const BaseConfig& config);

    void deinit();

    void process(const cv::Mat& mat);

    const ClsInfo& get_result();

private:
	AW_ANY_POINTER impl_;
};

#endif // MODULE_CLS_H

