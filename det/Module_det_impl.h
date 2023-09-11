#ifndef MODULE_DET_IMPL_H
#define MODULE_DET_IMPL_H

#include <string>
#include <vector>
#include "data_type.h"

class CModule_det_impl
{
public:
	CModule_det_impl();
	virtual ~CModule_det_impl() ;

    void init(const YoloConfig &config);

    void deinit();

    void process(const ImageInfoUint8 *imageInfos, int batch_size);

	const BoxInfos* get_result();

protected:
    virtual void engine_init() = 0;
    virtual void engine_deinit() = 0;
    virtual void engine_run() = 0;
	virtual void pre_process(const ImageInfoUint8& imageInfo);
    virtual void post_process();

protected:
    YoloConfig config_;
    std::vector<float> topK_boxes_scores_labels_;
    std::vector<int> keep_indexs_;
    int img_height_;
    int img_width_;
    int frame_id_;
    float roi_new_width_;
    float roi_new_height_;

    std::vector<BoxInfo> boxs_tmp_;
    BoxInfos boxs_res_;

    int topK_;
    void* des_mat_;
    uint8_t* src_resize_ptr_;
};

#endif // MODULE_DET_IMPL_H

