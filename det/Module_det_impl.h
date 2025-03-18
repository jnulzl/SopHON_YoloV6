#ifndef MODULE_DET_IMPL_H
#define MODULE_DET_IMPL_H

#include <string>
#include <vector>
#include "data_type.h"

namespace bm1684x_det
{
    #define MAX_DET_NUM 16

    class CModule_det_impl
    {
    public:
        CModule_det_impl();

        virtual ~CModule_det_impl();

        void init(const YoloConfig &config);

        void deinit();

        void process(const ImageInfoUint8 *imageInfos, int batch_size);

        const BoxInfos *get_result();

        const YoloConfig* get_config() const;

    protected:
        virtual void engine_init() = 0;

        virtual void engine_deinit() = 0;

        virtual void engine_run() = 0;

        virtual void pre_process(const ImageInfoUint8 *imageInfos, int batch_size);

        virtual void post_process();

    protected:
        YoloConfig config_;
        std::vector<float> topK_boxes_scores_labels_;
        std::vector<float> max_scores_;
        std::vector<float> max_indexs_;
        std::vector<int> keep_indexs_;
        int topK_;

        std::vector<int> img_heights_;
        std::vector<int> img_widths_;
        std::vector<int> frame_ids_;

        std::vector<BoxInfo> boxs_tmp_;
        std::vector<BoxInfos> boxs_batch_;
    };
}
#endif // MODULE_DET_IMPL_H

