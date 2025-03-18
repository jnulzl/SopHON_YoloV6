//
// Created by lizhaoliang-os on 2020/6/9.
//

#ifndef MODULE_DET_BM1684X_IMPL_H
#define MODULE_DET_BM1684X_IMPL_H

#include "libavutil/macros.h"
#include "bmnn_utils.h"
#include "utils.hpp"
#include "bm_wrapper.hpp"

#include "Module_det_impl.h"

namespace bm1684x_det
{
    class CModule_det_bm1684x_impl : public CModule_det_impl
    {
    public:
        CModule_det_bm1684x_impl();

        virtual ~CModule_det_bm1684x_impl();

    private:
        virtual void engine_init() override;

        virtual void engine_deinit() override;

        virtual void engine_run() override;

        virtual void pre_process(const ImageInfoUint8 *imageInfos, int batch_size) override;

    private:
        void convert_image_info_to_bm_image(const ImageInfoUint8 *img_info, bm_image *bm_img);

        float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *pIsAligWidth);

    private:
        BMNNHandlePtr m_handle_;
        std::shared_ptr<BMNNContext> m_bmContext_;
        std::shared_ptr<BMNNNetwork> m_bmNetwork_;
        std::vector<bm_image> m_resized_imgs_;
        std::vector<bm_image> m_converto_imgs_;

        int m_net_h_;
        int m_net_w_;
        int m_max_batch_;
        int m_output_num_;

        bmcv_convert_to_attr m_converto_attr_;
        bm_image_data_format_ext m_img_dtype_;

        std::vector<float> out_scales_;
        std::vector<int32_t> out_zps_;
    };
}
#endif //MODULE_DET_BM1684X_IMPL_H
