//
// Created by lizhaoliang-os on 2020/6/9.
//

#ifndef MODULE_DET_RV1126_IMPL_H
#define MODULE_DET_RV1126_IMPL_H

#include "Module_det_impl.h"
#include "rknn_api.h"

#ifdef USE_RGA
#include "im2d.h"
#define USE_ZERO_COPY
#endif

namespace rk35xx_det
{
    class CModule_det_rv1126_impl : public CModule_det_impl
    {
    public:
        CModule_det_rv1126_impl();

        virtual ~CModule_det_rv1126_impl();

    private:
        virtual void engine_init() override;

        virtual void engine_deinit() override;

        virtual void engine_run() override;

        virtual void pre_process(const ImageInfoUint8 &imageInfo) override;

    private:
        rknn_context ctx_;
        unsigned char *model_buffer_;
        std::vector<rknn_tensor_attr> output_attrs_;
        std::vector<rknn_tensor_attr> input_attrs_;
#ifdef USE_ZERO_COPY
        // for zero copy
        std::vector<rknn_tensor_mem*> inputs_mem_;
        std::vector<rknn_tensor_mem*> outputs_mem_;
        std::vector<float> out_scales_;
        std::vector<uint32_t> out_zps_;
#else
        // for no zero copy
        std::vector<rknn_output> outputs_;
#endif
        // init rga context
#ifdef USE_RGA
        rga_buffer_t src_rga_buffer_;
        rga_buffer_t dst_rga_buffer_;
        im_rect src_rect_;
        im_rect dst_rect_;
#endif
    };
}
#endif // MODULE_BENCHNARK_RV1126_IMPL_H
