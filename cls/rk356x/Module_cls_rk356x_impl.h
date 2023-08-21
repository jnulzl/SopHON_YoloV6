//
// Created by lizhaoliang-os on 2020/6/9.
//

#ifndef MODULE_CLS_RK365X_IMPL_H
#define MODULE_CLS_RK365X_IMPL_H

#include "Module_cls_impl.h"
#include "RgaUtils.h"
#include "rga.h"
#include "rknn_api.h"

class CModule_cls_rk356x_impl : public CModule_cls_impl
{
public:
    CModule_cls_rk356x_impl();
    virtual ~CModule_cls_rk356x_impl();

private:
    virtual void engine_init() override;
    virtual void engine_deinit() override;
    virtual void engine_run() override;

private:
    rknn_context ctx_;
    unsigned char* model_buffer_;
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_mem*> input_mems_;
    std::vector<rknn_tensor_attr> output_attrs_;
    std::vector<rknn_tensor_mem*> output_mems_;
};

#endif //MODULE_CLS_RK365X_IMPL_H
