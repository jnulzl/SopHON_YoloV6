#ifndef MODULE_CLS_C_API_H
#define MODULE_CLS_C_API_H

#include "data_type_api.h"

#ifdef __cplusplus
extern "C" {
#endif

void ALG_PUBLIC alg_cls_init(Handle* handle, const net_config_tag_c* config);

void ALG_PUBLIC alg_cls_release(Handle handle);

void ALG_PUBLIC alg_cls_run(Handle handle, const img_info_tag_c* img_info);

void ALG_PUBLIC alg_cls_get_result(Handle handle, cls_info_tag_c* cls_info);

#ifdef __cplusplus
}
#endif

#endif // MODULE_CLS_C_API_H

