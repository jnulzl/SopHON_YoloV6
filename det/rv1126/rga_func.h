//
// Created by Li Zhaoliang on 2024/6/19.
//

#ifndef RK35XX_DET_RGA_FUNC_H
#define RK35XX_DET_RGA_FUNC_H

#include <dlfcn.h>
#include "RgaApi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int(* FUNC_RGA_INIT)();
typedef void(* FUNC_RGA_DEINIT)();
typedef int(* FUNC_RGA_BLIT)(rga_info_t *, rga_info_t *, rga_info_t *);

typedef struct _rga_context{
    void *rga_handle;
    FUNC_RGA_INIT init_func;
    FUNC_RGA_DEINIT deinit_func;
    FUNC_RGA_BLIT blit_func;
} rga_context;

int RGA_init(rga_context* rga_ctx);

void rga_resize(rga_context *rga_ctx,
                int src_fd, void *src_virt, int src_xoffset, int src_yoffset, int src_w, int src_h, int src_sw, int src_sh,
                int  dst_fd, void *dst_virt, int dst_xoffset, int dst_yoffset, int dst_w, int dst_h, int dst_sw, int dst_sh);

int RGA_deinit(rga_context* rga_ctx);

#ifdef __cplusplus
}
#endif

#endif //RK35XX_DET_RGA_FUNC_H
