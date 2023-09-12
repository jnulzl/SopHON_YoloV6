#ifndef MODULE_DET_H
#define MODULE_DET_H

#include <string>
#include <vector>

#include "data_type.h"
#include "alg_define.h"

namespace rk35xx_det
{
    class AIALG_PUBLIC CModule_det
    {
    public:
        CModule_det();

        ~CModule_det();

        void init(const YoloConfig &config);

        void deinit();

        void process_batch(const ImageInfoUint8 *imageInfos, int batch_size);

        const BoxInfos *get_result();

    private:
        AW_ANY_POINTER impl_;
    };
}
#endif // MODULE_DET_H

