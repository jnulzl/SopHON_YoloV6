#include "Module_det_impl.h"
#include "Module_det.h"

#include "debug.h"

namespace bm1684x_det
{

    CModule_det::~CModule_det()
    {
        delete ANY_POINTER_CAST(impl_, CModule_det_impl);
    }

    void CModule_det::init(const YoloConfig &config)
    {
        ANY_POINTER_CAST(impl_, CModule_det_impl)->init(config);
    }

    void CModule_det::deinit()
    {
#ifdef ALG_DEBUG
        AIALG_PRINT("release success begin\n");
#endif
        if (ANY_POINTER_CAST(impl_, CModule_det_impl))
        {
            ANY_POINTER_CAST(impl_, CModule_det_impl)->deinit();
        }
#ifdef ALG_DEBUG
        AIALG_PRINT("release success end!\n");
#endif
    }

    void CModule_det::process_batch(const ImageInfoUint8 *imageInfos, int batch_size)
    {
        ANY_POINTER_CAST(impl_, CModule_det_impl)->process(imageInfos, batch_size);
    }

    const BoxInfos *CModule_det::get_result()
    {
        return ANY_POINTER_CAST(impl_, CModule_det_impl)->get_result();
    }

    const YoloConfig* CModule_det::get_config() const
    {
        return ANY_POINTER_CAST(impl_, CModule_det_impl)->get_config();
    }
}
