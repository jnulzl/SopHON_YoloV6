#include "Module_det.h"
#include "Module_det_bm1684x_impl.h"

namespace bm1684x_det
{
    CModule_det::CModule_det()
    {
        impl_ = new ALG_ENGINE_IMPL(det, bm1684x);
    }
}