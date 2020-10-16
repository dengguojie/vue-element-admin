#include "register/register.h"


namespace domi {
REGISTER_CUSTOM_OP("AscendDequant")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AscendDequant")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi