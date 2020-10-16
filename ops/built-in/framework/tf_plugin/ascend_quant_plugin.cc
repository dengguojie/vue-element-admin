#include "register/register.h"


namespace domi {
REGISTER_CUSTOM_OP("AscendQuant")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AscendQuant")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi