#include "register/register.h"


namespace domi {
REGISTER_CUSTOM_OP("AscendAntiQuant")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AscendAntiQuant")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi