#include "register/register.h"


namespace domi {
REGISTER_CUSTOM_OP("Sub")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AscendWeightQuant")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::TVM);
}  // namespace domi