#include "register/register.h"

namespace domi {
REGISTER_CUSTOM_OP("EncodeJpegVariableQuality")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("EncodeJpegVariableQuality")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::AI_CPU);
}  // namespace domi