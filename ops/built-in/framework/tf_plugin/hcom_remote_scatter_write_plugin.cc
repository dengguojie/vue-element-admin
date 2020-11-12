#include "register/register.h"

using namespace ge;
namespace domi{
REGISTER_CUSTOM_OP("HcomRemoteScatterWrite")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("HcomRemoteScatterWrite")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::HCCL);
}