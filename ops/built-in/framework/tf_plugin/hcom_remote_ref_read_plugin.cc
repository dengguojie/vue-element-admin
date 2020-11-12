#include "register/register.h"

using namespace ge;
namespace domi{
REGISTER_CUSTOM_OP("HcomRemoteRefRead")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("HcomRemoteRefRead")
    .ParseParamsFn(AutoMappingFn)
    .ImplyType(ImplyType::HCCL);
}