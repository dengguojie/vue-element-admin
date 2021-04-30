/**
Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
Description: op_proto for interp caffe operator
Author:
Create: 2020-6-11
*/
#include <string>
#include <vector>
#include "register/register.h"

using namespace ge;
namespace domi {

// register PostProcess op info to GE
REGISTER_CUSTOM_OP("Yolov2PostProcess")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("Yolov2PostProcess")
    .ParseParamsFn(AutoMappingFn);
}  // namespace domi
