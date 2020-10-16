/**
 * Copyright 2020 Huawei Technologies Co., Ltd
*/

#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "operator.h"
#include "common/util/error_manager/error_manager.h"

using namespace ge;
namespace domi {

const int INPUT_FILTER = 0;

// Replace ge ParseParams fuction to process graph maxpool3dgrad node attrs
Status ParseParamsMaxPool3DGRAD(const Message* op_src, ge::Operator& op) {

    // Convert original tf graph maxpool3dgrad attrs to GE graph attrs
    AutoMappingFn(op_src, op);

    // Escape GE require attr [pads] check here
    std::vector<int32_t> padList = {0,0,0,0,0,0};
    op.SetAttr("pads", padList);

    return SUCCESS;
}

REGISTER_CUSTOM_OP("MaxPool3DGrad")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("MaxPool3DGrad")
    .ParseParamsFn(ParseParamsMaxPool3DGRAD)
    .ImplyType(ImplyType::TVM);
}  // namespace domi

