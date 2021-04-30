/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: l2_normalize framework plugin cpp file
 * Author:
 * Create: 2020-8-11
 * Note:
 */

#include "register/register.h"
#include <vector>

namespace domi {
Status ParseNormalize(const ge::Operator& op_src, ge::Operator& op_dest)
{
    float eps = 0;
    if (ge::GRAPH_SUCCESS == op_src.GetAttr("eps", eps)) {
        op_dest.SetAttr("eps", eps);
    }

    // Caffe NCHW Format, reduce at CHW axes.
    std::vector<int64_t> axis = { 1, 2, 3 };
    op_dest.SetAttr("axis", axis);

    return SUCCESS;
}

// adapted Normalize op to L2Normalize
REGISTER_CUSTOM_OP("L2Normalize")
    .FrameworkType(CAFFE)
    .OriginOpType("Normalize")
    .ParseParamsByOperatorFn(ParseNormalize)
    .ImplyType(ImplyType::TVM);
}  // namespace domi

