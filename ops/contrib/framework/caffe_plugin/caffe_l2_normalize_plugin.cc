/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: l2_normalize framework plugin cpp file
 * Author:
 * Create: 2020-6-11
 * Note:
 */

#include "register/register.h"
using namespace ge;

namespace domi {

Status CaffeL2NormalizeParseParams(const Operator& op_src, ge::Operator& op_dest) {
  float eps = 0;
  if (ge::GRAPH_SUCCESS == op_src.GetAttr("eps", eps)) {
    op_dest.SetAttr("eps", eps);
  }
  return SUCCESS;
}

REGISTER_CUSTOM_OP("L2Normalize")
    .FrameworkType(CAFFE)
    .OriginOpType("L2Normalize")
    .ParseParamsByOperatorFn(CaffeL2NormalizeParseParams)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
