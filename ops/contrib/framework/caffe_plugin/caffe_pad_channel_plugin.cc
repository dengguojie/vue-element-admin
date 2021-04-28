/**
Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
Description: plugin for padchannel caffe operator
Author:
Create: 2020-6-15
*/

#include <memory>
#include <string>
#include <vector>
#include "register/register.h"

namespace domi {
Status CaffePadChannelParseParams(const ge::Operator& op_src, ge::Operator& op_dest) {
  int numChannelsToPad = 0;
  if (ge::GRAPH_SUCCESS == op_src.GetAttr("num_channels_to_pad", numChannelsToPad)) {
    op_dest.SetAttr("num_channels_to_pad", numChannelsToPad);
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("PadChannel")
    .FrameworkType(CAFFE)
    .OriginOpType("PadChannel")
    .ParseParamsByOperatorFn(CaffePadChannelParseParams)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
