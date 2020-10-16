/* Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use
 * this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"

namespace domi {
Status LayerNormGradParserParams(const std::vector<const google::protobuf::Message *> inside_nodes, ge::Operator &op) {
  OP_LOGI(op.GetName().c_str(), "Enter layer grad norm fusion parser.");
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr) {
    OP_LOGE(op.GetName().c_str(), "Get op desc failed.");
    return FAILED;
  }

  return SUCCESS;
}


REGISTER_CUSTOM_OP("LayerNormGrad")
    .FrameworkType(TENSORFLOW)
    .FusionParseParamsFn(LayerNormGradParserParams)
    .OriginOpType("LayerNormGrad")
    .ImplyType(ImplyType::TVM);
}  // namespace domi
