/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2018. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file layer_norm_grad_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"

#include "op_log.h"

namespace domi {
Status LayerNormGradParserParams(const std::vector<const google::protobuf::Message*> inside_nodes, ge::Operator& op) {
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
