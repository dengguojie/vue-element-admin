/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file instance_norm_plugin.cc
 * \brief
 */
#include "register/register.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "proto/tensorflow/node_def.pb.h"

#include "tensorflow_fusion_op_parser_util.h"
#include "op_log.h"

namespace domi {
Status InstanceNormParserParams(const std::vector<const google::protobuf::Message*> inside_nodes, ge::Operator& op) {
  OP_LOGI(op.GetName().c_str(), "Enter instance norm fusion parser.");
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr) {
    OP_LOGE(op.GetName().c_str(), "Get op desc failed.");
    return FAILED;
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("InstanceNorm")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("InstanceNorm")
    .FusionParseParamsFn(InstanceNormParserParams)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
