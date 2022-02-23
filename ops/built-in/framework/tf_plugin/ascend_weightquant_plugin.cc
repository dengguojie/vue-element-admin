/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
 * \file ascend_weightquant_plugin.cpp
 * \brief
 */
#include <string>

#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "proto/tensorflow/node_def.pb.h"
#include "register/register.h"
#include "tensorflow_fusion_op_parser_util.h"

using domi::tensorflow::NodeDef;
using domi::tensorflow::TensorProto;
using google::protobuf::Message;

namespace domi {

Status ParseParamsAscendWeightQuant(const google::protobuf::Message* op_src, ge::Operator& op) {
  AutoMappingFn(op_src, op);

  const NodeDef* node_def = reinterpret_cast<const NodeDef*>(op_src);
  if (node_def == nullptr) {
      OP_LOGE(op.GetName().c_str(), "Node_def is nullptr.");
      return FAILED;
  }

  auto attr_item = node_def->attr().find("dst_type");
  if (attr_item != node_def->attr().end()) {
      auto attr_value = attr_item->second;
      int dst_type = ge::DT_INT8;
      if (attr_value.s() == "INT4") {
          dst_type = ge::DT_INT4;
      }
      op.SetAttr("dst_type", dst_type);
  }

  OP_LOGI("AscendWeightQuant", "op[AscendWeightQuant] tensowflow plugin parser [AutoMapping] success.");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("AscendWeightQuant")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AscendWeightQuant")
    .ParseParamsFn(ParseParamsAscendWeightQuant)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
