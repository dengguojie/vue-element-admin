/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file ascend_quant_plugin.cpp
 * \brief
 */
#include "op_log.h"
#include "graph/types.h"
#include "register/register.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "proto/tensorflow/node_def.pb.h"
#include "tensorflow_fusion_op_parser_util.h"

using domi::tensorflow::NodeDef;

namespace domi {

Status AutoMappingFnQuant(const google::protobuf::Message* op_src, ge::Operator& op) {
  AutoMappingFn(op_src, op);
  const NodeDef* node_def = reinterpret_cast<const NodeDef*>(op_src);
  if (node_def == nullptr) {
     OP_LOGE(op.GetName().c_str(), "node_def is nullptr.");
     return FAILED;
  }
  int dst_type = ge::DT_INT8;
  auto it = node_def->attr().find("dst_type");
  if (it != node_def->attr().end()) {
    auto attr_val = it->second;
    if (attr_val.s() == "INT4") {
      dst_type = ge::DT_INT4;
    }
  }
  op.SetAttr("dst_type",dst_type);
  return SUCCESS;
}
REGISTER_CUSTOM_OP("AscendQuant")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("AscendQuant")
    .ParseParamsFn(AutoMappingFnQuant)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
