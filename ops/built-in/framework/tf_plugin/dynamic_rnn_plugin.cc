// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
// This program is free software; you can redistribute it and/or modify
// it under the terms of the Apache License Version 2.0.You may not use
// this file except in compliance with the License.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// Apache License for more details at
// http:// www.apache.org/licenses/LICENSE-2.0

#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "proto/tensorflow/node_def.pb.h"
#include "graph/operator.h"

#include "tensorflow_fusion_op_parser_util.h"
#include "op_log.h"

using domi::tensorflow::NodeDef;
using domi::tensorflow::TensorProto;
using google::protobuf::Message;
using std::vector;

namespace domi {
uint32_t wRnnInputPosition = 1;
static const char* const kForgetBias = "lstm_cell/add/y";
static const char* const kTransposeNode = "Transpose";

Status DynamicRNNParserParams(const std::vector<const google::protobuf::Message*> inside_nodes, ge::Operator& op) {
  OP_LOGI(op.GetName().c_str(), "Enter DynamicRNN fusion parser.");
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr) {
    OP_LOGE(op.GetName().c_str(), "Get op desc failed.");
    return FAILED;
  }

  float forget_bias = 0.0;
  bool time_major = true;
  for (auto node : inside_nodes) {
    const NodeDef* node_def = reinterpret_cast<const NodeDef*>(node);
    if (node_def == nullptr) {
      OP_LOGE(op.GetName().c_str(), "Node_def is nullptr.");
      return FAILED;
    }
    OP_LOGD(op.GetName().c_str(), "DynamicRNN NodeDef is %s ", node_def->name().c_str());
    if (node_def->op() == kTransposeNode) {
      time_major = false;
    }
    if (node_def->name().find(kForgetBias) != string::npos) {
      if (ParseParamFromConst(node_def, forget_bias) != SUCCESS) {
        OP_LOGE("ParseParamFromConst data from const NodeDef %s failed", nodeDef->name().c_str());
        return PARAM_INVALID;
      }
    }
  }
  op.SetAttr("time_major", time_major);
  op.SetAttr("forget_bias", forget_bias);
  OP_LOGD(op.GetName().c_str(), "parser stage set DynamicRNN's attr time_major is %s forget_bias is %.1f",
          time_major ? "true" : "false", forget_bias);

  ge::GeTensorDesc input_desc = op_desc->GetInputDesc(wRnnInputPosition);
  input_desc.SetOriginFormat(ge::FORMAT_HWCN);
  input_desc.SetFormat(ge::FORMAT_HWCN);

  if (op_desc->UpdateInputDesc(wRnnInputPosition, input_desc) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update input desc fail, index:%u.", wRnnInputPosition);
    return FAILED;
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("DynamicRNN")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DynamicRNN")
    .ParseParamsFn(AutoMappingFn)
    .FusionParseParamsFn(DynamicRNNParserParams)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
