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
#include "tensorflow_fusion_op_parser_util.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "proto/tensorflow/node_def.pb.h"
#include "op_log.h"

using std::vector;
using google::protobuf::Message;
using domi::tensorflow::NodeDef;
using domi::tensorflow::TensorProto;

namespace domi {
static const int kMeanInputSize = 2;

Status ParseValueFromConst(const vector<const NodeDef *> &v_input_const, const string &names, int &value) {
  for (auto nodeDef : v_input_const) {
    string name = nodeDef->name();
    if (name == names) {
      if (ParseParamFromConst(nodeDef, value) != SUCCESS) {
        OP_LOGE("ParseParamFromConst data from const NodeDef %s failed", nodeDef->name().c_str());
        return PARAM_INVALID;
      }
      return SUCCESS;
    }
  }
  return FAILED;
}

Status LayerNormParserParams(const std::vector<const google::protobuf::Message *> inside_nodes, ge::Operator &op) {
  OP_LOGI(op.GetName().c_str(), "Enter layer norm fusion parser.");
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr) {
    OP_LOGE(op.GetName().c_str(), "Get op desc failed.");
    return FAILED;
  }

  vector<const NodeDef *> v_input_const;
  std::string mean_const_input_node_name;
  for (auto inside_node : inside_nodes) {
    const NodeDef* node_def = reinterpret_cast<const NodeDef *>(inside_node);
    if (node_def == nullptr) {
      OP_LOGE(op.GetName().c_str(), "Node_def is nullptr.");
      return FAILED;
    }
    v_input_const.push_back(node_def);

    if (node_def->op() == "Mean") {
      if (node_def->input().size() < kMeanInputSize) {
        OP_LOGE(op.GetName().c_str(), "Input size of node mean is invalid, which is %zu.", node_def->input().size());
        return FAILED;
      }
      mean_const_input_node_name = node_def->input(1);
    }
  }
  if (mean_const_input_node_name.empty()) {
    OP_LOGE(op.GetName().c_str(), "cann't find Mean op node in layernorm");
    return FAILED;
  }

  int begin_norm_axis = 1;
  int begin_params_axis = -1;

  if (ParseValueFromConst(v_input_const, mean_const_input_node_name, begin_norm_axis) != SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Convert begin_norm_axis data failed");
    return PARAM_INVALID;
  }

  OP_LOGI(op.GetName().c_str(), "begin_norm_axis = %d, begin_params_axis= %d", begin_norm_axis, begin_params_axis);

  if (!ge::AttrUtils::SetInt(op_desc, "begin_norm_axis", begin_norm_axis)) {
    OP_LOGE(op.GetName().c_str(), "Set begin_norm_axis failed");
    return PARAM_INVALID;
  }
  if (!ge::AttrUtils::SetInt(op_desc, "begin_params_axis", begin_params_axis)) {
    OP_LOGE(op.GetName().c_str(), "Set begin_params_axis failed");
    return PARAM_INVALID;
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("LayerNorm")
  .FrameworkType(TENSORFLOW)
  .OriginOpType("LayerNorm")
  .FusionParseParamsFn(LayerNormParserParams)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
