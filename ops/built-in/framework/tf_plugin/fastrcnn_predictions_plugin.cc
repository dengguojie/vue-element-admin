/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
 * \file fastrcnn_predictions_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "tensorflow_fusion_op_parser_util.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "proto/tensorflow/node_def.pb.h"

#include "op_log.h"

using domi::tensorflow::NodeDef;
using domi::tensorflow::TensorProto;
using google::protobuf::Message;
using std::vector;

namespace domi {
static const char * const nms = "NonMaxSuppressionV2";
static const char * const greater = "Greater";
static const char * const minimum = "Minimum";

Status ParseFloatValueFromConst(const vector<const NodeDef*>& v_input_const, const string& names, float& value) {
  for (auto nodeDef : v_input_const) {
    string name = nodeDef->name();
    if (name == names) {
      if (ParseParamFromConst(nodeDef, value) != SUCCESS) {
        OP_LOGE("FastrcnnPredictions", "ParseParamFromConst data from const NodeDef %s failed", name.c_str());
        return PARAM_INVALID;
      }
      return SUCCESS;
    }
  }
  return FAILED;
}

Status ParseIntValueFromConst(const vector<const NodeDef*>& v_input_const, const string& names, int& value) {
  for (auto nodeDef : v_input_const) {
    string name = nodeDef->name();
    if (name == names) {
      if (ParseParamFromConst(nodeDef, value) != SUCCESS) {
        OP_LOGE("FastrcnnPredictions", "ParseParamFromConst data from const NodeDef %s failed", name.c_str());
        return PARAM_INVALID;
      }
      return SUCCESS;
    }
  }
  return FAILED;
}

Status FastrcnnPredictionsParams(const std::vector<const google::protobuf::Message*> insideNodes, ge::Operator& op) {
  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (opDesc == nullptr) {
    OP_LOGE(TbeGetName(op).c_str(), "Get op desc failed.");
    return FAILED;
  }

  vector<const NodeDef*> v_input_const;
  std::string input_attr_name;
  float attr_nms_threshold = 0.0;
  float attr_score_threshold = 0.0;
  int attr_k = 0;

  for (auto node : insideNodes) {
    const NodeDef* node_def = reinterpret_cast<const NodeDef*>(node);
    if (node_def == nullptr) {
      OP_LOGE(TbeGetName(op).c_str(), "Node_def is nullptr.");
      return FAILED;
    }
    v_input_const.push_back(node_def);
    if (node_def->op() == nms) {
      static const size_t input_attr_name_index = 3;
      input_attr_name = node_def->input(input_attr_name_index);
      OP_LOGI(TbeGetName(op).c_str(), "get node name %s .", input_attr_name.c_str());
    }
  }
  if (input_attr_name.empty()) {
    OP_LOGE(TbeGetName(op).c_str(), "cann't find non_max_suppression op node in FastrcnnPredictions");
    return FAILED;
  }

  if (ParseFloatValueFromConst(v_input_const, input_attr_name, attr_nms_threshold) != SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Convert begin_norm_axis data failed");
    return PARAM_INVALID;
  }

  for (auto node : insideNodes) {
    const NodeDef* node_def = reinterpret_cast<const NodeDef*>(node);
    if (node_def == nullptr) {
      OP_LOGE(TbeGetName(op).c_str(), "Node_def is nullptr.");
      return FAILED;
    }
    v_input_const.push_back(node_def);
    if (node_def->op() == greater) {
      input_attr_name = node_def->input(1);
      OP_LOGI(TbeGetName(op).c_str(), "get node name %s .", input_attr_name.c_str());
    }
  }
  if (input_attr_name.empty()) {
    OP_LOGE(TbeGetName(op).c_str(), "cann't find Greater op node in FastrcnnPredictions");
    return FAILED;
  }

  if (ParseFloatValueFromConst(v_input_const, input_attr_name, attr_score_threshold) != SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Convert begin_norm_axis data failed");
    return PARAM_INVALID;
  }

  for (auto node : insideNodes) {
    const NodeDef* node_def = reinterpret_cast<const NodeDef*>(node);
    if (node_def == nullptr) {
      OP_LOGE(TbeGetName(op).c_str(), "Node_def is nullptr.");
      return FAILED;
    }
    v_input_const.push_back(node_def);
    if (node_def->op() == minimum) {
      input_attr_name = node_def->input(0);
      OP_LOGI(TbeGetName(op).c_str(), "get node name %s .", input_attr_name.c_str());
    }
  }
  if (input_attr_name.empty()) {
    OP_LOGE(TbeGetName(op).c_str(), "cann't find Minimum op node in FastrcnnPredictions");
    return FAILED;
  }

  if (ParseIntValueFromConst(v_input_const, input_attr_name, attr_k) != SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Convert attr_k data failed");
    return PARAM_INVALID;
  }

  if (!ge::AttrUtils::SetFloat(opDesc, "nms_threshold", attr_nms_threshold)) {
    OP_LOGE(TbeGetName(op).c_str(), "Set attr nms_threshold failed.");
    return FAILED;
  }
  OP_LOGI(TbeGetName(op).c_str(), "Set attr nms_threshold %1.2f.", attr_nms_threshold);

  if (!ge::AttrUtils::SetFloat(opDesc, "score_threshold", attr_score_threshold)) {
    OP_LOGE(TbeGetName(op).c_str(), "Set attr score_threshold failed.");
    return FAILED;
  }
  OP_LOGI(TbeGetName(op).c_str(), "Set attr score_threshold %1.2f.", attr_score_threshold);

  if (!ge::AttrUtils::SetInt(opDesc, "k", attr_k)) {
    OP_LOGE(TbeGetName(op).c_str(), "Set attr k failed.");
    return FAILED;
  }
  OP_LOGI(TbeGetName(op).c_str(), "Set attr k %d.", attr_k);

  return SUCCESS;
}

REGISTER_CUSTOM_OP("FastrcnnPredictions")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FastrcnnPredictions")
    .FusionParseParamsFn(FastrcnnPredictionsParams)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
