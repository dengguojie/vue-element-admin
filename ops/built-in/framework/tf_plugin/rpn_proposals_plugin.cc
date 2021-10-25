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
 * \file rpn_proposals_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "proto/tensorflow/node_def.pb.h"

#include "tensorflow_fusion_op_parser_util.h"
#include "op_log.h"

using domi::tensorflow::NodeDef;
using domi::tensorflow::TensorProto;
using google::protobuf::Message;
using std::vector;

namespace domi {
static const char* const kScoreTthreshold = "score_threshold";
static const char* const kK = "k";
static const char* const kMinSize = "min_size";
static const char* const kNmsTthreshold = "nms_threshold";
static const char* const kPostNmsNum = "post_nms_num";

static const char* const kScoreTthresholdOp = "Greater";            // 1
static const char* const kKOp = "Minimum";                          // 0
static const char* const kMinSizeOp = "Greater";                    // 1
static const char* const kNmsTthresholdOp = "NonMaxSuppressionV2";  // 3
static const char* const kPostNmsNumOp = "NonMaxSuppressionV2";     // 2

Status ParseValueFromConstFloat(const vector<const NodeDef*>& v_input_const, const string& names, float& value) {
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

Status ParseValueFromConstInt(const vector<const NodeDef*>& v_input_const, const string& names, int& value) {
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

Status RpnProposalsParams(const std::vector<const google::protobuf::Message*> insideNodes, ge::Operator& op) {
  OP_LOGI(op.GetName().c_str(), "Enter RpnProposals fusion parser.");

  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op);
  if (opDesc == nullptr) {
    OP_LOGE(op.GetName().c_str(), "Get op desc failed.");
    return FAILED;
  }

  vector<const NodeDef*> v_input_const;
  std::string inputy_node_name;
  int attr_num = 0;

  for (auto node : insideNodes) {
    const NodeDef* node_def = reinterpret_cast<const NodeDef*>(node);
    if (node_def == nullptr) {
      OP_LOGE(op.GetName().c_str(), "Node_def is nullptr.");
      return FAILED;
    }

    v_input_const.push_back(node_def);

    // score_threshold
    if (node_def->op() == kScoreTthresholdOp) {
      inputy_node_name = node_def->input(1);

      if (node_def->name() == "task_0/generate_rpn_proposals/filtered_boxes" ||
          node_def->name() == "task_1/generate_rpn_proposals/filtered_boxes" ||
          node_def->name() == "task_2/generate_rpn_proposals/filtered_boxes" ||
          node_def->name() == "task_3/generate_rpn_proposals/filtered_boxes" ||
          node_def->name() == "task_4/generate_rpn_proposals/filtered_boxes" ||
          node_def->name() == "task_5/generate_rpn_proposals/filtered_boxes" ||
          node_def->name() == "task_6/generate_rpn_proposals/filtered_boxes" ||
          node_def->name() == "task_7/generate_rpn_proposals/filtered_boxes" ||
          node_def->name() == "task_8/generate_rpn_proposals/filtered_boxes") {
        float param = 0.0;
        if (ParseValueFromConstFloat(v_input_const, inputy_node_name, param) != SUCCESS) {
          OP_LOGE(op.GetName().c_str(), "Convert score_threshold data failed");
          return PARAM_INVALID;
        }

        if (!ge::AttrUtils::SetFloat(opDesc, kScoreTthreshold, param)) {
          OP_LOGE(op.GetName().c_str(), "Set score_threshold failed.");
          return FAILED;
        }
        attr_num = attr_num + 1;
        OP_LOGI(op.GetName().c_str(), "Set score_threshold SUCCESS!!!!");
      }
    }

    // k
    if (node_def->op() == kKOp) {
      inputy_node_name = node_def->input(0);

      if (node_def->name() == "task_0/generate_rpn_proposals/Minimum" ||
          node_def->name() == "task_1/generate_rpn_proposals/Minimum" ||
          node_def->name() == "task_2/generate_rpn_proposals/Minimum" ||
          node_def->name() == "task_3/generate_rpn_proposals/Minimum" ||
          node_def->name() == "task_4/generate_rpn_proposals/Minimum" ||
          node_def->name() == "task_5/generate_rpn_proposals/Minimum" ||
          node_def->name() == "task_6/generate_rpn_proposals/Minimum" ||
          node_def->name() == "task_7/generate_rpn_proposals/Minimum" ||
          node_def->name() == "task_8/generate_rpn_proposals/Minimum") {
        int param = 0;
        if (ParseValueFromConstInt(v_input_const, inputy_node_name, param) != SUCCESS) {
          OP_LOGE(op.GetName().c_str(), "Convert topk k data failed");
          return PARAM_INVALID;
        }

        if (!ge::AttrUtils::SetInt(opDesc, kK, param)) {
          OP_LOGE(op.GetName().c_str(), "Set topk k failed.");
          return FAILED;
        }
        attr_num = attr_num + 1;
        OP_LOGI(op.GetName().c_str(), "Set topk k SUCCESS!!!!");
      }
    }

    // min_size
    if (node_def->op() == kMinSizeOp) {
      inputy_node_name = node_def->input(1);

      if (node_def->name() == "task_0/generate_rpn_proposals/Greater" ||
          node_def->name() == "task_1/generate_rpn_proposals/Greater" ||
          node_def->name() == "task_2/generate_rpn_proposals/Greater" ||
          node_def->name() == "task_3/generate_rpn_proposals/Greater" ||
          node_def->name() == "task_4/generate_rpn_proposals/Greater" ||
          node_def->name() == "task_5/generate_rpn_proposals/Greater" ||
          node_def->name() == "task_6/generate_rpn_proposals/Greater" ||
          node_def->name() == "task_7/generate_rpn_proposals/Greater" ||
          node_def->name() == "task_8/generate_rpn_proposals/Greater") {
        float param = 0.0;
        if (ParseValueFromConstFloat(v_input_const, inputy_node_name, param) != SUCCESS) {
          OP_LOGE(op.GetName().c_str(), "Convert min_size data failed");
          return PARAM_INVALID;
        }

        if (!ge::AttrUtils::SetFloat(opDesc, kMinSize, param)) {
          OP_LOGE(op.GetName().c_str(), "Set min_size failed.");
          return FAILED;
        }
        attr_num = attr_num + 1;
        OP_LOGI(op.GetName().c_str(), "Set min_size SUCCESS!!!!");
      }
    }

    // nms_threshold
    if (node_def->op() == kNmsTthresholdOp) {
      inputy_node_name = node_def->input(3);

      float param = 0.0;
      if (ParseValueFromConstFloat(v_input_const, inputy_node_name, param) != SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Convert nms_threshold data failed");
        return PARAM_INVALID;
      }

      if (!ge::AttrUtils::SetFloat(opDesc, kNmsTthreshold, param)) {
        OP_LOGE(op.GetName().c_str(), "Set nms_threshold failed.");
        return FAILED;
      }
      attr_num = attr_num + 1;
      OP_LOGI(op.GetName().c_str(), "Set nms_threshold SUCCESS!!!");
    }

    //  post_nms_num
    if (node_def->op() == kPostNmsNumOp) {
      inputy_node_name = node_def->input(2);

      int param = 0;
      if (ParseValueFromConstInt(v_input_const, inputy_node_name, param) != SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Convert post_nms_num data failed");
        return PARAM_INVALID;
      }

      if (!ge::AttrUtils::SetInt(opDesc, kPostNmsNum, param)) {
        OP_LOGE(op.GetName().c_str(), "Set post_nms_num failed.");
        return FAILED;
      }
      attr_num = attr_num + 1;
      OP_LOGI(op.GetName().c_str(), "Set post_nms_num SUCCESS!!!");
    }
  }

  if (inputy_node_name.empty()) {
    OP_LOGE(op.GetName().c_str(), "cann't find specific op node in rpn_proposals");
    return FAILED;
  }

  if (attr_num != 5) {
    OP_LOGE(op.GetName().c_str(), "cann't find right num of attr node in rpn_proposals!");
    return FAILED;
  }

  OP_LOGI(op.GetName().c_str(), "Obtain attributes for rpn_proposals SUCCESS!");
  return SUCCESS;
}

REGISTER_CUSTOM_OP("RpnProposals")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("RpnProposals")
    .FusionParseParamsFn(RpnProposalsParams)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
