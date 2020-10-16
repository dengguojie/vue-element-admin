/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
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

static const char* const kBoxesUnpack = "/unstack";
static const char* const kBoxesDiv = "RealDiv";

Status ParseValueFromConstV2(const vector<const NodeDef *> &v_input_const, const string &names, float &value) {
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

Status DecodeBboxV2Params(const std::vector<const google::protobuf::Message *> insideNodes, ge::Operator &op) {
    auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op);
        if (opDesc == nullptr) {
            OP_LOGE(op.GetName().c_str(), "Get op desc failed.");
            return FAILED;
        }
        vector<const NodeDef *> v_input_const;
        std::string div_node_name;
        vector<string> v_const_str;
        for (auto node : insideNodes) {
                const NodeDef* node_def = reinterpret_cast<const NodeDef *>(node);
                if (node_def == nullptr) {
                        OP_LOGE(op.GetName().c_str(), "Node_def is nullptr.");
                        return FAILED;
                }
                v_input_const.push_back(node_def);
                OP_LOGD(op.GetName().c_str(), "DecodeBoxV2 NodeDef is %s ", node_def->name().c_str());
                if (node_def->op() == kBoxesDiv) {
                        if (node_def->input(0).find(kBoxesUnpack) != string::npos) {
                                div_node_name = node_def->input(1);
                                v_const_str.push_back(div_node_name);
                        }
                }
        }
        float scale = 1.0;
        float scale_1 = 1.0;
        float scale_2 = 1.0;
        float scale_3 = 1.0;
        if (v_const_str.size() != 4) {
                OP_LOGI(op.GetName().c_str(), "Boxes don't need to scale.");
        }
        else {
                if (ParseValueFromConstV2(v_input_const, v_const_str[0], scale) != SUCCESS) {
                        OP_LOGE(op.GetName().c_str(), "Convert div const failed");
                        return PARAM_INVALID;
                }
                if (ParseValueFromConstV2(v_input_const, v_const_str[1], scale_1) != SUCCESS) {
                        OP_LOGE(op.GetName().c_str(), "Convert div_1 const failed");
                        return PARAM_INVALID;
                }
                if (ParseValueFromConstV2(v_input_const, v_const_str[2], scale_2) != SUCCESS) {
                        OP_LOGE(op.GetName().c_str(), "Convert div_2 const failed");
                        return PARAM_INVALID;
                }
                if (ParseValueFromConstV2(v_input_const, v_const_str[3], scale_3) != SUCCESS) {
                        OP_LOGE(op.GetName().c_str(), "Convert div_3 const failed");
                        return PARAM_INVALID;
                }
        }
        std::vector<float> scales_list;
        scales_list.push_back(scale);
        scales_list.push_back(scale_1);
        scales_list.push_back(scale_2);
        scales_list.push_back(scale_3);
        if (!ge::AttrUtils::SetListFloat(opDesc, "scales", scales_list)) {
        OP_LOGE(op.GetName().c_str(), "Set scales_list failed.");
        return FAILED;
        }
        OP_LOGD(op.GetName().c_str(), "DecodeBboxV2's attr scales is [%.1f, %.1f, %.1f, %.1f]", scales_list[0],
                scales_list[1], scales_list[2], scales_list[3]);

    return SUCCESS;
}

REGISTER_CUSTOM_OP("DecodeBboxV2")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DecodeBboxV2")
    .FusionParseParamsFn(DecodeBboxV2Params)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
