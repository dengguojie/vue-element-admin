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

static const char* const kDecodeClip = "decode_clip";
static const char* const kMinium = "Minimum";

Status ParseValueFromConst(const vector<const NodeDef *> &v_input_const, const string &names, float &value) {
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

Status DecodeBboxParams(const std::vector<const google::protobuf::Message *> insideNodes, ge::Operator &op) {
    auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op);
	if (opDesc == nullptr) {
		OP_LOGE(op.GetName().c_str(), "Get op desc failed.");
		return FAILED;
    }

	vector<const NodeDef *> v_input_const;
	std::string inputy_node_name;
	for (auto node : insideNodes) {
		const NodeDef* node_def = reinterpret_cast<const NodeDef *>(node);
		if (node_def == nullptr) {
			OP_LOGE(op.GetName().c_str(), "Node_def is nullptr.");
			return FAILED;
		}
		
		v_input_const.push_back(node_def);
		if (node_def->op() == kMinium) {
			inputy_node_name = node_def->input(1);
		}
	}
	if (inputy_node_name.empty()) {
		OP_LOGE(op.GetName().c_str(), "cann't find Minimum op node in decode_bbox");
		return FAILED;
	}
	float param = 0.0;
	if (ParseValueFromConst(v_input_const, inputy_node_name, param) != SUCCESS) {
		OP_LOGE(op.GetName().c_str(), "Convert begin_norm_axis data failed");
		return PARAM_INVALID;
	}
	if (!ge::AttrUtils::SetFloat(opDesc, kDecodeClip, param)) {
        OP_LOGE(op.GetName().c_str(), "Set spatial_scale failed.");
        return FAILED;
	}
	OP_LOGI(op.GetName().c_str(), "kDecodeClip%.1f", param);	
    return SUCCESS;
}

REGISTER_CUSTOM_OP("DecodeBbox")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DecodeBbox")
    .FusionParseParamsFn(DecodeBboxParams)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
