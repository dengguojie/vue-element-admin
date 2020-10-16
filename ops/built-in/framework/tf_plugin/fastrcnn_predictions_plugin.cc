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

static const char* const nms = "NonMaxSuppressionV2";
static const char* const greater = "Greater";
static const char* const minimum = "Minimum";

Status ParseFloatValueFromConst(const vector<const NodeDef *> &v_input_const, const string &names, float &value) {
  for (auto nodeDef : v_input_const) {
    string name = nodeDef->name();
    if (name == names) {
      if (ParseParamFromConst(nodeDef, value) != SUCCESS) {
        OP_LOGE("ParseParamFromConst data from const NodeDef %s failed", name.c_str());
        return PARAM_INVALID;
      }
      return SUCCESS;
    }
  }
  return FAILED;
}

Status ParseIntValueFromConst(const vector<const NodeDef *> &v_input_const, const string &names, int &value) {
  for (auto nodeDef : v_input_const) {
    string name = nodeDef->name();
    if (name == names) {
      if (ParseParamFromConst(nodeDef, value) != SUCCESS) {
        OP_LOGE("ParseParamFromConst data from const NodeDef %s failed", name.c_str());
        return PARAM_INVALID;
      }
      return SUCCESS;
    }
  }
  return FAILED;
}

Status FastrcnnPredictionsParams(const std::vector<const google::protobuf::Message *> insideNodes, ge::Operator &op) {
    auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op);

	if (opDesc == nullptr) {
		OP_LOGE(op.GetName().c_str(), "Get op desc failed.");
		return FAILED;
    }

	vector<const NodeDef *> v_input_const;
	std::string input_attr_name;
	float attr_nms_threshold = 0.0;
	float attr_score_threshold = 0.0;
	int attr_k = 0;
	
	for (auto node : insideNodes) {
		const NodeDef* node_def = reinterpret_cast<const NodeDef *>(node);
		if (node_def == nullptr) {
			OP_LOGE(op.GetName().c_str(), "Node_def is nullptr.");
			return FAILED;
		}
		//OP_LOGI(op.GetName().c_str(), "node name %s .", node_def->op().c_str());
		v_input_const.push_back(node_def);
		if (node_def->op() == nms) {
			input_attr_name = node_def->input(3);
			OP_LOGI(op.GetName().c_str(), "get node name %s .", input_attr_name.c_str());
		}
	}
	if (input_attr_name.empty()) {
		OP_LOGE(op.GetName().c_str(), "cann't find non_max_suppression op node in FastrcnnPredictions");
		return FAILED;
	}
	
	if (ParseFloatValueFromConst(v_input_const, input_attr_name, attr_nms_threshold) != SUCCESS) {
		OP_LOGE(op.GetName().c_str(), "Convert begin_norm_axis data failed");
		return PARAM_INVALID;
	}

	for (auto node : insideNodes) {
		const NodeDef* node_def = reinterpret_cast<const NodeDef *>(node);
		if (node_def == nullptr) {
			OP_LOGE(op.GetName().c_str(), "Node_def is nullptr.");
			return FAILED;
		}
		v_input_const.push_back(node_def);
		if (node_def->op() == greater) {
			input_attr_name = node_def->input(1);
			OP_LOGI(op.GetName().c_str(), "get node name %s .", input_attr_name.c_str());
		}
	}
	if (input_attr_name.empty()) {
		OP_LOGE(op.GetName().c_str(), "cann't find Greater op node in FastrcnnPredictions");
		return FAILED;
	}
	
	if (ParseFloatValueFromConst(v_input_const, input_attr_name, attr_score_threshold) != SUCCESS) {
		OP_LOGE(op.GetName().c_str(), "Convert begin_norm_axis data failed");
		return PARAM_INVALID;
	}

	for (auto node : insideNodes) {
		const NodeDef* node_def = reinterpret_cast<const NodeDef *>(node);
		if (node_def == nullptr) {
			OP_LOGE(op.GetName().c_str(), "Node_def is nullptr.");
			return FAILED;
		}
		v_input_const.push_back(node_def);
		if (node_def->op() == minimum) {
			input_attr_name = node_def->input(0);
			OP_LOGI(op.GetName().c_str(), "get node name %s .", input_attr_name.c_str());
		}
	}
	if (input_attr_name.empty()) {
		OP_LOGE(op.GetName().c_str(), "cann't find Minimum op node in FastrcnnPredictions");
		return FAILED;
	}
	
	if (ParseIntValueFromConst(v_input_const, input_attr_name, attr_k) != SUCCESS) {
		OP_LOGE(op.GetName().c_str(), "Convert attr_k data failed");
		return PARAM_INVALID;
	}

	if (!ge::AttrUtils::SetFloat(opDesc, "nms_threshold", attr_nms_threshold)) {
        OP_LOGE(op.GetName().c_str(), "Set attr nms_threshold failed.");
        return FAILED;
    }
	OP_LOGI(op.GetName().c_str(), "Set attr nms_threshold %1.2f.", attr_nms_threshold);

	if (!ge::AttrUtils::SetFloat(opDesc, "score_threshold", attr_score_threshold)) {
        OP_LOGE(op.GetName().c_str(), "Set attr score_threshold failed.");
        return FAILED;
    }
	OP_LOGI(op.GetName().c_str(), "Set attr score_threshold %1.2f.", attr_score_threshold);

	if (!ge::AttrUtils::SetInt(opDesc, "k", attr_k)) {
        OP_LOGE(op.GetName().c_str(), "Set attr k failed.");
        return FAILED;
    }
	OP_LOGI(op.GetName().c_str(), "Set attr k %d.", attr_k);


    return SUCCESS;
}

REGISTER_CUSTOM_OP("FastrcnnPredictions")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FastrcnnPredictions")
    .FusionParseParamsFn(FastrcnnPredictionsParams)
    .ImplyType(ImplyType::TVM);
}  // namespace domi

