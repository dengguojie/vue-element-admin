/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2018. All rights reserved.
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
 * \file many_dynamic_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "register/auto_mapping_util.h"
#include "graph/utils/op_desc_utils.h"

#include "op_log.h"

namespace domi {
Status GetAttrValue(const domi::tensorflow::NodeDef* node, std::string& attr_name, int32_t& dynamic_tensor_num) {
  domi::tensorflow::AttrValue attr_num;
  if (!(ge::AutoMappingUtil::FindAttrValue(node, attr_name, attr_num))) {
    OP_LOGE("In NodeDef %s dynamic attr [%s] is not exist.", node->name().c_str(), attr_name.c_str());
    return FAILED;
  }

  if (attr_num.has_list()) {
    dynamic_tensor_num = attr_num.list().type_size();
  } else {
    dynamic_tensor_num = static_cast<int32_t>(attr_num.i());
  }
  if (dynamic_tensor_num <= 0) {
    OP_LOGW("In NodeDef %s dynamic num %d is less than 0.", node->name().c_str(), dynamic_tensor_num);
    // don't return
  }
  OP_LOGI("In NodeDef %s dynamic attr [%s] is  exist: %d.", node->name().c_str(), attr_name.c_str(),
          dynamic_tensor_num);
  return SUCCESS;
}

// register SparseSplit op to GE
Status SparseSplitMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  std::shared_ptr<ge::OpDesc> op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  Status ret = AutoMappingFn(op_src, op);
  if (ret != SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), "call auto mapping function failed.");
    return FAILED;
  }
  OP_LOGI("op[%s] call auto mapping function success.", op_desc->GetName().c_str());

  std::string attr_name = "num_split";
  std::vector<std::string> dynamic_output{"y_indices", "y_values", "y_shape"};
  const domi::tensorflow::NodeDef* node = reinterpret_cast<const domi::tensorflow::NodeDef*>(op_src);

  int32_t dynamic_tensor_num = 0;
  ret = GetAttrValue(node, attr_name, dynamic_tensor_num);
  if (ret != SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), attr_name.c_str(), "get attr [%s] value failed.");
    return FAILED;
  }

  bool is_pushback = true;
  for (std::string output_name : dynamic_output) {
    auto graph_ret = op_desc->AddDynamicOutputDesc(output_name, static_cast<uint32_t>(dynamic_tensor_num), is_pushback);
    if (graph_ret != ge::GRAPH_SUCCESS) {
      OP_LOGE(op_desc->GetName().c_str(), "AddDynamicOutputDesc failed.");
      return FAILED;
    }
    OP_LOGI("In NodeDef %s add dynamic output, output name = %s, size = %d", node->name().c_str(), output_name.c_str(),
            dynamic_tensor_num);
  }
  return SUCCESS;
}

REGISTER_CUSTOM_OP("SparseSplit")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseSplit")
    .ParseParamsFn(SparseSplitMapping)
    .ImplyType(ImplyType::AI_CPU);

// register BoostedTreesBucketize op to GE
Status BoostedTreesBucketizeMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  std::shared_ptr<ge::OpDesc> op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  Status ret = AutoMappingFn(op_src, op);
  if (ret != SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), "call auto mapping function failed.");
    return FAILED;
  }
  OP_LOGI("op[%s] call auto mapping function success.", op_desc->GetName().c_str());

  std::string attr_name = "num_features";
  std::vector<std::string> dynamic_input{"float_values", "bucket_boundaries"};
  std::string dynamic_output = "y";
  const domi::tensorflow::NodeDef* node = reinterpret_cast<const domi::tensorflow::NodeDef*>(op_src);

  int32_t dynamic_tensor_num = 0;
  ret = GetAttrValue(node, attr_name, dynamic_tensor_num);
  if (ret != SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), attr_name.c_str(), "get attr [%s] value failed.");
    return FAILED;
  }

  bool is_pushback = true;
  for (std::string input_name : dynamic_input) {
    auto graph_ret = op_desc->AddDynamicInputDesc(input_name, static_cast<uint32_t>(dynamic_tensor_num), is_pushback);
    if (graph_ret != ge::GRAPH_SUCCESS) {
      OP_LOGE(op_desc->GetName().c_str(), "AddDynamicInputDesc failed.");
      return FAILED;
    }
    OP_LOGI("In NodeDef %s add dynamic input, input name = %s, size = %d", node->name().c_str(), input_name.c_str(),
            dynamic_tensor_num);
  }
  auto graph_ret =
      op_desc->AddDynamicOutputDesc(dynamic_output, static_cast<uint32_t>(dynamic_tensor_num), is_pushback);
  if (graph_ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), "AddDynamicOutputDesc failed.");
    return FAILED;
  }
  OP_LOGI("In NodeDef %s add dynamic output, output name = %s, size = %d", node->name().c_str(), dynamic_output.c_str(),
          dynamic_tensor_num);

  return SUCCESS;
}

REGISTER_CUSTOM_OP("BoostedTreesBucketize")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BoostedTreesBucketize")
    .ParseParamsFn(BoostedTreesBucketizeMapping)
    .ImplyType(ImplyType::AI_CPU);

// register DynamicStitch op to GE
Status DynamicStitchMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  std::shared_ptr<ge::OpDesc> op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  Status ret = AutoMappingFn(op_src, op);
  if (ret != SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), "call auto mapping function failed.");
    return FAILED;
  }
  OP_LOGI("op[%s] call auto mapping function success.", op_desc->GetName().c_str());

  std::string attr_name = "N";
  std::vector<std::string> dynamic_input{"indices", "x"};

  const domi::tensorflow::NodeDef* node = reinterpret_cast<const domi::tensorflow::NodeDef*>(op_src);

  int32_t dynamic_tensor_num = 0;
  ret = GetAttrValue(node, attr_name, dynamic_tensor_num);
  if (ret != SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), attr_name.c_str(), "get attr [%s] value failed.");
    return FAILED;
  }

  bool is_pushback = true;
  for (std::string input_name : dynamic_input) {
    auto graph_ret = op_desc->AddDynamicInputDesc(input_name, static_cast<uint32_t>(dynamic_tensor_num), is_pushback);
    if (graph_ret != ge::GRAPH_SUCCESS) {
      OP_LOGE(op_desc->GetName().c_str(), "AddDynamicInputDesc failed.");
      return FAILED;
    }
    OP_LOGI("In NodeDef %s add dynamic input, input name = %s, size = %d", node->name().c_str(), input_name.c_str(),
            dynamic_tensor_num);
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("DynamicStitch")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DynamicStitch")
    .ParseParamsFn(DynamicStitchMapping)
    .ImplyType(ImplyType::AI_CPU);

// register ParallelDynamicStitch op to GE
Status ParallelDynamicStitchMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  std::shared_ptr<ge::OpDesc> op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  Status ret = AutoMappingFn(op_src, op);
  if (ret != SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), "call auto mapping function failed.");
    return FAILED;
  }
  OP_LOGI("op[%s] call auto mapping function success.", op_desc->GetName().c_str());

  std::string attr_name = "N";
  std::vector<std::string> dynamic_input{"indices", "x"};
  const domi::tensorflow::NodeDef* node = reinterpret_cast<const domi::tensorflow::NodeDef*>(op_src);

  int32_t dynamic_tensor_num = 0;
  ret = GetAttrValue(node, attr_name, dynamic_tensor_num);
  if (ret != SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), attr_name.c_str(), "get attr [%s] value failed.");
    return FAILED;
  }

  bool is_pushback = true;
  for (std::string input_name : dynamic_input) {
    auto graph_ret = op_desc->AddDynamicInputDesc(input_name, static_cast<uint32_t>(dynamic_tensor_num), is_pushback);
    if (graph_ret != ge::GRAPH_SUCCESS) {
      OP_LOGE(op_desc->GetName().c_str(), "AddDynamicInputDesc failed.");
      return FAILED;
    }
    OP_LOGI("In NodeDef %s add dynamic input, input name = %s, size = %d", node->name().c_str(), input_name.c_str(),
            dynamic_tensor_num);
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("ParallelDynamicStitch")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ParallelDynamicStitch")
    .ParseParamsFn(ParallelDynamicStitchMapping)
    .ImplyType(ImplyType::AI_CPU);

// register SparseConcat op to GE
Status SparseConcatMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  std::shared_ptr<ge::OpDesc> op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  Status ret = AutoMappingFn(op_src, op);
  if (ret != SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), "call auto mapping function failed.");
    return FAILED;
  }
  OP_LOGI("op[%s] call auto mapping function success.", op_desc->GetName().c_str());

  std::string attr_name = "N";
  std::vector<std::string> dynamic_input{"indices", "values", "shapes"};
  const domi::tensorflow::NodeDef* node = reinterpret_cast<const domi::tensorflow::NodeDef*>(op_src);

  int32_t dynamic_tensor_num = 0;
  ret = GetAttrValue(node, attr_name, dynamic_tensor_num);
  if (ret != SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), attr_name.c_str(), "get attr [%s] value failed.");
    return FAILED;
  }

  bool is_pushback = true;
  for (std::string input_name : dynamic_input) {
    auto graph_ret = op_desc->AddDynamicInputDesc(input_name, static_cast<uint32_t>(dynamic_tensor_num), is_pushback);
    if (graph_ret != ge::GRAPH_SUCCESS) {
      OP_LOGE(op_desc->GetName().c_str(), "AddDynamicInputDesc failed.");
      return FAILED;
    }
    OP_LOGI("In NodeDef %s add dynamic input, input name = %s, size = %d", node->name().c_str(), input_name.c_str(),
            dynamic_tensor_num);
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("SparseConcat")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseConcat")
    .ParseParamsFn(SparseConcatMapping)
    .ImplyType(ImplyType::AI_CPU);

// register SparseCross op to GE
Status SparseCrossMapping(const google::protobuf::Message* op_src, ge::Operator& op) {
  std::shared_ptr<ge::OpDesc> op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  Status ret = AutoMappingFn(op_src, op);
  if (ret != SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), "call auto mapping function failed.");
    return FAILED;
  }
  OP_LOGI("op[%s] call auto mapping function success.", op_desc->GetName().c_str());

  const domi::tensorflow::NodeDef* node = reinterpret_cast<const domi::tensorflow::NodeDef*>(op_src);

  std::vector<std::string> dynamic_input{"indices", "shapes"};
  std::string attr_N = "N";
  int32_t N_value = 0;
  ret = GetAttrValue(node, attr_N, N_value);
  if (ret != SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), attr_N.c_str(), "get attr [%s] value failed.");
    return FAILED;
  }

  std::string value_input = "values";
  std::string attr_sparse_types = "sparse_types";
  int32_t sparse_types_value = 0;
  ret = GetAttrValue(node, attr_sparse_types, sparse_types_value);
  if (ret != SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), attr_sparse_types.c_str(), "get attr [%s] value failed.");
    return FAILED;
  }

  std::string dense_inputs = "dense_inputs";
  std::string attr_dense_types = "dense_types";
  int32_t dense_types_value = 0;
  ret = GetAttrValue(node, attr_dense_types, dense_types_value);
  if (ret != SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), attr_dense_types.c_str(), "get attr [%s] value failed.");
    return FAILED;
  }

  bool is_pushback = true;
  for (std::string input_name : dynamic_input) {
    auto graph_ret = op_desc->AddDynamicInputDesc(input_name, static_cast<uint32_t>(N_value), is_pushback);
    if (graph_ret != ge::GRAPH_SUCCESS) {
      OP_LOGE(op_desc->GetName().c_str(), "AddDynamicInputDesc failed.");
      return FAILED;
    }
    OP_LOGI("In NodeDef %s add dynamic input, input name = %s, size = %d", node->name().c_str(), input_name.c_str(),
            N_value);
  }
  auto graph_ret = op_desc->AddDynamicInputDesc(value_input, static_cast<uint32_t>(sparse_types_value), is_pushback);
  if (graph_ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), "AddDynamicInputDesc failed.");
    return FAILED;
  }
  OP_LOGI("In NodeDef %s add dynamic input, input name = %s, size = %d", node->name().c_str(), value_input.c_str(),
          sparse_types_value);

  graph_ret = op_desc->AddDynamicInputDesc(dense_inputs, static_cast<uint32_t>(dense_types_value), is_pushback);
  if (graph_ret != ge::GRAPH_SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), "AddDynamicInputDesc failed.");
    return FAILED;
  }
  OP_LOGI("In NodeDef %s add dynamic input, input name = %s, size = %d", node->name().c_str(), dense_inputs.c_str(),
          dense_types_value);

  return SUCCESS;
}

REGISTER_CUSTOM_OP("SparseCross")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("SparseCross")
    .ParseParamsFn(SparseCrossMapping)
    .ImplyType(ImplyType::AI_CPU);
}  // namespace domi
