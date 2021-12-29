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
 * \file scope_dynamic_lstm_pass.cc
 * \brief
 */
#include "scope_dynamic_lstm_pass.h"

#include "op_log.h"

namespace ge {
static const char* const kScopeType = "DynamicRNN";
static const char* const kLstmCellTanhType = "general_basic_lstm_cell_tanh";
static const char* const kLstmCellReluType = "general_basic_lstm_cell_relu";
static const char* const kLstmCellRelu6Type = "general_basic_lstm_cell_relu6";
static const char* const kLstmCellSigmoidType = "general_basic_lstm_cell_sigmoid";
static const char* const kFwWhileType = "fw_while";
static const char* const kBwWhileType = "bw_while";
static const char* const kRnnWhileType = "rnn_while";
static const char* const kWhileType = "while";
static const char* const kOpType = "DynamicRNN";
static const char* const kTranspose = "transpose";

std::vector<ScopeFusionPatterns> ScopeDynamicLSTMPass::DefinePatterns() {
  std::vector<ScopeFusionPatterns> patterns_list;
  ScopeFusionPatterns patterns;
  GenScopePatterns(patterns);
  patterns_list.push_back(patterns);
  return patterns_list;
}

std::string ScopeDynamicLSTMPass::PassName() {
  return std::string("ScopeDynamicLSTMPass");
}

void ScopeDynamicLSTMPass::GenScopePatterns(ScopeFusionPatterns& patterns) {
  // distinguish basic_lstm_cell
  std::vector<ScopePattern*> batch1;
  ScopePattern* basic_lstm_cell_tanh = new (std::nothrow) ScopePattern();
  if (basic_lstm_cell_tanh == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }

  basic_lstm_cell_tanh->SetSubType(kLstmCellTanhType);
  basic_lstm_cell_tanh->AddNodeOpTypeFeature(
      NodeOpTypeFeature("Sigmoid", 0, 3));  // The number of Sigmoid is a multiple of 3.
  basic_lstm_cell_tanh->AddNodeOpTypeFeature(NodeOpTypeFeature("Mul", 0, 3));  // The number of Mul is a multiple of 3.
  basic_lstm_cell_tanh->AddNodeOpTypeFeature(
      NodeOpTypeFeature("Tanh", 0, 2));  // The number of Tanh is a multiple of 2.
  basic_lstm_cell_tanh->AddNodeOpTypeFeature(NodeOpTypeFeature("MatMul", 1, 0));  // The number of MatMul is 1.
  ScopeAttrValue split_attr_value1;
  // The Split node has a "num_split" attribute which value is 4.
  split_attr_value1.SetIntValue(4);
  basic_lstm_cell_tanh->AddNodeAttrFeature(NodeAttrFeature("Split", "num_split", ge::DT_INT32, split_attr_value1));
  batch1.push_back(basic_lstm_cell_tanh);

  ScopePattern* basic_lstm_cell_relu = new (std::nothrow) ScopePattern();
  if (basic_lstm_cell_relu == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch1);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }

  basic_lstm_cell_relu->SetSubType(kLstmCellReluType);
  basic_lstm_cell_relu->AddNodeOpTypeFeature(
      NodeOpTypeFeature("Sigmoid", 0, 3));  // The number of Sigmoid is a multiple of 3.
  basic_lstm_cell_relu->AddNodeOpTypeFeature(NodeOpTypeFeature("Mul", 0, 3));  // The number of Mul is a multiple of 3.
  basic_lstm_cell_relu->AddNodeOpTypeFeature(
      NodeOpTypeFeature("Relu", 0, 2));  // The number of Relu is a multiple of 2.
  basic_lstm_cell_relu->AddNodeOpTypeFeature(NodeOpTypeFeature("MatMul", 1, 0));  // The number of MatMul is 1.
  ScopeAttrValue split_attr_value2;
  // The Split node has a "num_split" attribute which value is 4.
  split_attr_value2.SetIntValue(4);
  basic_lstm_cell_relu->AddNodeAttrFeature(NodeAttrFeature("Split", "num_split", ge::DT_INT32, split_attr_value2));
  batch1.push_back(basic_lstm_cell_relu);

  ScopePattern* basic_lstm_cell_relu6 = new (std::nothrow) ScopePattern();
  if (basic_lstm_cell_relu6 == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch1);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  basic_lstm_cell_relu6->SetSubType(kLstmCellRelu6Type);
  basic_lstm_cell_relu6->AddNodeOpTypeFeature(
      NodeOpTypeFeature("Sigmoid", 0, 3));  // The number of Sigmoid is a multiple of 3.
  basic_lstm_cell_relu6->AddNodeOpTypeFeature(NodeOpTypeFeature("Mul", 0, 3));  // The number of Mul is a multiple of 3.
  basic_lstm_cell_relu6->AddNodeOpTypeFeature(
      NodeOpTypeFeature("Relu6", 0, 2));  // The number of Relu6 is a multiple of 2.
  basic_lstm_cell_relu6->AddNodeOpTypeFeature(NodeOpTypeFeature("MatMul", 1, 0));  // The number of MatMul is 1.
  ScopeAttrValue split_attr_value3;
  // The Split node has a "num_split" attribute which value is 4.
  split_attr_value3.SetIntValue(4);
  basic_lstm_cell_relu6->AddNodeAttrFeature(NodeAttrFeature("Split", "num_split", ge::DT_INT32, split_attr_value3));
  batch1.push_back(basic_lstm_cell_relu6);

  ScopePattern* basic_lstm_cell_sigmoid = new (std::nothrow) ScopePattern();
  if (basic_lstm_cell_sigmoid == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch1);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  basic_lstm_cell_sigmoid->SetSubType(kLstmCellSigmoidType);
  basic_lstm_cell_sigmoid->AddNodeOpTypeFeature(
      NodeOpTypeFeature("Sigmoid", 0, 5));  // The number of Sigmoid is a multiple of 5.
  basic_lstm_cell_sigmoid->AddNodeOpTypeFeature(
      NodeOpTypeFeature("Mul", 0, 3));  // The number of Mul is a multiple of 3.
  basic_lstm_cell_sigmoid->AddNodeOpTypeFeature(NodeOpTypeFeature("Relu6", -1, 0));  // Doesn't has Relu6.
  basic_lstm_cell_sigmoid->AddNodeOpTypeFeature(NodeOpTypeFeature("Relu", -1, 0));   // Doesn't has Relu.
  basic_lstm_cell_sigmoid->AddNodeOpTypeFeature(NodeOpTypeFeature("Tanh", -1, 0));   // Doesn't has Tanh.
  basic_lstm_cell_sigmoid->AddNodeOpTypeFeature(NodeOpTypeFeature("MatMul", 1, 0));  // The number of MatMul is 1.
  ScopeAttrValue split_attr_value4;
  // The Split node has a "num_split" attribute which value is 4.
  split_attr_value4.SetIntValue(4);
  basic_lstm_cell_sigmoid->AddNodeAttrFeature(NodeAttrFeature("Split", "num_split", ge::DT_INT32, split_attr_value4));
  batch1.push_back(basic_lstm_cell_sigmoid);
  patterns.push_back(batch1);

  // distinguish while
  std::vector<ScopePattern*> batch2;
  ScopePattern* p_lstm_tanh_while = new (std::nothrow) ScopePattern();
  if (p_lstm_tanh_while == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch2);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  p_lstm_tanh_while->SetSubType(kWhileType);
  p_lstm_tanh_while->AddScopeFeature(ScopeFeature(kLstmCellTanhType, 1, "while"));
  batch2.push_back(p_lstm_tanh_while);

  ScopePattern* p_lstm_relu_while = new (std::nothrow) ScopePattern();
  if (p_lstm_relu_while == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch2);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  p_lstm_relu_while->SetSubType(kWhileType);
  p_lstm_relu_while->AddScopeFeature(ScopeFeature(kLstmCellReluType, 1, "while"));
  batch2.push_back(p_lstm_relu_while);

  ScopePattern* p_lstm_relu6_while = new (std::nothrow) ScopePattern();
  if (p_lstm_relu6_while == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch2);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  p_lstm_relu6_while->SetSubType(kWhileType);
  p_lstm_relu6_while->AddScopeFeature(ScopeFeature(kLstmCellRelu6Type, 1, "while"));
  batch2.push_back(p_lstm_relu6_while);
  ScopePattern* p_lstm_sigmoid_while = new (std::nothrow) ScopePattern();
  if (p_lstm_sigmoid_while == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch2);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  p_lstm_sigmoid_while->SetSubType(kWhileType);
  p_lstm_sigmoid_while->AddScopeFeature(ScopeFeature(kLstmCellSigmoidType, 1, "while"));
  batch2.push_back(p_lstm_sigmoid_while);
  patterns.push_back(batch2);

  std::vector<ScopePattern*> batch3;
  ScopePattern* fw_while = new (std::nothrow) ScopePattern();
  if (fw_while == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch3);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  fw_while->SetSubType(kFwWhileType);
  fw_while->AddScopeFeature(ScopeFeature(kWhileType, 1, "fw"));
  fw_while->AddNodeOpTypeFeature(NodeOpTypeFeature("StridedSlice", 0, 1));
  fw_while->AddNodeOpTypeFeature(NodeOpTypeFeature("TensorArrayV3", 0, 1));
  fw_while->AddNodeOpTypeFeature(NodeOpTypeFeature("Minimum", 0, 1));
  batch3.push_back(fw_while);

  ScopePattern* bw_while = new (std::nothrow) ScopePattern();
  if (bw_while == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch3);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  bw_while->SetSubType(kBwWhileType);
  bw_while->AddScopeFeature(ScopeFeature(kWhileType, 1, "bw"));
  bw_while->AddNodeOpTypeFeature(NodeOpTypeFeature("StridedSlice", 0, 1));
  bw_while->AddNodeOpTypeFeature(NodeOpTypeFeature("TensorArrayV3", 0, 1));
  bw_while->AddNodeOpTypeFeature(NodeOpTypeFeature("Minimum", 0, 1));
  bw_while->AddNodeOpTypeFeature(NodeOpTypeFeature("ReverseSequence", -1, 0));
  batch3.push_back(bw_while);

  ScopePattern* rnn_while = new (std::nothrow) ScopePattern();
  if (rnn_while == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch3);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  rnn_while->SetSubType(kRnnWhileType);
  rnn_while->AddScopeFeature(ScopeFeature(kWhileType, 1, "rnn"));
  rnn_while->AddNodeOpTypeFeature(NodeOpTypeFeature("StridedSlice", 0, 1));
  rnn_while->AddNodeOpTypeFeature(NodeOpTypeFeature("TensorArrayV3", 0, 1));
  rnn_while->AddNodeOpTypeFeature(NodeOpTypeFeature("Minimum", 0, 1));
  batch3.push_back(rnn_while);
  patterns.push_back(batch3);
}

Status ScopeDynamicLSTMPass::LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph>& scope_graph,
                                                   std::vector<ScopesResult>& results) {
  if (scope_graph == nullptr) {
    OP_LOGE(kOpType, "Input params is nullptr.");
    return domi::PARAM_INVALID;
  }
  const ScopeTree* scope_tree = scope_graph->GetScopeTree();
  if (scope_tree == nullptr) {
    OP_LOGE(kOpType, "Scope tree is nullptr.");
    return domi::PARAM_INVALID;
  }
  // Class ScopeGraph guarantees scope_tree is not empty.
  const std::vector<Scope*>& scopes = scope_tree->GetAllScopes();
  for (auto& scope : scopes) {
    // Class ScopeTree guarantees scope is not empty.
    if ((scope->SubType() == kFwWhileType) || (scope->SubType() == kBwWhileType) ||
        (scope->SubType() == kRnnWhileType)) {
      ScopesResult result;
      std::vector<Scope*> result_scopes;
      result_scopes.push_back(scope);
      result.SetScopes(result_scopes);
      results.push_back(result);
    }
  }
  return (!(results.empty())) ? SUCCESS : FAILED;
}

void ScopeDynamicLSTMPass::GenerateFusionResult(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt) {
  if (fusion_rlt == nullptr) {
    OP_LOGE(kOpType, "Input fusion_rlt is nullptr.");
    return;
  }
  for (auto& scope : scopes) {
    bool time_major = true;
    const std::unordered_map<std::string, ge::OperatorPtr>& nodes_map = scope->AllNodesMap();
    for (auto& it : nodes_map) {
      auto node_def = it.second;
      std::string sub_name = node_def->GetName().c_str();
      if (sub_name.find(kTranspose) != sub_name.npos) {
        time_major = false;
        break;
      }
    }
    if (scope->SubType() == kFwWhileType) {
      if (time_major) {
        fusion_rlt->InsertInputs("fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3",
                                 {kFusionDisableIndex, kFusionDisableIndex, 0, kFusionDisableIndex});  // Input 0 : x
        fusion_rlt->InsertOutputs("fw/fw/TensorArrayStack/TensorArrayGatherV3", {0});  // Output index 0 : outputs
      } else {
        fusion_rlt->InsertInputs("fw/fw/transpose", {0, kFusionDisableIndex});     // Input 0 : x
        fusion_rlt->InsertOutputs("fw/fw/transpose_1", {0, kFusionDisableIndex});  // Output index 0 : outputs
      }
      fusion_rlt->InsertInputs("fw/fw/while/basic_lstm_cell/MatMul/Enter", {1});   // Input index 3 : w
      fusion_rlt->InsertInputs("fw/fw/while/basic_lstm_cell/BiasAdd/Enter", {2});  // Input index 4 : b
      fusion_rlt->SetType(kScopeType);
      fusion_rlt->SetDescription("");
      std::string scope_name = scope->Name();
      fusion_rlt->SetName(scope_name.substr(0, scope_name.length() - 1));
    }
    if (scope->SubType() == kBwWhileType) {
      if (time_major) {
        fusion_rlt->InsertInputs("bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3",
                                 {kFusionDisableIndex, kFusionDisableIndex, 0, kFusionDisableIndex});  // Input 0 : x
        fusion_rlt->InsertOutputs("bw/TensorArrayStack/TensorArrayGatherV3", {0});  // Output index 0 : outputs
      } else {
        fusion_rlt->InsertInputs("bw/transpose", {0, kFusionDisableIndex});     // Input 0 : x
        fusion_rlt->InsertOutputs("bw/transpose_1", {0, kFusionDisableIndex});  // Output index 0 : outputs
      }
      fusion_rlt->InsertInputs("bw/while/basic_lstm_cell/MatMul/Enter", {1});   // Input index 1 : w
      fusion_rlt->InsertInputs("bw/while/basic_lstm_cell/BiasAdd/Enter", {2});  // Input index 2 : b
      fusion_rlt->SetType(kScopeType);
      fusion_rlt->SetDescription("");
      std::string scope_name = scope->Name();
      fusion_rlt->SetName(scope_name.substr(0, scope_name.length() - 1));
    }
    if (scope->SubType() == kRnnWhileType) {
      if (time_major) {
        fusion_rlt->InsertInputs("rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3",
                                 {kFusionDisableIndex, kFusionDisableIndex, 0, kFusionDisableIndex});  // Input 0 : x
        fusion_rlt->InsertOutputs("rnn/TensorArrayStack/TensorArrayGatherV3", {0});  // Output index 0 : outputs
      } else {
        fusion_rlt->InsertInputs("rnn/transpose", {0, kFusionDisableIndex});     // Input 0 : x
        fusion_rlt->InsertOutputs("rnn/transpose_1", {0, kFusionDisableIndex});  // Output index 0 : outputs
      }
      fusion_rlt->InsertInputs("rnn/while/basic_lstm_cell/MatMul/Enter", {1});   // Input index 1 : w
      fusion_rlt->InsertInputs("rnn/while/basic_lstm_cell/BiasAdd/Enter", {2});  // Input index 2 : b
      fusion_rlt->SetType(kScopeType);
      fusion_rlt->SetDescription("");
      std::string scope_name = scope->Name();
      fusion_rlt->SetName(scope_name.substr(0, scope_name.length() - 1));
    }
  }

  OP_LOGI(kOpType, "DynamicLSTM Scope fusion success.");
  return;
}
}  // namespace ge
