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
 * \file scope_basic_lstm_cell_pass.cc
 * \brief
 */
#include "scope_basic_lstm_cell_pass.h"

#include "op_log.h"
#include "register/scope/scope_fusion_pass_register.h"

namespace ge {
static const char* const kScopeType = "BasicLSTMCell";
static const char* const kLstmCellTanhType = "general_basic_lstm_cell_tanh";
static const char* const kLstmCellReluType = "general_basic_lstm_cell_relu";
static const char* const kLstmCellRelu6Type = "general_basic_lstm_cell_relu6";
static const char* const kLstmCellSigmoidType = "general_basic_lstm_cell_sigmoid";
static const char* const kOpType = "BasicLSTMCell";

std::vector<ScopeFusionPatterns> ScopeBasicLSTMCellPass::DefinePatterns() {
  std::vector<ScopeFusionPatterns> patterns_list;
  ScopeFusionPatterns patterns;
  GenScopePatterns(patterns);
  patterns_list.push_back(patterns);
  return patterns_list;
}

std::string ScopeBasicLSTMCellPass::PassName() {
  return std::string("ScopeBasicLSTMCellPass");
}

Status ScopeBasicLSTMCellPass::LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph>& scope_graph,
                                                     std::vector<ScopesResult>& results) {
  if (scope_graph == nullptr) {
    OP_LOGE(kOpType, "Input params is nullptr.");
    return domi::PARAM_INVALID;
  }
  const ScopeTree* scope_tree = scope_graph->GetScopeTree();
  // Class ScopeGraph guarantees scope_tree is not empty.
  const std::vector<Scope*>& scopes = scope_tree->GetAllScopes();
  for (auto& scope : scopes) {
    // Class ScopeTree guarantees scope is not empty.
    if ((scope->SubType() == kLstmCellTanhType) || (scope->SubType() == kLstmCellReluType) ||
        (scope->SubType() == kLstmCellRelu6Type) || (scope->SubType() == kLstmCellSigmoidType)) {
      std::string scope_name = scope->Name();
      if (scope_name.find("while") == std::string::npos) {
        ScopesResult result;
        std::vector<Scope*> result_scopes;
        result_scopes.push_back(scope);
        result.SetScopes(result_scopes);
        results.push_back(result);
      }
    }
  }

  return (!(results.empty())) ? SUCCESS : FAILED;
}

void ScopeBasicLSTMCellPass::GenScopePatterns(ScopeFusionPatterns& patterns) {
  // distinguish basic_lstm_cell
  std::vector<ScopePattern*> batch;
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
  batch.push_back(basic_lstm_cell_tanh);

  ScopePattern* basic_lstm_cell_relu = new (std::nothrow) ScopePattern();
  if (basic_lstm_cell_relu == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch);
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
  batch.push_back(basic_lstm_cell_relu);

  ScopePattern* basic_lstm_cell_relu6 = new (std::nothrow) ScopePattern();
  if (basic_lstm_cell_relu6 == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch);
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
  batch.push_back(basic_lstm_cell_relu6);

  ScopePattern* basic_lstm_cell_sigmoid = new (std::nothrow) ScopePattern();
  if (basic_lstm_cell_sigmoid == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch);
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
  batch.push_back(basic_lstm_cell_sigmoid);

  patterns.push_back(batch);
}

void ScopeBasicLSTMCellPass::GenerateFusionResult(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt) {
  if (fusion_rlt == nullptr) {
    OP_LOGE(kOpType, "Input fusion_rlt is nullptr.");
    return;
  }
  fusion_rlt->InsertInputs("concat", {0, 1});                     // Input index 0 and 1 : x and h
  fusion_rlt->InsertInputs("Mul", {2, kFusionDisableIndex});      // Input index 2 : c
  fusion_rlt->InsertInputs("MatMul", {kFusionDisableIndex, 3});   // Input index 3 : w
  fusion_rlt->InsertInputs("BiasAdd", {kFusionDisableIndex, 4});  // Input index 4 : b

  fusion_rlt->InsertOutputs("Add_1", {0});  // Output index 0 : ct
  fusion_rlt->InsertOutputs("Mul_2", {1});  // Output index 1 : ht
  fusion_rlt->SetType(kScopeType);
  fusion_rlt->SetDescription("");

  for (auto& scope : scopes) {
    // The upper call guarantees that the scope is not empty.
    if (scope->SubType() == kLstmCellTanhType || scope->SubType() == kLstmCellReluType ||
        scope->SubType() == kLstmCellRelu6Type || scope->SubType() == kLstmCellSigmoidType) {
      std::string scope_name = scope->Name();
      fusion_rlt->SetName(scope_name.substr(0, scope_name.length() - 1));
    }
  }

  OP_LOGI(kOpType, "BasicLSTMCell Scope fusion success.");
  return;
}
}  // namespace ge
