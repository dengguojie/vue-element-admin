/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include "scope_dynamic_rnn_pass.h"

#include <set>

#include "op_log.h"
#include "register/scope/scope_fusion_pass_register.h"

namespace ge {
static const char* const kScopeType = "DynamicRNN";
static const char* const kLstmCellTanhType = "general_basic_lstm_cell_tanh";
static const char* const kFwWhileType = "fw_while";
static const char* const kBwWhileType = "bw_while";
static const char* const kRnnWhileType = "rnn_while";
static const char* const kFwWhile_noTranposeType = "fw_no_transpose_while";
static const char* const kBwWhile_noTranposeType = "bw_no_transpose_while";
static const char* const kKernelBiasType = "kernel_bias_in";
static const char* const kWhileType = "while";
static const char* const kOpType = "DynamicRNN";
static const char* const kTranspose = "transpose";
static const char* const kLstmTransType = "lstm_transpose_scope";
static const char* const kLstmNoTransType = "lstm_no_transpose_scope";

std::vector<ScopeFusionPatterns> ScopeDynamicRNNPass::DefinePatterns() {
  std::vector<ScopeFusionPatterns> patterns_list;
  ScopeFusionPatterns patterns1;
  GenScopePatterns(patterns1);
  patterns_list.push_back(patterns1);

  ScopeFusionPatterns patterns2;
  GenTacotronScopePatterns(patterns2);
  patterns_list.push_back(patterns2);

  return patterns_list;
}

std::string ScopeDynamicRNNPass::PassName() {
  return std::string("ScopeDynamicRNNPass");
}

void ScopeDynamicRNNPass::GenScopePatterns(ScopeFusionPatterns& patterns) {
  // distinguish dynamic_rnn_cell
  std::vector<ScopePattern*> batch1;
  ScopePattern* basic_lstm_cell_tanh = new (std::nothrow) ScopePattern();
  if (basic_lstm_cell_tanh == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch1);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }

  basic_lstm_cell_tanh->SetSubType(kLstmCellTanhType);
  // The number of Sigmoid is a multiple of 3.
  basic_lstm_cell_tanh->AddNodeOpTypeFeature(NodeOpTypeFeature("Sigmoid", 0, 3));
  // The number of Mul is a multiple of 3.
  basic_lstm_cell_tanh->AddNodeOpTypeFeature(NodeOpTypeFeature("Mul", 0, 3));
  // The number of Tanh is a multiple of 2.
  basic_lstm_cell_tanh->AddNodeOpTypeFeature(NodeOpTypeFeature("Tanh", 0, 2));
  // The number of MatMul is 1.
  basic_lstm_cell_tanh->AddNodeOpTypeFeature(NodeOpTypeFeature("MatMul", 1, 0));
  ScopeAttrValue split_attr_value1;
  // The Split node has a "num_split" attribute which value is 4.
  split_attr_value1.SetIntValue(4);
  basic_lstm_cell_tanh->AddNodeAttrFeature(NodeAttrFeature("Split", "num_split", ge::DT_INT32, split_attr_value1));

  batch1.push_back(basic_lstm_cell_tanh);
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

  ScopePattern* p_lstm_cell_with_kernel_bias = new (std::nothrow) ScopePattern();
  if (p_lstm_cell_with_kernel_bias == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch2);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  p_lstm_cell_with_kernel_bias->SetSubType(kKernelBiasType);
  p_lstm_cell_with_kernel_bias->AddNodeOpTypeFeature(NodeOpTypeFeature("Identity", 2, 0));
  batch2.push_back(p_lstm_cell_with_kernel_bias);
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
  fw_while->AddScopeFeature(ScopeFeature(kKernelBiasType, 0));
  fw_while->AddNodeOpTypeFeature(NodeOpTypeFeature("StridedSlice", 0, 1));
  fw_while->AddNodeOpTypeFeature(NodeOpTypeFeature("Transpose", 2, 0));
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
  bw_while->AddNodeOpTypeFeature(NodeOpTypeFeature("Transpose", 2, 0));
  bw_while->AddNodeOpTypeFeature(NodeOpTypeFeature("ReverseSequence", -1, 0));
  bw_while->AddNodeOpTypeFeature(NodeOpTypeFeature("ReverseV2", -1, 0));
  bw_while->AddScopeFeature(ScopeFeature(kKernelBiasType, 0));
  batch3.push_back(bw_while);

  ScopePattern* fw_no_trans_while = new (std::nothrow) ScopePattern();
  if (fw_no_trans_while == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch3);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  fw_no_trans_while->SetSubType(kFwWhile_noTranposeType);
  fw_no_trans_while->AddScopeFeature(ScopeFeature(kWhileType, 1, "fw"));
  fw_no_trans_while->AddScopeFeature(ScopeFeature(kKernelBiasType, 0));
  fw_no_trans_while->AddNodeOpTypeFeature(NodeOpTypeFeature("StridedSlice", 0, 1));
  fw_no_trans_while->AddNodeOpTypeFeature(NodeOpTypeFeature("Transpose", -1, 0));
  batch3.push_back(fw_no_trans_while);

  ScopePattern* bw_no_trans_while = new (std::nothrow) ScopePattern();
  if (bw_no_trans_while == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch3);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  bw_no_trans_while->SetSubType(kBwWhile_noTranposeType);
  bw_no_trans_while->AddScopeFeature(ScopeFeature(kWhileType, 1, "bw"));
  bw_no_trans_while->AddNodeOpTypeFeature(NodeOpTypeFeature("StridedSlice", 0, 1));
  bw_no_trans_while->AddNodeOpTypeFeature(NodeOpTypeFeature("Transpose", -1, 0));
  bw_no_trans_while->AddNodeOpTypeFeature(NodeOpTypeFeature("ReverseSequence", -1, 0));
  bw_no_trans_while->AddNodeOpTypeFeature(NodeOpTypeFeature("ReverseV2", -1, 0));
  bw_no_trans_while->AddScopeFeature(ScopeFeature(kKernelBiasType, 0));
  batch3.push_back(bw_no_trans_while);

  ScopePattern* rnn_while = new (std::nothrow) ScopePattern();
  if (rnn_while == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch3);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  rnn_while->SetSubType(kRnnWhileType);
  rnn_while->AddScopeFeature(ScopeFeature(kWhileType, 1, "rnn"));
  rnn_while->AddScopeFeature(ScopeFeature(kKernelBiasType, 0));
  rnn_while->AddNodeOpTypeFeature(NodeOpTypeFeature("TensorArrayV3", 0, 1));
  rnn_while->AddNodeOpTypeFeature(NodeOpTypeFeature("Transpose", 2, 0));
  batch3.push_back(rnn_while);
  patterns.push_back(batch3);
}

void ScopeDynamicRNNPass::GenTacotronScopePatterns(ScopeFusionPatterns &patterns){
  // distinguish lstm
  std::vector<ScopePattern *> batch1;
  ScopePattern *lstm_while = new(std::nothrow) ScopePattern();
  if (lstm_while == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch1);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }

  lstm_while->SetSubType(kWhileType);
  // The number of BiasAdd is a multiple of 3.
  lstm_while->AddNodeOpTypeFeature(NodeOpTypeFeature("BiasAdd", 0, 4));
  // The number of Mul is a Sigmoid of 3.
  lstm_while->AddNodeOpTypeFeature(NodeOpTypeFeature("Sigmoid", 0, 3));
  // The number of Tanh is a multiple of 2.
  lstm_while->AddNodeOpTypeFeature(NodeOpTypeFeature("Tanh", 0, 2));
  // The number of MatMul is 8.
  lstm_while->AddNodeOpTypeFeature(NodeOpTypeFeature("MatMul", 8, 0));
  ScopeAttrValue split_attr_value1;
  // The Split node has a "num_split" attribute which value is 4.
  split_attr_value1.SetIntValue(4);
  lstm_while->AddNodeAttrFeature(NodeAttrFeature("Split", "num_split", ge::DT_INT32, split_attr_value1));

  batch1.push_back(lstm_while);
  patterns.push_back(batch1);

  // distinguish Lstm
  std::vector<ScopePattern *> batch2;
  ScopePattern *p_lstm_no_transpose = new(std::nothrow) ScopePattern();
  if (p_lstm_no_transpose == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch2);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  p_lstm_no_transpose->SetSubType(kLstmNoTransType);
  p_lstm_no_transpose->AddScopeFeature(ScopeFeature(kWhileType, 1, ""));
  p_lstm_no_transpose->AddNodeOpTypeFeature(NodeOpTypeFeature("Transpose", -1, 0));
  batch2.push_back(p_lstm_no_transpose);

  ScopePattern *p_lstm_transpose = new(std::nothrow) ScopePattern();
  if (p_lstm_transpose == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch2);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  p_lstm_transpose->SetSubType(kLstmTransType);
  p_lstm_transpose->AddScopeFeature(ScopeFeature(kWhileType, 1, ""));
  p_lstm_transpose->AddNodeOpTypeFeature(NodeOpTypeFeature("Transpose", 0, 1));
  batch2.push_back(p_lstm_transpose);
  patterns.push_back(batch2);
}

Status ScopeDynamicRNNPass::LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph>& scope_graph,
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
  std::set<string> support_types = {kFwWhileType, kBwWhileType, kFwWhile_noTranposeType, kBwWhile_noTranposeType,
                                    kRnnWhileType, kLstmNoTransType, kLstmTransType};
  for (auto& scope : scopes) {
    // Class ScopeTree guarantees scope is not empty.
    if (support_types.count(scope->SubType()) != 0) {
      ScopesResult result;
      std::vector<Scope*> result_scopes;
      result_scopes.push_back(scope);
      result.SetScopes(result_scopes);
      results.push_back(result);
    }
  }
  return (!(results.empty())) ? SUCCESS : FAILED;
}

void ScopeDynamicRNNPass::GenerateFusionResult(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt) {
  if (fusion_rlt == nullptr) {
    OP_LOGE(kOpType, "Input fusion_rlt is nullptr.");
    return;
  }
  for (auto& scope : scopes) {
    std::string w_in_node_name;
    std::string b_in_node_name;
    OP_LOGD(kOpType, "Match DynamicRNN scope name is %s ", scope->Name().c_str());
    const std::unordered_map<std::string, ge::OperatorPtr>& nodes_map = scope->AllNodesMap();
    for (auto& it : nodes_map) {
      auto node_def = it.second;
      std::string sub_node_name = node_def->GetName().c_str();
      if (scope->SubType() == kLstmTransType || scope->SubType() == kLstmNoTransType) {
        // keras lstm for tacotron
        if (sub_node_name.find("split/ReadVariableOp/Enter") != string::npos) {
          w_in_node_name = sub_node_name;
        }
        if (sub_node_name.find("split_1/ReadVariableOp/Enter") != string::npos) {
          b_in_node_name = sub_node_name;
        }
      } else {
        // original
        if (sub_node_name.find("lstm_cell/MatMul/Enter") != string::npos) {
          w_in_node_name = sub_node_name;
        }
        if (sub_node_name.find("lstm_cell/BiasAdd/Enter") != string::npos) {
          b_in_node_name = sub_node_name;
        }
      }
      if (!w_in_node_name.empty() && !b_in_node_name.empty()) {
        OP_LOGD(kOpType, "DynamicRNN w input node name is %s ", w_in_node_name.c_str());
        OP_LOGD(kOpType, "DynamicRNN b input node name is %s ", b_in_node_name.c_str());
        break;
      }
    }
    if (scope->SubType() == kFwWhileType || scope->SubType() == kBwWhileType ||
        scope->SubType() == kRnnWhileType) {
      fusion_rlt->InsertInputs("transpose", {0, kFusionDisableIndex});  // Input 0 : x
      fusion_rlt->InsertOutputs("transpose_1", {0});                    // Output index 0 : outputs
    }
    if (scope->SubType() == kLstmTransType){
      fusion_rlt->InsertInputs("transpose", {0, kFusionDisableIndex});  // Input 0 : x
      fusion_rlt->InsertOutputs("transpose_2", {0});                    // Output index 0 : outputs
    }
    if (scope->SubType() == kFwWhile_noTranposeType || scope->SubType() == kBwWhile_noTranposeType
            || scope->SubType() == kLstmNoTransType) {
      fusion_rlt->InsertInputs("TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3",
                               {kFusionDisableIndex, kFusionDisableIndex, 0, kFusionDisableIndex});  // Input 0 : x
      fusion_rlt->InsertOutputs("TensorArrayStack/TensorArrayGatherV3", {0});  // Output index 0 : outputs
    }
    if (scope->SubType() == kRnnWhileType) {
      fusion_rlt->InsertInputs("while/Enter_4", {4});  // Input 4 : init_h
      fusion_rlt->InsertInputs("while/Enter_3", {5});  // Input 5 : init_c
    }
    fusion_rlt->InsertInputs(w_in_node_name, {1});  // Input index 1 : w
    fusion_rlt->InsertInputs(b_in_node_name, {2});  // Input index 2 : b
    fusion_rlt->SetType(kScopeType);
    fusion_rlt->SetDescription("");
    std::string scope_name = scope->Name();
    fusion_rlt->SetName(scope_name.substr(0, scope_name.length() - 1));
  }
  OP_LOGI(kOpType, "DynamicRNN Scope fusion success.");
  return;
}
}  // namespace ge
