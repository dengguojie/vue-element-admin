/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

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
#include "graph/utils/op_desc_utils.h"
#include "graph/operator.h"
#include "register/scope/scope_fusion_pass_register.h"
#include "register/register.h"

namespace ge {
static const char* kScopeType = "DynamicRNN";
static const char* kLstmCellTanhType = "general_basic_lstm_cell_tanh";
static const char* kFwWhileType = "fw_while";
static const char* kBwWhileType = "bw_while";
static const char* kRnnWhileType = "rnn_while";
static const char* kFwWhile_noTranposeType = "fw_no_transpose_while";
static const char* kBwWhile_noTranposeType = "bw_no_transpose_while";
static const char* kKernelBiasType = "kernel_bias_in";
static const char* kWhileType = "while";
static const char* kOpType = "DynamicRNN";
static const char* kLstmTransType = "lstm_transpose_scope";
static const char* kLstmNoTransType = "lstm_no_transpose_scope";
static const char* kLstmTwoTransType = "lstm_two_transpose_scope";
static const char* kLstmCell = "ganeral_lstm_cell";
static const char* kMultiRNNCell = "ganeral_rnn_cell";
static const char* kRNNTwoTransMaximumType = "lt_crnn_lstm";

std::vector<ScopeFusionPatterns> ScopeDynamicRNNPass::DefinePatterns() {
  std::vector<ScopeFusionPatterns> patterns_list;
  ScopeFusionPatterns patterns1;
  GenScopePatterns(patterns1);
  patterns_list.push_back(patterns1);

  ScopeFusionPatterns patterns2;
  GenTacotronScopePatterns(patterns2);
  patterns_list.push_back(patterns2);

  ScopeFusionPatterns patterns3;
  GenChinaMobileScopePatterns(patterns3);
  patterns_list.push_back(patterns3);

  ScopeFusionPatterns patterns4;
  GenLTCRNNScopePatterns(patterns4);
  patterns_list.push_back(patterns4);

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

void ScopeDynamicRNNPass::GenLTCRNNScopePatterns(ScopeFusionPatterns &patterns){
  std::vector<ScopePattern *> batch1;
  ScopePattern *lstm_crnn = new(std::nothrow) ScopePattern();
  if (lstm_crnn == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch1);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }

  lstm_crnn->SetSubType(kRNNTwoTransMaximumType);
  // The number of BiasAdd is a multiple of 4.
  lstm_crnn->AddNodeOpTypeFeature(NodeOpTypeFeature("BiasAdd", 0, 4));
  // The number of MatMul is 8.
  lstm_crnn->AddNodeOpTypeFeature(NodeOpTypeFeature("MatMul", 8, 0));
  // The number of Maximum is a multiple of 5.
  lstm_crnn->AddNodeOpTypeFeature(NodeOpTypeFeature("Maximum", 0, 5));
  // The number of Transpose is a multiple of 2.
  lstm_crnn->AddNodeOpTypeFeature(NodeOpTypeFeature("Transpose", 0, 2));

  batch1.push_back(lstm_crnn);
  patterns.push_back(batch1);
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
  p_lstm_transpose->AddNodeOpTypeFeature(NodeOpTypeFeature("Transpose", 0, 2));
  batch2.push_back(p_lstm_transpose);

  ScopePattern *p_lstm_two_transpose = new(std::nothrow) ScopePattern();
  if (p_lstm_two_transpose == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch2);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  p_lstm_two_transpose->SetSubType(kLstmTwoTransType);
  p_lstm_two_transpose->AddScopeFeature(ScopeFeature(kWhileType, 1, ""));
  p_lstm_two_transpose->AddNodeOpTypeFeature(NodeOpTypeFeature("Transpose", 0, 3));
  batch2.push_back(p_lstm_two_transpose);
  patterns.push_back(batch2);
}

void ScopeDynamicRNNPass::GenChinaMobileScopePatterns(ScopeFusionPatterns &patterns){
  // distinguish lstm
  std::vector<ScopePattern *> batch1;
  ScopePattern *basic_lstm_cell_tanh = new(std::nothrow) ScopePattern();
  if (basic_lstm_cell_tanh == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch1);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }

  basic_lstm_cell_tanh->SetSubType(kLstmCellTanhType);
  // The number of Sigmoid is a multiple of 3.
  basic_lstm_cell_tanh->AddNodeOpTypeFeature(NodeOpTypeFeature("Sigmoid", 0, 3));
  // The number of Mul is a Sigmoid of 3.
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

  // distinguish Lstm
  std::vector<ScopePattern *> batch2;
  ScopePattern *p_lstm_cell = new(std::nothrow) ScopePattern();
  if (p_lstm_cell == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch2);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  p_lstm_cell->SetSubType(kLstmCell);
  p_lstm_cell->AddScopeFeature(ScopeFeature(kLstmCellTanhType, 1, ""));
  batch2.push_back(p_lstm_cell);
  patterns.push_back(batch2);

  std::vector<ScopePattern *> batch3;
  ScopePattern *p_multi_rnn_cell = new(std::nothrow) ScopePattern();
  if (p_multi_rnn_cell == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch3);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  p_multi_rnn_cell->SetSubType(kMultiRNNCell);
  p_multi_rnn_cell->AddScopeFeature(ScopeFeature(kLstmCell, 3, ""));
  batch3.push_back(p_multi_rnn_cell);
  patterns.push_back(batch3);
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
                                    kRnnWhileType, kLstmNoTransType, kLstmTransType, kLstmTwoTransType, kMultiRNNCell,
                                    kRNNTwoTransMaximumType};
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

void ScopeDynamicRNNPass::DynamicRNNPassParserParams(const std::unordered_map<std::string, ge::OperatorPtr> &nodes_map,
                                                     const std::string &origin_node_name, const std::string &op_type,
                                                     ge::Operator *inner_node) {
  for (auto& it : nodes_map) {
    auto node_def = it.second;
    std::string sub_node_name = node_def->GetName().c_str();
    if (sub_node_name == origin_node_name) {
      if (op_type == "Const") {
        DataType dtype;
        if (node_def->GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
          OP_LOGE(node_def->GetName().c_str(), "get attr dtype failed");
          return;
        }
        inner_node->SetAttr("dtype", dtype);

        ge::Tensor tensor;
        if (node_def->GetAttr("value", tensor) != GRAPH_SUCCESS) {
          OP_LOGE(node_def->GetName().c_str(), "get attr value failed");
          return;
        }
        inner_node->SetAttr("value", tensor);
        break;
      }
      if (op_type == "Transpose") {
        DataType dtype;
        if (node_def->GetAttr("T", dtype) != GRAPH_SUCCESS) {
          OP_LOGE(node_def->GetName().c_str(), "get attr T failed");
          return;
        }
        inner_node->SetAttr("T", dtype);

        DataType dtype_perm;
        if (node_def->GetAttr("Tperm", dtype_perm) != GRAPH_SUCCESS) {
          OP_LOGE(node_def->GetName().c_str(), "get attr Tperm failed");
          return;
        }
        inner_node->SetAttr("Tperm", dtype_perm);
        break;
      }
      if (op_type == "Range") {
        DataType tidx;
        if (node_def->GetAttr("Tidx", tidx) != GRAPH_SUCCESS) {
          OP_LOGE(node_def->GetName().c_str(), "get attr Tidx failed");
          return;
        }
        inner_node->SetAttr("Tidx", tidx);
        break;
      }
    }
  }
}

void ScopeDynamicRNNPass::ConcatParserParams(const std::string &origin_node_name, const std::string &op_type, ge::Operator* inner_node, FusionScopesResult *fusion_rlt){
  std::vector<domi::DynamicInputOutputInfo> dynamic_name_attr_value;
  domi::DynamicInputOutputInfo dyn_info;
  dyn_info.type = domi::kInput;
  dyn_info.port_name = "x";
  dyn_info.port_name_len = 1;
  dyn_info.attr_name = "N";
  dyn_info.attr_name_len = 1;
  dynamic_name_attr_value.emplace_back(dyn_info);
  Operator op_src(origin_node_name, op_type);
  int dyn_num = 2;
  op_src.SetAttr("N", dyn_num);
  CHECK_INNER_NODE_CONDITION(inner_node != nullptr, fusion_rlt);
  AutoMappingByOpFnDynamic(op_src, *(inner_node), dynamic_name_attr_value);
  op_src.BreakConnect();
}

std::string ScopeDynamicRNNPass::GetNodeNameFromScope(const std::unordered_map<std::string, ge::OperatorPtr>& nodes_map, const std::string &sub_name){
  for (auto& it : nodes_map){
    auto node_def = it.second;
    std::string sub_node_name = node_def->GetName().c_str();
    int src_len = sub_node_name.size();
    int sub_len = sub_name.size();
    if (sub_node_name.find(sub_name) != string::npos && sub_node_name.substr(src_len - sub_len, sub_len) == sub_name){
      return sub_node_name;
    }
  }
  OP_LOGE(kOpType, "Cannot find node [%s] in scope.", sub_name.c_str());
  return "";
}

void ScopeDynamicRNNPass::GenerateFusionResultForMultiLSTM(const Scope* scope, FusionScopesResult *fusion_rlt) {
  OP_LOGD(kOpType, "Match DynamicRNN scope name is %s ", scope->Name().c_str());
  const std::unordered_map<std::string, ge::OperatorPtr> &nodes_map = scope->AllNodesMap();
  fusion_rlt->InsertInputs("transpose", {0, kFusionDisableIndex});
  fusion_rlt->InsertOutputs("transpose_1", {0});
  fusion_rlt->SetType(kScopeToMultiNodes);
  std::string scope_name = scope->Name();
  fusion_rlt->SetName(scope_name.substr(0, scope_name.length() - 1));
  fusion_rlt->SetDescription("");

  // start to range
  auto start_0 = fusion_rlt->AddInnerNode("start_0", "Const");
  CHECK_INNER_NODE_CONDITION(start_0 != nullptr, fusion_rlt);
  Status ret = start_0->InsertOutput("range_0", 0).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, "rnn/range/start", start_0->GetType(), start_0->MutableOperator());

  // rank to range
  auto rank_0 = fusion_rlt->AddInnerNode("rank_0", "Const");
  CHECK_INNER_NODE_CONDITION(rank_0 != nullptr, fusion_rlt);
  ret = rank_0->InsertOutput("range_0", 1).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, "rnn/Rank", rank_0->GetType(), rank_0->MutableOperator());

  // delta to range
  auto delta_0 = fusion_rlt->AddInnerNode("delta_0", "Const");
  CHECK_INNER_NODE_CONDITION(delta_0 != nullptr, fusion_rlt);
  ret = delta_0->InsertOutput("range_0", 2).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, "rnn/range/delta", delta_0->GetType(), delta_0->MutableOperator());

  // range to concat
  auto range_0 = fusion_rlt->AddInnerNode("range_0", "Range");
  CHECK_INNER_NODE_CONDITION(range_0 != nullptr, fusion_rlt);
  ret = range_0->InsertInput("start_0", 0)
          .InsertInput("rank_0", 0)
          .InsertInput("delta_0", 0)
          .InsertOutput("concat_0", 0)
          .BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, "rnn/range", range_0->GetType(), range_0->MutableOperator());

  // values to concat
  auto values_0 = fusion_rlt->AddInnerNode("values_0", "Const");
  CHECK_INNER_NODE_CONDITION(values_0 != nullptr, fusion_rlt);
  ret = values_0->InsertOutput("concat_0", 0).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, "rnn/concat/values_0", values_0->GetType(), values_0->MutableOperator());

  // axis to concat
  auto axis_0 = fusion_rlt->AddInnerNode("axis_0", "Const");
  CHECK_INNER_NODE_CONDITION(axis_0 != nullptr, fusion_rlt);
  ret = axis_0->InsertOutput("concat_0", 1).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, "rnn/concat/axis", axis_0->GetType(), axis_0->MutableOperator());

  // concat, to transpose_0
  auto concat_0 = fusion_rlt->AddInnerNode("concat_0", "ConcatV2");
  CHECK_INNER_NODE_CONDITION(concat_0 != nullptr, fusion_rlt);
  ret = concat_0->InsertInput("range_0", 0)
          .InsertInput("values_0", 0)
          .InsertInput("axis_0", 0)
          .InsertOutput("transpose_0", 0)
          .BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  ConcatParserParams(concat_0->GetName(), concat_0->GetType(), concat_0->MutableOperator(), fusion_rlt);

  // transpose to dynamicrnn_0
  auto transpose_0 = fusion_rlt->AddInnerNode("transpose_0", "Transpose");
  CHECK_INNER_NODE_CONDITION(transpose_0 != nullptr, fusion_rlt);
  ret = transpose_0->InsertInput(kInputFromFusionScope, 0)
          .InsertInput("concat_0", 0)
          .InsertOutput("DynamicRNN_0", 0)
          .BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, "rnn/transpose", transpose_0->GetType(), transpose_0->MutableOperator());

  // kernel to dynamicrnn_0
  auto weight_0 = fusion_rlt->AddInnerNode("weight_0", "Const");
  CHECK_INNER_NODE_CONDITION(weight_0 != nullptr, fusion_rlt);
  ret = weight_0->InsertOutput("DynamicRNN_0", 1).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, "rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel", weight_0->GetType(), weight_0->MutableOperator());

  // bias to dynamicrnn_0
  auto bias_0 = fusion_rlt->AddInnerNode("bias_0", "Const");
  CHECK_INNER_NODE_CONDITION(bias_0 != nullptr, fusion_rlt);
  ret = bias_0->InsertOutput("DynamicRNN_0", 2).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, "rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias", bias_0->GetType(), bias_0->MutableOperator());

  // dynamicrnn_0 to dynamicrnn_1
  auto DynamicRNN_0 = fusion_rlt->AddInnerNode("DynamicRNN_0", kScopeType);
  CHECK_INNER_NODE_CONDITION(DynamicRNN_0 != nullptr, fusion_rlt);
  ret = DynamicRNN_0->InsertInput("transpose_0", 0)
          .InsertInput("weight_0", 0)
          .InsertInput("bias_0", 0)
          .InsertOutput("DynamicRNN_1", 0)
          .BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);

  // kernel_1 to dynamicrnn_1
  auto weight_1 = fusion_rlt->AddInnerNode("weight_1", "Const");
  CHECK_INNER_NODE_CONDITION(weight_1 != nullptr, fusion_rlt);
  ret = weight_1->InsertOutput("DynamicRNN_1", 1).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, "rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel", weight_1->GetType(), weight_1->MutableOperator());

  // bias_1 to dynamicrnn_1
  auto bias_1 = fusion_rlt->AddInnerNode("bias_1", "Const");
  CHECK_INNER_NODE_CONDITION(bias_1 != nullptr, fusion_rlt);
  ret = bias_1->InsertOutput("DynamicRNN_1", 2).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, "rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias", bias_1->GetType(), bias_1->MutableOperator());

  // dynamicrnn_1 to dynamicrnn_2
  auto DynamicRNN_1 = fusion_rlt->AddInnerNode("DynamicRNN_1", kScopeType);
  CHECK_INNER_NODE_CONDITION(DynamicRNN_1 != nullptr, fusion_rlt);
  ret = DynamicRNN_1->InsertInput("DynamicRNN_0", 0)
          .InsertInput("weight_1", 0)
          .InsertInput("bias_1", 0)
          .InsertOutput("DynamicRNN_2", 0)
          .BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);

  // kernel_2 to dynamicrnn_2
  auto weight_2 = fusion_rlt->AddInnerNode("weight_2", "Const");
  CHECK_INNER_NODE_CONDITION(weight_2 != nullptr, fusion_rlt);
  ret = weight_2->InsertOutput("DynamicRNN_2", 1).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, "rnn/multi_rnn_cell/cell_2/basic_lstm_cell/kernel", weight_2->GetType(), weight_2->MutableOperator());

  // bias_2 to dynamicrnn_2
  auto bias_2 = fusion_rlt->AddInnerNode("bias_2", "Const");
  CHECK_INNER_NODE_CONDITION(bias_2 != nullptr, fusion_rlt);
  ret = bias_2->InsertOutput("DynamicRNN_2", 2).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, "rnn/multi_rnn_cell/cell_2/basic_lstm_cell/bias", bias_2->GetType(), bias_2->MutableOperator());

  // dynamicrnn_2 to transpose_1
  auto DynamicRNN_2 = fusion_rlt->AddInnerNode("DynamicRNN_2", kScopeType);
  CHECK_INNER_NODE_CONDITION(DynamicRNN_2 != nullptr, fusion_rlt);
  ret = DynamicRNN_2->InsertInput("DynamicRNN_1", 0)
          .InsertInput("weight_2", 0)
          .InsertInput("bias_2", 0)
          .InsertOutput("transpose_1", 0)
          .BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);

  // start to range
  auto start_1 = fusion_rlt->AddInnerNode("start_1", "Const");
  CHECK_INNER_NODE_CONDITION(start_1 != nullptr, fusion_rlt);
  ret = start_1->InsertOutput("range_1", 0).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, "rnn/range_1/start", start_1->GetType(), start_1->MutableOperator());

  // rank to range
  auto rank_1 = fusion_rlt->AddInnerNode("rank_1", "Const");
  CHECK_INNER_NODE_CONDITION(rank_1 != nullptr, fusion_rlt);
  ret = rank_1->InsertOutput("range_1", 1).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, "rnn/Rank_1", rank_1->GetType(), rank_1->MutableOperator());

  // delta to range
  auto delta_1 = fusion_rlt->AddInnerNode("delta_1", "Const");
  CHECK_INNER_NODE_CONDITION(delta_1 != nullptr, fusion_rlt);
  ret = delta_1->InsertOutput("range_1", 2).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, "rnn/range_1/delta", delta_1->GetType(), delta_1->MutableOperator());

  // range to concat
  auto range_1 = fusion_rlt->AddInnerNode("range_1", "Range");
  CHECK_INNER_NODE_CONDITION(range_1 != nullptr, fusion_rlt);
  ret = range_1->InsertInput("start_1", 0)
          .InsertInput("rank_1", 0)
          .InsertInput("delta_1", 0)
          .InsertOutput("concat_1", 0)
          .BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, "rnn/range_1", range_1->GetType(), range_1->MutableOperator());

  // values to concat
  auto values_1 = fusion_rlt->AddInnerNode("values_1", "Const");
  CHECK_INNER_NODE_CONDITION(values_1 != nullptr, fusion_rlt);
  ret = values_1->InsertOutput("concat_1", 0).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, "rnn/concat_2/values_0", values_1->GetType(), values_1->MutableOperator());

  // axis to concat
  auto axis_1 = fusion_rlt->AddInnerNode("axis_1", "Const");
  CHECK_INNER_NODE_CONDITION(axis_1 != nullptr, fusion_rlt);
  ret = axis_1->InsertOutput("concat_1", 1).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, "rnn/concat_2/axis", axis_1->GetType(), axis_1->MutableOperator());

  // concat, to transpose_0
  auto concat_1 = fusion_rlt->AddInnerNode("concat_1", "ConcatV2");
  CHECK_INNER_NODE_CONDITION(concat_1 != nullptr, fusion_rlt);
  ret = concat_1->InsertInput("range_1", 0)
          .InsertInput("values_1", 0)
          .InsertInput("axis_1", 0)
          .InsertOutput("transpose_1", 0)
          .BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  ConcatParserParams(concat_1->GetName(), concat_1->GetType(), concat_1->MutableOperator(), fusion_rlt);

  // transpose to out
  auto transpose_1 = fusion_rlt->AddInnerNode("transpose_1", "Transpose");
  CHECK_INNER_NODE_CONDITION(transpose_1 != nullptr, fusion_rlt);
  ret = transpose_1->InsertInput("DynamicRNN_2", 0)
          .InsertInput("concat_1", 0)
          .InsertOutput(kOutputToFusionScope, 0)
          .BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, "rnn/transpose_1", transpose_1->GetType(), transpose_1->MutableOperator());

  ret = fusion_rlt->CheckInnerNodesInfo();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  OP_LOGI(kOpType, "Set fusion multi-to-multi result successfully.");
  return;
}

void ScopeDynamicRNNPass::GenerateFusionResultForLTCRNN(const Scope* scope, FusionScopesResult* fusion_rlt) {
  OP_LOGD(kOpType, "Match DynamicRNN scope name is %s ", scope->Name().c_str());
  const std::unordered_map<std::string, ge::OperatorPtr> &nodes_map = scope->AllNodesMap();
  fusion_rlt->InsertInputs("transpose", {0, kFusionDisableIndex});
  fusion_rlt->InsertOutputs("transpose_1", {0});
  fusion_rlt->SetType(kScopeToMultiNodes);
  std::string scope_name = scope->Name();
  fusion_rlt->SetName(scope_name.substr(0, scope_name.length() - 1));
  fusion_rlt->SetDescription("");

  std::string w_in_node_name = GetNodeNameFromScope(nodes_map, "/kernel");
  std::string w_recurrent_in_node_name = GetNodeNameFromScope(nodes_map, "/recurrent_kernel");
  std::string b_in_node_name = GetNodeNameFromScope(nodes_map, "/bias");
  std::string transpose_perm_name = GetNodeNameFromScope(nodes_map, "transpose/perm");
  std::string transpose1_perm_name = GetNodeNameFromScope(nodes_map, "transpose_1/perm");
  std::string transpose_name = GetNodeNameFromScope(nodes_map, "/transpose");
  std::string transpose1_name = GetNodeNameFromScope(nodes_map, "/transpose_1");

  auto perm_0 = fusion_rlt->AddInnerNode("perm_0", "Const");
  CHECK_INNER_NODE_CONDITION(perm_0 != nullptr, fusion_rlt);
  Status ret = perm_0->InsertOutput("transpose_0", 1).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, transpose_perm_name, perm_0->GetType(), perm_0->MutableOperator());

  // transpose to dynamicrnn
  auto transpose_0 = fusion_rlt->AddInnerNode("transpose_0", "Transpose");
  CHECK_INNER_NODE_CONDITION(transpose_0 != nullptr, fusion_rlt);
  ret = transpose_0->InsertInput(kInputFromFusionScope, 0)
          .InsertInput("perm_0", 0)
          .InsertOutput("DynamicRNN_0", 0)
          .BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, transpose_name, transpose_0->GetType(), transpose_0->MutableOperator());

  // kernel and recurrent kernel to concat
  auto weight_0 = fusion_rlt->AddInnerNode("weight_0", "Const");
  CHECK_INNER_NODE_CONDITION(weight_0 != nullptr, fusion_rlt);
  ret = weight_0->InsertOutput("DynamicRNN_0", 1).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, w_in_node_name, weight_0->GetType(), weight_0->MutableOperator());

  auto weight_1 = fusion_rlt->AddInnerNode("weight_1", "Const");
  CHECK_INNER_NODE_CONDITION(weight_1 != nullptr, fusion_rlt);
  ret = weight_1->InsertOutput("DynamicRNN_0", 2).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, w_recurrent_in_node_name, weight_1->GetType(), weight_1->MutableOperator());

  // bias to dynamicrnn
  auto bias_0 = fusion_rlt->AddInnerNode("bias_0", "Const");
  CHECK_INNER_NODE_CONDITION(bias_0 != nullptr, fusion_rlt);
  ret = bias_0->InsertOutput("DynamicRNN_0", 3).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, b_in_node_name, bias_0->GetType(), bias_0->MutableOperator());

  // dynamicrnn, input x,w,b output transpose
  auto DynamicRNN_0 = fusion_rlt->AddInnerNode("DynamicRNN_0", "DynamicRNNV2");
  CHECK_INNER_NODE_CONDITION(DynamicRNN_0 != nullptr, fusion_rlt);
  ret = DynamicRNN_0->InsertInput("transpose_0", 0)
          .InsertInput("weight_0", 0)
          .InsertInput("weight_1", 0)
          .InsertInput("bias_0", 0)
          .InsertOutput("transpose_1", 0)
          .BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  auto rnnNode = DynamicRNN_0->MutableOperator();
  rnnNode->SetAttr("activation", "clip");
  rnnNode->SetAttr("recurrent_activation", "hard_sigmoid");
  rnnNode->SetAttr("gate_order", "ifco");

  // transpose to out
  auto perm_1 = fusion_rlt->AddInnerNode("perm_1", "Const");
  CHECK_INNER_NODE_CONDITION(perm_1 != nullptr, fusion_rlt);
  ret = perm_1->InsertOutput("transpose_1", 1).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, transpose1_perm_name, perm_1->GetType(), perm_1->MutableOperator());

  auto transpose_1 = fusion_rlt->AddInnerNode("transpose_1", "Transpose");
  CHECK_INNER_NODE_CONDITION(transpose_1 != nullptr, fusion_rlt);
  ret = transpose_1->InsertInput("DynamicRNN_0", 0)
          .InsertInput("perm_1", 0)
          .InsertOutput(kOutputToFusionScope, 0)
          .BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, transpose1_name, transpose_1->GetType(), transpose_1->MutableOperator());

  DynamicRNN_0->SetInputFormat("weight_input", "ND");
  DynamicRNN_0->SetInputFormat("weight_hidden", "ND");
  DynamicRNN_0->SetInputFormat("b", "ND");
  ret = fusion_rlt->CheckInnerNodesInfo();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  OP_LOGI(kOpType, "Set fusion multi-to-multi result successfully.");
  return;
}

void ScopeDynamicRNNPass::GenerateFusionResultForMultiNetease(const Scope* scope, FusionScopesResult* fusion_rlt){
  OP_LOGD(kOpType, "Match DynamicRNN scope name is %s ", scope->Name().c_str());
  const std::unordered_map<std::string, ge::OperatorPtr> &nodes_map = scope->AllNodesMap();
  fusion_rlt->InsertInputs("transpose", {0, kFusionDisableIndex});
  fusion_rlt->InsertOutputs("transpose_2", {0});
  fusion_rlt->SetType(kScopeToMultiNodes);
  std::string scope_name = scope->Name();
  fusion_rlt->SetName(scope_name.substr(0, scope_name.length() - 1));
  fusion_rlt->SetDescription("");
    
  std::string w_in_node_name = GetNodeNameFromScope(nodes_map, "/kernel");
  std::string w_recurrent_in_node_name = GetNodeNameFromScope(nodes_map, "/recurrent_kernel");
  std::string b_in_node_name = GetNodeNameFromScope(nodes_map, "/bias");
  std::string transpose_perm_name = GetNodeNameFromScope(nodes_map, "transpose/perm");
  std::string transpose2_perm_name = GetNodeNameFromScope(nodes_map, "transpose_2/perm");
  std::string transpose_name = GetNodeNameFromScope(nodes_map, "/transpose");
  std::string transpose2_name = GetNodeNameFromScope(nodes_map, "/transpose_2");

  // 添加小算子
  // start to range
  auto perm_0 = fusion_rlt->AddInnerNode("perm_0", "Const");
  CHECK_INNER_NODE_CONDITION(perm_0 != nullptr, fusion_rlt);
  Status ret = perm_0->InsertOutput("transpose_0", 1).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, transpose_perm_name, perm_0->GetType(), perm_0->MutableOperator());
    
  // transpose to dynamicrnn
  auto transpose_0 = fusion_rlt->AddInnerNode("transpose_0", "Transpose");
  CHECK_INNER_NODE_CONDITION(transpose_0 != nullptr, fusion_rlt);
  ret = transpose_0->InsertInput(kInputFromFusionScope, 0)
          .InsertInput("perm_0", 0)
          .InsertOutput("DynamicRNN_0", 0)
          .BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, transpose_name, transpose_0->GetType(), transpose_0->MutableOperator());

  // kernel and recurrent kernel to concat
  auto weight_0 = fusion_rlt->AddInnerNode("weight_0", "Const");
  CHECK_INNER_NODE_CONDITION(weight_0 != nullptr, fusion_rlt);
  ret = weight_0->InsertOutput("concat_0", 0).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, w_in_node_name, weight_0->GetType(), weight_0->MutableOperator());

  auto weight_1 = fusion_rlt->AddInnerNode("weight_1", "Const");
  CHECK_INNER_NODE_CONDITION(weight_1 != nullptr, fusion_rlt);
  ret = weight_1->InsertOutput("concat_0", 0).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, w_recurrent_in_node_name, weight_1->GetType(), weight_1->MutableOperator());
    
  auto concat_axis = fusion_rlt->AddInnerNode("concat_axis", "Const");
  CHECK_INNER_NODE_CONDITION(concat_axis != nullptr, fusion_rlt);
  ret = concat_axis->InsertOutput("concat_0", 1).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  TensorDesc batchDesc(ge::Shape({1}), FORMAT_ND, DT_INT32);
  int32_t* beginBatchData = nullptr;
  beginBatchData = new int32_t[1];
  *(beginBatchData) = 0;
  Tensor batchTensor(batchDesc, (uint8_t*)beginBatchData, sizeof(int32_t));
  auto constNodeOp = concat_axis->MutableOperator();
  if (constNodeOp != nullptr){
    constNodeOp->SetAttr("value", batchTensor);
  }
  delete[] beginBatchData;

  // input: kernel, recurrent kernel, output: dynamicrnn
  auto concat_0 = fusion_rlt->AddInnerNode("concat_0", "ConcatV2");
  ret = concat_0->InsertInput("weight_0", 0)
          .InsertInput("weight_1", 0)
          .InsertInput("concat_axis", 0)
          .InsertOutput("DynamicRNN_0", 1)
          .BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  ConcatParserParams(concat_0->GetName(), concat_0->GetType(), concat_0->MutableOperator(), fusion_rlt);

  // bias to dynamicrnn
  auto bias_0 = fusion_rlt->AddInnerNode("bias_0", "Const");
  CHECK_INNER_NODE_CONDITION(bias_0 != nullptr, fusion_rlt);
  ret = bias_0->InsertOutput("DynamicRNN_0", 2).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, b_in_node_name, bias_0->GetType(), bias_0->MutableOperator());

  // dynamicrnn, input x,w,b output transpose
  auto DynamicRNN_0 = fusion_rlt->AddInnerNode("DynamicRNN_0", kScopeType);
  CHECK_INNER_NODE_CONDITION(DynamicRNN_0 != nullptr, fusion_rlt);
  ret = DynamicRNN_0->InsertInput("transpose_0", 0)
          .InsertInput("concat_0", 0)
          .InsertInput("bias_0", 0)
          .InsertOutput("transpose_1", 0)
          .BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);

  // transpose to out
  auto perm_1 = fusion_rlt->AddInnerNode("perm_1", "Const");
  CHECK_INNER_NODE_CONDITION(perm_1 != nullptr, fusion_rlt);
  ret = perm_1->InsertOutput("transpose_1", 1).BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, transpose2_perm_name, perm_1->GetType(), perm_1->MutableOperator());

  auto transpose_1 = fusion_rlt->AddInnerNode("transpose_1", "Transpose");
  CHECK_INNER_NODE_CONDITION(transpose_1 != nullptr, fusion_rlt);
  ret = transpose_1->InsertInput("DynamicRNN_0", 0)
          .InsertInput("perm_1", 0)
          .InsertOutput(kOutputToFusionScope, 0)
          .BuildInnerNode();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  DynamicRNNPassParserParams(nodes_map, transpose2_name, transpose_1->GetType(), transpose_1->MutableOperator());

  ret = fusion_rlt->CheckInnerNodesInfo();
  CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  OP_LOGI(kOpType, "Set fusion multi-to-multi result successfully.");
  return;
}

void ScopeDynamicRNNPass::GenerateFusionResult(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt) {
  if (fusion_rlt == nullptr) {
    OP_LOGE(kOpType, "Input fusion_rlt is nullptr.");
    return;
  }
  for (auto& scope : scopes) {
    if (scope->SubType() == kRNNTwoTransMaximumType) {
      GenerateFusionResultForLTCRNN(scope, fusion_rlt);
      return;
    }
    if (scope->SubType() == kMultiRNNCell) {
      GenerateFusionResultForMultiLSTM(scope, fusion_rlt);
      return;
    }
    if (scope->SubType() == kLstmTwoTransType) {
      GenerateFusionResultForMultiNetease(scope, fusion_rlt);
      return;
    }
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
