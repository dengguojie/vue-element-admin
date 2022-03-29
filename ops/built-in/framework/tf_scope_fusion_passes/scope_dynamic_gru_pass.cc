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
#include "scope_dynamic_gru_pass.h"

#include "op_log.h"

namespace ge {
static const char* const kScopeType = "DynamicGRUV2";
static const char* const kWhileType = "while";
static const char* const kOpType = "DynamicGRU";
static const char* const kTranspose = "transpose";
static const char* const kGruFwWhileNoTransposeType = "gru_fw_while_no_transpose";
static const char* const kGruBwWhileNoTransposeType = "gru_bw_while_no_transpose";


std::vector<ScopeFusionPatterns> ScopeDynamicGRUPass::DefinePatterns() {
  std::vector<ScopeFusionPatterns> patterns_list;
  ScopeFusionPatterns patterns;
  GenScopePatterns(patterns);
  patterns_list.push_back(patterns);
  return patterns_list;
}

std::string ScopeDynamicGRUPass::PassName() {
  return std::string("ScopeDynamicGRUPass");
}

void ScopeDynamicGRUPass::GenScopePatterns(ScopeFusionPatterns& patterns) {
  std::vector<ScopePattern*> batch1;
  ScopePattern* fw_while_no_transpose = new (std::nothrow) ScopePattern();
  if (fw_while_no_transpose == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch1);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  fw_while_no_transpose->SetSubType(kGruFwWhileNoTransposeType);
  // The number of AddV2 is a multiple of 3.
  fw_while_no_transpose->AddNodeOpTypeFeature(NodeOpTypeFeature("AddV2", 5, 0));
  // The number of Mul is a multiple of 3.
  fw_while_no_transpose->AddNodeOpTypeFeature(NodeOpTypeFeature("Mul", 3, 0));
  // The number of Tanh is a multiple of 1.
  fw_while_no_transpose->AddNodeOpTypeFeature(NodeOpTypeFeature("Tanh", 1, 0));
  // The number of Transpose is -1.
  fw_while_no_transpose->AddNodeOpTypeFeature(NodeOpTypeFeature("Transpose", -1, 0));

  batch1.push_back(fw_while_no_transpose);

  ScopePattern* bw_while_no_transpose = new (std::nothrow) ScopePattern();
  if (bw_while_no_transpose == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch1);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  bw_while_no_transpose->SetSubType(kGruFwWhileNoTransposeType);
  // The number of AddV2 is a multiple of 3.
  bw_while_no_transpose->AddNodeOpTypeFeature(NodeOpTypeFeature("AddV2", 5, 0));
  // The number of Mul is a multiple of 3.
  bw_while_no_transpose->AddNodeOpTypeFeature(NodeOpTypeFeature("Mul", 3, 0));
  // The number of ReverseV2 is -1.
  bw_while_no_transpose->AddNodeOpTypeFeature(NodeOpTypeFeature("ReverseV2", -1, 0));
  // The number of Tanh is a multiple of 1.
  bw_while_no_transpose->AddNodeOpTypeFeature(NodeOpTypeFeature("Tanh", 1, 0));
  // The number of Transpose is -1.
  bw_while_no_transpose->AddNodeOpTypeFeature(NodeOpTypeFeature("Transpose", -1, 0));

  batch1.push_back(bw_while_no_transpose);
  patterns.push_back(batch1);
}

Status ScopeDynamicGRUPass::LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph>& scope_graph,
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
  std::set<string> support_types = {kGruFwWhileNoTransposeType, kGruBwWhileNoTransposeType};
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

void ScopeDynamicGRUPass::GenerateFusionResult(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt) {
  if (fusion_rlt == nullptr) {
    OP_LOGE(kOpType, "Input fusion_rlt is nullptr.");
    return;
  }
  for (auto& scope : scopes) {
    std::string w_in_node_name;
    std::string w_hidden_in_node_name;
    std::string b_in_node_name;
    std::string b_hidden_in_node_name;
    OP_LOGD(kOpType, "Match DynamicGRU scope name is %s ", scope->Name().c_str());
    const std::unordered_map<std::string, ge::OperatorPtr>& nodes_map = scope->AllNodesMap();
    for (auto& it : nodes_map) {
      auto node_def = it.second;
      std::string sub_node_name = node_def->GetName().c_str();
      if (sub_node_name.find("ReadVariableOp_1/Enter") != string::npos) {
        w_in_node_name = sub_node_name;
      }
      if (sub_node_name.find("ReadVariableOp_4/Enter") != string::npos) {
        w_hidden_in_node_name = sub_node_name;
      }
      if (sub_node_name.find("ReadVariableOp_00/Enter") != string::npos) {
        b_in_node_name = sub_node_name;
      }
      if (sub_node_name.find("ReadVariableOp_01/Enter") != string::npos) {
        b_hidden_in_node_name = sub_node_name;
      }
      if (!w_in_node_name.empty() && !b_in_node_name.empty() && !w_hidden_in_node_name.empty() && !b_hidden_in_node_name.empty()) {
        OP_LOGD(kOpType, "DynamicGRU w input node name is %s ", w_in_node_name.c_str());
        OP_LOGD(kOpType, "DynamicGRU w hidden input node name is %s ", w_hidden_in_node_name.c_str());
        OP_LOGD(kOpType, "DynamicGRU b input node name is %s ", b_in_node_name.c_str());
        OP_LOGD(kOpType, "DynamicGRU b hidden input node name is %s ", b_hidden_in_node_name.c_str());
        break;
      }
    }

    if (scope->SubType() == kGruFwWhileNoTransposeType || scope->SubType() == kGruBwWhileNoTransposeType) {
      fusion_rlt->InsertInputs("TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3",
                               {kFusionDisableIndex, kFusionDisableIndex, 0, kFusionDisableIndex});  // Input 0 : x
      fusion_rlt->InsertInputs(w_in_node_name, {1});  // Input index 1 : w
      fusion_rlt->InsertInputs(w_hidden_in_node_name, {2});  // Input index 2 : w_hidden
      fusion_rlt->InsertOutputs("TensorArrayStack/TensorArrayGatherV3", {0});  // Output index 0 : outputs
    }

    if (!b_in_node_name.empty() && !b_hidden_in_node_name.empty()) {
      fusion_rlt->InsertInputs(b_in_node_name, {3});  // Input 4 : init_h
      fusion_rlt->InsertInputs(b_hidden_in_node_name, {4});  // Input 5 : init_c
    }

    fusion_rlt->SetType(kScopeType);
    fusion_rlt->SetDescription("");
    std::string scope_name = scope->Name();
    fusion_rlt->SetName(scope_name.substr(0, scope_name.length() - 1));
  }
  OP_LOGI(kOpType, "DynamicGRU Scope fusion success.");
  return;
}
}  // namespace ge
