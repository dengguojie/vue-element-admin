/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
 * \file scope_normalize_bbox_pass.cpp
 * \brief scope fusion of NormalizeBBox
 */
#include "scope_normalize_bbox_pass.h"

#include "op_log.h"

namespace ge {
std::vector<ScopeFusionPatterns> ScopeNormalizeBBoxPass::DefinePatterns() {
  std::vector<ScopeFusionPatterns> patterns;
  patterns.push_back(GenWhileScopePatterns());
  return patterns;
}

Status ScopeNormalizeBBoxPass::LastMatchScopesAndOPs(shared_ptr<ScopeGraph>& scope_graph,
                                                     std::vector<ScopesResult>& results) {
  if (!scope_graph) {
    OP_LOGE(kOpType.c_str(), "Input params is nullptr.");
    return domi::PARAM_INVALID;
  }

  const ScopeTree* scope_tree = scope_graph->GetScopeTree();
  if (scope_tree == nullptr) {
    OP_LOGE(kOpType.c_str(), "Scope tree is nullptr.");
    return domi::PARAM_INVALID;
  }

  const std::vector<Scope*>& scopes = scope_tree->GetAllScopes();
  for (auto& scope : scopes) {
    if (scope->LastName() == kScopeWhile) {
      OP_LOGD(kOpType.c_str(), "scope name:%s", scope->Name().c_str());
      auto father_scope = scope->GetFatherScope();
      if (father_scope == nullptr) {
        continue;
      }

      if (!MatchedSubScopes(father_scope)) {
        continue;
      }

      ScopesResult result;
      std::vector<Scope*> result_scopes;
      result_scopes.push_back(const_cast<Scope*>(father_scope));
      result.SetScopes(result_scopes);
      results.push_back(result);
    }
  }

  return (!(results.empty())) ? SUCCESS : FAILED;
}

void ScopeNormalizeBBoxPass::GenerateFusionResult(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt) {
  if (fusion_rlt == nullptr) {
    OP_LOGE(kOpType.c_str(), "Input fusion_rlt is nullptr.");
    return;
  }
  fusion_rlt->InsertInputs("map/Shape", {0});                              // input normalized_boxes
  fusion_rlt->InsertInputs("map/TensorArrayUnstack_1/Shape", {1});         // input shape_hw
  fusion_rlt->InsertOutputs("TensorArrayStack/TensorArrayGatherV3", {0});  // output y

  OP_LOGD(kOpType.c_str(), "all scopes:%s", to_string(scopes).c_str());
  fusion_rlt->SetName(kScopeResultType);
  for (auto& scope : scopes) {
    // The upper call guarantees that the scope is not empty.
    auto fusion_node_name = scope->Name();
    (void)fusion_node_name.erase(fusion_node_name.length() - 1, 1);
    fusion_rlt->SetName(fusion_node_name);
    OP_LOGI(kOpType.c_str(), "scope name:%s", fusion_node_name.c_str());
    break;
  }

  fusion_rlt->SetType(kScopeResultType);  // op Type NormalizeBBox
  fusion_rlt->SetDescription("");

  OP_LOGI(kOpType.c_str(), "%s Scope fusion success.", kScopeResultType.c_str());
  return;
}

ScopeFusionPatterns ScopeNormalizeBBoxPass::GenWhileScopePatterns() {
  ScopePattern* while_cell = new (std::nothrow) ScopePattern();
  if (while_cell == nullptr) {
    OP_LOGE(kOpType.c_str(), "Alloc an object failed.");
    return ScopeFusionPatterns();
  }

  (void)while_cell->SetSubType("while");
  (void)while_cell->AddScopeFeature(ScopeFeature("", -1, "while"));
  static const size_t mul_num = 4;
  (void)while_cell->AddNodeOpTypeFeature(NodeOpTypeFeature("Mul", mul_num));

  ScopeFusionPatterns while_scope_pattern = {{while_cell}};
  return while_scope_pattern;
}

string ScopeNormalizeBBoxPass::to_string(const std::vector<Scope*>& scopes) const {
  string result;
  for (auto& scope : scopes) {
    result += scope->Name();
    result += " ";
  }
  return result;
}

bool ScopeNormalizeBBoxPass::MatchedSubScopes(const Scope* root_scope) const {
  const std::vector<string> scopes2check = {"while/", "ToNormalizedCoordinates/", "Scale/"};

  string full_name;
  auto root = root_scope;
  for (auto& scope_name : scopes2check) {
    full_name = root->Name();
    full_name += scope_name;
    auto sub_scope = root->GetSubScope(full_name);
    if (sub_scope == nullptr) {
      OP_LOGD(kOpType.c_str(), "Get sub scope:%s failed, %s's sub scopes:%s", full_name.c_str(), root->Name().c_str(),
              to_string(root->GetAllSubScopes()).c_str());
      return false;
    }
    root = sub_scope;
  }
  OP_LOGI(kOpType.c_str(), "MatchedSubScopes:%s success.", root_scope->Name().c_str());
  return true;
}
}  // namespace ge
