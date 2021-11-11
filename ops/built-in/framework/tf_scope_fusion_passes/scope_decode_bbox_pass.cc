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
 * \file scope_decode_bbox_pass.cc
 * \brief
 */
#include "scope_decode_bbox_pass.h"

#include <string>
#include <cstring>

#include "op_log.h"
#include "register/scope/scope_fusion_pass_register.h"

namespace ge {
namespace {
const char* const kScopeType = "DecodeBbox";
const char* const kScopeTypeDecodeBbox = "DecodeBbox";
const char* const kOpType = "DecodeBbox";
const char* const decode_box2_name = "transpose";
}  // namespace

std::vector<ScopeFusionPatterns> ScopeDecodeBboxPass::DefinePatterns() {
  std::vector<ScopeFusionPatterns> patterns_list;
  ScopeFusionPatterns pattern;
  GenScopePatterns(pattern);  // match batchnorm Scope
  patterns_list.push_back(pattern);
  return patterns_list;
}

std::string ScopeDecodeBboxPass::PassName() {
  return std::string("ScopeDecodeBboxPass");
}

/**
 * @brief LastMatch for multiple scopes
 */
Status ScopeDecodeBboxPass::LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph>& scope_graph,
                                                  std::vector<ScopesResult>& results) {
  OP_LOGI(kOpType, "LastMatchScopesAndOPs start.");
  if (scope_graph == nullptr) {
    OP_LOGE(kOpType, "Input params is nullptr.");
    return domi::PARAM_INVALID;
  }
  const ScopeTree* scope_tree = scope_graph->GetScopeTree();
  if (scope_tree == nullptr) {
    OP_LOGE(kOpType, "Scope tree is nullptr.");
    return domi::PARAM_INVALID;
  }
  const std::vector<Scope*>& scopes = scope_tree->GetAllScopes();

  for (auto& scope : scopes) {
    // Class ScopeTree guarantees scope is not empty.
    if (scope->SubType() == kScopeTypeDecodeBbox) {
      OP_LOGI(kOpType, "DecodeBbox LastMatchScopesAndOPs match SubType %s.", scope->Name().c_str());
      ScopesResult result;
      std::vector<Scope*> result_scopes;
      result_scopes.push_back(scope);
      result.SetScopes(result_scopes);
      results.push_back(result);
    }
  }

  return (!(results.empty())) ? SUCCESS : FAILED;
}

void ScopeDecodeBboxPass::GenScopePatterns(ScopeFusionPatterns& patterns) {
  OP_LOGI(kOpType, "GenScopePatterns start.");
  // match batchnorm
  std::vector<ScopePattern*> batch;
  ScopePattern* decodeBboxPattern = new (std::nothrow) ScopePattern();
  if (decodeBboxPattern == nullptr) {
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  decodeBboxPattern->SetSubType(kScopeTypeDecodeBbox);

  decodeBboxPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Reshape", 0, 3));   // Reshape num is 3
  decodeBboxPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Split", 0, 2));     // Split num is 2
  decodeBboxPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Minimum", 1, 0));   // Minimum num is 1
  decodeBboxPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Add", 0, 3));       // Add num is 3
  decodeBboxPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("ConcatV2", 1, 0));  // ConcatV2 num is 1
  decodeBboxPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Sub", 0, 2));       // Sub num is 2

  decodeBboxPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Greater", -1, 0));       // Squeeze num is -1
  decodeBboxPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Squeeze", -1, 0));       // Squeeze num is -1
  decodeBboxPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Gather_2", -1, 0));      // Gather_2 num is -1
  decodeBboxPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("TopKV2", -1, 0));        // TopKV2 num is -1
  decodeBboxPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("boolean_mask", -1, 0));  // boolean_mask num is -1

  OP_LOGI(kOpType, "Add GenScopePatterns DecodeBbox.");
  batch.push_back(decodeBboxPattern);
  patterns.push_back(batch);
}

void ScopeDecodeBboxPass::GenerateFusionResult(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt) {
  OP_LOGI(kOpType, "GenerateFusionResult start.");
  if (fusion_rlt == nullptr) {
    OP_LOGE(kOpType, "Input fusion_rlt is nullptr.");
    return;
  }
  for (auto& scope : scopes) {
    // The upper call guarantees that the scope is not empty.
    if (scope->SubType() != kScopeTypeDecodeBbox) {
      continue;
    }
    bool decode_optype = false;
    const std::unordered_map<std::string, ge::OperatorPtr>& nodes_map = scope->AllNodesMap();
    for (auto& it : nodes_map) {
      auto node_def = it.second;
      if (node_def->GetName().find(decode_box2_name) != string::npos) {
        decode_optype = true;
        break;
      }
    }

    if (decode_optype) {
      OP_LOGI(kOpType, "fusion type 1.");
      fusion_rlt->InsertInputs("transpose", {0, kFusionDisableIndex});    // Input index 1 of decode_bbox
      fusion_rlt->InsertInputs("transpose_1", {1, kFusionDisableIndex});  // Input index 0 of decode_bbox

      fusion_rlt->InsertOutputs("transpose_2", {0});  // Output index 0 of decode_bbox
    } else {
      OP_LOGI(kOpType, "fusion type 2.");
      fusion_rlt->InsertInputs("Shape", {1});                           // Input index 0 of decode_bbox
      fusion_rlt->InsertInputs("Reshape_1", {1, kFusionDisableIndex});  // Input index 0 of decode_bbox
      fusion_rlt->InsertInputs("Reshape", {0, kFusionDisableIndex});    // Input index 1 of decode_bbox

      fusion_rlt->InsertOutputs("Reshape_2", {0});  // Output index 0 of decode_bbox
    }

    fusion_rlt->SetType(kScopeType);
    fusion_rlt->SetDescription("");

    std::string scope_name = scope->Name();
    fusion_rlt->SetName(scope_name.substr(0, scope_name.length() - 1));
  }
  return;
}
}  // namespace ge
