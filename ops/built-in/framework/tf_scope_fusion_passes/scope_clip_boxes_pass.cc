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
 * \file scope_clip_boxes_pass.cc
 * \brief
 */
#include "scope_clip_boxes_pass.h"

#include "op_log.h"


namespace ge {
namespace {
const char * const kScopeType = "ClipBoxes";
const char * const kScopeTypeClipBoxes = "ClipBoxes";
const char * const kOpType = "ClipBoxes";
}  // namespace

std::vector<ScopeFusionPatterns> ScopeClipBoxesPass::DefinePatterns() {
  std::vector<ScopeFusionPatterns> patterns_list;
  ScopeFusionPatterns pattern;
  GenScopePatterns(pattern);  // match batchnorm Scope
  patterns_list.push_back(pattern);
  return patterns_list;
}

std::string ScopeClipBoxesPass::PassName() {
  return std::string("ScopeClipBoxesPass");
}

/**
 * @brief LastMatch for multiple scopes
 */
Status ScopeClipBoxesPass::LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph>& scope_graph,
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
  const std::vector<Scope*>& scopes = scope_tree->GetAllScopes();

  for (auto& scope : scopes) {
    // Class ScopeTree guarantees scope is not empty.
    if (scope->SubType() == kScopeTypeClipBoxes) {
      OP_LOGI(kOpType, "ClipBoxes LastMatchScopesAndOPs match SubType.");
      ScopesResult result;
      std::vector<Scope*> result_scopes;
      if (scope->Name().find("generate_rpn_proposals") != std::string::npos) {
        continue;
      }
      result_scopes.push_back(scope);
      result.SetScopes(result_scopes);
      results.push_back(result);
    }
  }

  return (!(results.empty())) ? SUCCESS : FAILED;
}

void ScopeClipBoxesPass::GenScopePatterns(ScopeFusionPatterns& patterns) {
  // match batchnorm
  std::vector<ScopePattern*> batch;
  ScopePattern* clipBoxesPattern = new (std::nothrow) ScopePattern();
  if (clipBoxesPattern == nullptr) {
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  (void)clipBoxesPattern->SetSubType(kScopeTypeClipBoxes);

  (void)clipBoxesPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Minimum", 1, 0));        // Minimum num is 1
  (void)clipBoxesPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Maximum", 1, 0));        // Maximum num is 1
  (void)clipBoxesPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Tile", 1, 0));           // Tile num is 1
  (void)clipBoxesPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("ReverseV2", 1, 0));      // ReverseV2 num is 1
  (void)clipBoxesPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Gather_2", -1, 0));      // Gather_2 num is -1
  (void)clipBoxesPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("TopKV2", -1, 0));        // TopKV2 num is -1
  (void)clipBoxesPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Reshape_2", -1, 0));     // Reshape_2 num is -1
  (void)clipBoxesPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Split", -1, 0));         // Split num is -1
  (void)clipBoxesPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Greater", -1, 0));       // Greater num is -1
  (void)clipBoxesPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Squeeze", -1, 0));       // Squeeze num is -1
  (void)clipBoxesPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Gather", -1, 0));        // Gather num is -1
  (void)clipBoxesPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("boolean_mask", -1, 0));  // boolean_mask num is -1
  (void)clipBoxesPattern->AddNodeOpTypeFeature(
      NodeOpTypeFeature("decode_bbox_target", -1, 0));  // decode_bbox_target num is -1

  OP_LOGI(kOpType, "Add GenScopePatterns ClipBoxes.");
  batch.push_back(clipBoxesPattern);
  patterns.push_back(batch);
}

void ScopeClipBoxesPass::GenerateFusionResult(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt) {
  if (fusion_rlt == nullptr) {
    OP_LOGE(kOpType, "Input fusion_rlt is nullptr.");
    return;
  }

  for (auto& scope : scopes) {
    // The upper call guarantees that the scope is not empty.
    if (scope->SubType() != kScopeTypeClipBoxes) {
      continue;
    }
    fusion_rlt->InsertInputs("clip_boxes/Maximum", {0, kFusionDisableIndex});    // Input index 0 and 1 : x and h
    fusion_rlt->InsertInputs("clip_boxes/ReverseV2", {1, kFusionDisableIndex});  // Input index 1 : c
    fusion_rlt->InsertOutputs("clip_boxes/fastrcnn_all_boxes", {0});             // Output index 0 : ct
    fusion_rlt->SetType(kScopeType);
    fusion_rlt->SetDescription("");

    std::string scope_name = scope->Name();
    fusion_rlt->SetName(scope_name.substr(0, scope_name.length() - 1));
    OP_LOGI(kOpType, "ClipBoxes Scope fusion success.");
    return;
  }
}
}  // namespace ge
