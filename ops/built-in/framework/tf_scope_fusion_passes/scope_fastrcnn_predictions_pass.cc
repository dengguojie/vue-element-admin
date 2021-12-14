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
 * \file scope_fastrcnn_predictions_pass.cc
 * \brief
 */
#include "scope_fastrcnn_predictions_pass.h"

#include "op_log.h"
#include "register/scope/scope_fusion_pass_register.h"

namespace ge {
namespace {
const char* kScopeType = "FastrcnnPredictions";
const char* kScopeTypeFastrcnnPredictions = "FastrcnnPredictions";
const char* kOpType = "FastrcnnPredictions";
}  // namespace

std::vector<ScopeFusionPatterns> ScopeFastrcnnPredictionsPass::DefinePatterns() {
  std::vector<ScopeFusionPatterns> patterns_list;
  ScopeFusionPatterns pattern;
  GenScopePatterns(pattern);  // match fastrcnn Scope
  patterns_list.push_back(pattern);
  return patterns_list;
}

std::string ScopeFastrcnnPredictionsPass::PassName() {
  return std::string("ScopeFastrcnnPredistionsPass");
}

/**
 * @brief LastMatch for multiple scopes
 */
Status ScopeFastrcnnPredictionsPass::LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph>& scope_graph,
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
    if (scope->SubType() == kScopeTypeFastrcnnPredictions) {
      OP_LOGI(kOpType, "FastrcnnPredictions LastMatchScopesAndOPs match SubType.");
      ScopesResult result;
      std::vector<Scope*> result_scopes;
      result_scopes.push_back(scope);
      result.SetScopes(result_scopes);
      results.push_back(result);
    }
  }

  return (!(results.empty())) ? SUCCESS : FAILED;
}

void ScopeFastrcnnPredictionsPass::GenScopePatterns(ScopeFusionPatterns& patterns) const {
  // match batchnorm
  std::vector<ScopePattern*> batch;
  ScopePattern* fastPredictionPattern = new (std::nothrow) ScopePattern();
  if (fastPredictionPattern == nullptr) {
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }

  fastPredictionPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("TopKV2", 0, 2));  // ReverseV2 num is 1
  fastPredictionPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Where", 0, 3));   // ReverseV2 num is 1
  fastPredictionPattern->AddNodeOpTypeFeature(
      NodeOpTypeFeature("NonMaxSuppressionV2", 1));                               // NonMaxSuppressionV2 num is 1
  fastPredictionPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Less", 1));      // Less num is 1
  fastPredictionPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("LoopCond", 1));  // LoopCond num is 1

  fastPredictionPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("ExpandDims", -1, 0));          // Gather_2 num is -1
  fastPredictionPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("clip_boxes", -1, 0));          // Gather_2 num is -1
  fastPredictionPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("decode_bbox_target", -1, 0));  // Gather_2 num is -1

  fastPredictionPattern->SetSubType(kScopeTypeFastrcnnPredictions);
  OP_LOGI(kOpType, "Add GenScopePatterns fastPredictionPattern.");
  batch.push_back(fastPredictionPattern);
  patterns.push_back(batch);
}

void ScopeFastrcnnPredictionsPass::GenerateFusionResult(const std::vector<Scope*>& scopes,
                                                        FusionScopesResult* fusion_rlt) {
  if (fusion_rlt == nullptr) {
    OP_LOGE(kOpType, "Input fusion_rlt is nullptr.");
    return;
  }

  for (auto& scope : scopes) {
    // The upper call guarantees that the scope is not empty.
    if (scope->SubType() != kScopeTypeFastrcnnPredictions) {
      continue;
    }

    fusion_rlt->InsertInputs("fastrcnn_predictions/transpose", {0, kFusionDisableIndex});  // Input clip_boxes
    fusion_rlt->InsertInputs("fastrcnn_predictions/GatherNd", {kFusionDisableIndex, 0});   // Input clip_boxes
    fusion_rlt->InsertInputs("fastrcnn_predictions/strided_slice", {1, kFusionDisableIndex, kFusionDisableIndex,
                                                                    kFusionDisableIndex});  // Input fastrcnn_all_probs

    fusion_rlt->InsertOutputs("fastrcnn_predictions/TopKV2", {1, kFusionDisableIndex});  // final boxes
    fusion_rlt->InsertOutputs("fastrcnn_predictions/GatherNd", {0});                     // final probs
    fusion_rlt->InsertOutputs("fastrcnn_predictions/Add", {2});                          // final labels

    fusion_rlt->SetType(kScopeType);
    fusion_rlt->SetDescription("");

    std::string scope_name = scope->Name();
    fusion_rlt->SetName(scope_name.substr(0, scope_name.length() - 1));
  }

  OP_LOGI(kOpType, "fastrcnn_predictions Scope fusion success.");
  return;
}
}  // namespace ge
