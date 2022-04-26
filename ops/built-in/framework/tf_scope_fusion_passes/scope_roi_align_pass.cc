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
 * \file scope_roi_align_pass.cc
 * \brief
 */
#include "scope_roi_align_pass.h"

#include "op_log.h"
#include "register/scope/scope_fusion_pass_register.h"

namespace ge {
static const char * const kScopeType = "ROIAlign";
static const char * const kROIAlignType = "ROIAlign";
static const char * const kOpType = "ROIAlign";

std::vector<ScopeFusionPatterns> ScopeROIAlignPass::DefinePatterns() {
  std::vector<ScopeFusionPatterns> patternsList;
  ScopeFusionPatterns patterns;
  GenScopePatterns(patterns);
  patternsList.push_back(patterns);
  return patternsList;
}

std::string ScopeROIAlignPass::PassName() {
  return std::string("ScopeROIAlignPass");
}

Status ScopeROIAlignPass::LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph>& scopeGraph,
                                                std::vector<ScopesResult>& results) {
  if (scopeGraph == nullptr) {
    OP_LOGE(kOpType, "Input params is nullptr.");
    return domi::PARAM_INVALID;
  }
  const ScopeTree* scopeTree = scopeGraph->GetScopeTree();
  const std::vector<Scope*>& scopes = scopeTree->GetAllScopes();
  for (auto& scope : scopes) {
    if (scope->SubType() == kROIAlignType && scope->Name().find("roi_align") != std::string::npos) {
      ScopesResult result;
      std::vector<Scope*> result_scopes;
      result_scopes.push_back(scope);
      result.SetScopes(result_scopes);
      results.push_back(result);
    }
  }
  return (!(results.empty())) ? SUCCESS : FAILED;
}

void ScopeROIAlignPass::GenScopePatterns(ScopeFusionPatterns& patterns) {
  std::vector<ScopePattern*> batch;
  ScopePattern* roiAlignPattern = new (std::nothrow) ScopePattern();
  if (roiAlignPattern == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  (void)roiAlignPattern->SetSubType(kROIAlignType);
  (void)roiAlignPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("AvgPool", 1, 0));
  (void)roiAlignPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Merge", -1, 0));
  (void)roiAlignPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("CropAndResize", 1, 0));
  batch.push_back(roiAlignPattern);
  patterns.push_back(batch);
}

void ScopeROIAlignPass::GenerateFusionResult(const std::vector<Scope*>& scopes, FusionScopesResult* fusionRlt) {
  if (fusionRlt == nullptr) {
    OP_LOGE(kOpType, "Input fusionRlt is nullptr.");
    return;
  }

  for (auto& scope : scopes) {
    if (scope->SubType() != kROIAlignType) {
      continue;
    }
    if ((scope->Name().find("task_7/cond_2/roi_align") != std::string::npos) ||
        (scope->Name().find("task_8/cond_2/roi_align") != std::string::npos)) {
      // featuremap
      fusionRlt->InsertInputs("roi_align/crop_and_resize/Shape/Switch", {0, kFusionDisableIndex});
      // rois
      fusionRlt->InsertInputs("roi_align/Shape/Switch", {1, kFusionDisableIndex});
      // output
      fusionRlt->InsertOutputs("AvgPool", {0});
      fusionRlt->SetType(kScopeType);
      fusionRlt->SetDescription("");
    } else {
      // featuremap
      fusionRlt->InsertInputs("crop_and_resize/Shape/Switch", {0, kFusionDisableIndex});
      // rois
      fusionRlt->InsertInputs("Shape", {1});
      fusionRlt->InsertInputs("crop_and_resize/transform_fpcoor_for_tf/split", {kFusionDisableIndex, 1});
      // output
      fusionRlt->InsertOutputs("AvgPool", {0});
      fusionRlt->SetType(kScopeType);
      fusionRlt->SetDescription("");
    }
    std::string scopeName = scope->Name();
    OP_LOGI(kOpType, "ROIalign Scope name in TF is: %s", scopeName.c_str());
    fusionRlt->SetName(scopeName.substr(0, scopeName.length() - 1));
  }
  OP_LOGI(kOpType, "ROIalign Scope fusion success.");
  return;
}
}  // namespace ge
