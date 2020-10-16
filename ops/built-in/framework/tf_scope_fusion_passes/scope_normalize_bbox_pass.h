/**
 * Copyright (C)  2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file scope_to_absolute_bbox_pass.h
 *
 * @brief scope fusion of NormalizeBBox
 *
 * @version 1.0
 *
 */

#ifndef OPS_BUILT_IN_FRAMEWORK_TF_SCOPE_FUSION_PASSES_SCOPE_NORMALIZE_BBOX_PASS_H_
#define OPS_BUILT_IN_FRAMEWORK_TF_SCOPE_FUSION_PASSES_SCOPE_NORMALIZE_BBOX_PASS_H_

#include "external/register/scope/scope_fusion_pass_register.h"

namespace ge {
class ScopeNormalizeBBoxPass : public ScopeBasePass {
 protected:
  std::string PassName() override {
    return std::string("ScopeNormalizeBBoxPass");
  }

  std::vector<ScopeFusionPatterns> DefinePatterns() override;
  Status LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph> &scope_graph,
                               std::vector<ScopesResult> &results) override;
  void GenerateFusionResult(const std::vector<Scope *> &scopes,
                           FusionScopesResult *fusion_rlt) override;

 private:
  ScopeFusionPatterns GenWhileScopePatterns();
  bool MatchedSubScopes(const Scope* root_scope) const;
  std::string to_string(const std::vector<Scope*> &scopes) const;

 private:
  const std::string kOpType = "NormalizeBBox";
  const std::string kScopeResultType = "NormalizeBBox";
  const std::string kScopeWhile = "while";
};

} // namespace ge

#endif //OPS_BUILT_IN_FRAMEWORK_TF_SCOPE_FUSION_PASSES_SCOPE_NORMALIZE_BBOX_PASS_H_
