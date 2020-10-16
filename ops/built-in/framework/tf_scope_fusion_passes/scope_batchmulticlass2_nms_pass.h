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
 * @file scope_batchmulticlass_nms_pass.h
 *
 * @brief scope fusion of BatchMultiClassNonMaxSuppression
 *
 * @version 1.0
 *
 */

#ifndef OPS_BUILT_IN_FRAMEWORK_TF_SCOPE_FUSION_PASSES_SCOPE_TO_NMS_PASS_H_
#define OPS_BUILT_IN_FRAMEWORK_TF_SCOPE_FUSION_PASSES_SCOPE_TO_NMS_PASS_H_

#include "register/scope/scope_fusion_pass_register.h"

namespace ge {
    class ScopeBatchMultiClassNMSPass : public ScopeBasePass {
        protected:
        std::string PassName() override {
        return std::string("ScopeBatchMultiClassNMSPass");
    }

    std::vector<ScopeFusionPatterns> DefinePatterns() override;
    Status LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph> &scope_graph,
    std::vector<ScopesResult> &results) override;
    void GenerateFusionResult(const std::vector<Scope *> &scopes, FusionScopesResult *fusion_rlt) override;

    private:
    ScopeFusionPatterns GenWhileScopePatterns();
    bool MatchedSubScopes(const Scope* root_scope, const std::vector<string> scopes2check) const;
    std::string to_string(const std::vector<Scope*> &scopes) const;

    private:
    const std::string kOpType = "BatchMultiClassNonMaxSuppression";
    const std::string kScopeResultType = "BatchMultiClassNonMaxSuppression";
    const std::string kScopeMap = "map";
};

} // namespace ge

#endif //OPS_BUILT_IN_FRAMEWORK_TF_SCOPE_FUSION_PASSES_SCOPE_TO_NMS_PASS_H_

