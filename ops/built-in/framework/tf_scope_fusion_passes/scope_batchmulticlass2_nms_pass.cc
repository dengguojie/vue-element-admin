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
 * @file scope_batchmulticlass2_nms_pass.cpp
 *
 * @brief scope fusion of ToAbsoluteBBox
 *
 * @version 1.0
 *
 */

#include "scope_batchmulticlass2_nms_pass.h"
#include "op_log.h"
#include "register/scope/scope_fusion_pass_register.h"

namespace ge {
    using namespace std;

    std::vector<ScopeFusionPatterns> ScopeBatchMultiClassNMSPass::DefinePatterns() {
        vector<ScopeFusionPatterns> patterns;
        patterns.push_back(GenWhileScopePatterns());
        return patterns;
    }

    Status ScopeBatchMultiClassNMSPass::LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph> &scope_graph,
    std::vector<ScopesResult> &results) {
        if (!scope_graph) {
            OP_LOGE(kOpType.c_str(), "Input params is nullptr.");
            return domi::PARAM_INVALID;
        }

        const ScopeTree *scope_tree = scope_graph->GetScopeTree();
        if (scope_tree == nullptr) {
            OP_LOGE(kOpType.c_str(), "Scope tree is nullptr.");
            return domi::PARAM_INVALID;
        }

        std::vector<Scope *> scopes = scope_tree->GetAllScopes();
        for (auto &scope : scopes) {
            if (scope->Name() == "rootmap/") {
                continue;
            }
            if (scope->LastName() == kScopeMap) {
                const Scope *fatherScope = scope->GetFatherScope();
                if (fatherScope == nullptr) {
                    continue;
                }

                if (!MatchedSubScopes(fatherScope, {"map/", "while/", "MultiClassNonMaxSuppression/"})) {
                    continue;
                }
                OP_LOGI(kOpType.c_str(), "find scope name: %s", fatherScope->Name().c_str());
                ScopesResult result;
                std::vector<Scope *> result_scopes;
                result_scopes.push_back(const_cast<Scope *>(fatherScope));
                result.SetScopes(result_scopes);
                results.push_back(result);

            }
        }

        return (!(results.empty())) ? SUCCESS : FAILED;
    }

    void ScopeBatchMultiClassNMSPass::GenerateFusionResult(const std::vector<Scope *> &scopes,
    FusionScopesResult *fusion_rlt) {
        if (fusion_rlt == nullptr) {
            OP_LOGE(kOpType.c_str(), "Input fusion_rlt is nullptr.");
            return;
        }
        for (auto &scope : scopes) {
            // do insert input 0 boxes like map/TensorArrayUnstack/Shape
            std::string scopeInputName0 = scope->Name() + "map/TensorArrayUnstack/Shape";
            fusion_rlt->InsertInputs(scopeInputName0, {0});

            // do insert input 1 scores like map/TensorArrayUnstack_1/Shape
            std::string scopeInputName1 = scope->Name() + "map/TensorArrayUnstack_1/Shape";
            fusion_rlt->InsertInputs(scopeInputName1, {1});

            // do insert input 2 clip_to_window like map/TensorArrayUnstack_3/Shape
            if (MatchedSubScopes(scope, {"map/", "TensorArrayUnstack_3/"})) {
                std::string scopeInputName2 = scope->Name() + "map/TensorArrayUnstack_3/Shape";
                fusion_rlt->InsertInputs(scopeInputName2, {2});
                if (MatchedSubScopes(scope, {"map/", "TensorArrayUnstack_4/"}) && !MatchedSubScopes(scope, {"ones/"})) {
                    std::string scopeInputName3 = scope->Name() + "map/TensorArrayUnstack_4/Shape";
                    fusion_rlt->InsertInputs(scopeInputName3, {3});

                }
            }
            // do insert outputs
            if (MatchedSubScopes(scope, {"map/", "TensorArrayStack/"})) {
                std::string scopeOutputName0 = scope->Name() + "map/TensorArrayStack/TensorArrayGatherV3";
                fusion_rlt->InsertOutputs(scopeOutputName0, {0});
            }
            if (MatchedSubScopes(scope, {"map/", "TensorArrayStack_1/"})) {
                std::string scopeOutputName0 = scope->Name() + "map/TensorArrayStack_1/TensorArrayGatherV3";
                fusion_rlt->InsertOutputs(scopeOutputName0, {1});
            }
            if (MatchedSubScopes(scope, {"map/", "TensorArrayStack_2/"})) {
                std::string scopeOutputName0 = scope->Name() + "map/TensorArrayStack_2/TensorArrayGatherV3";
                fusion_rlt->InsertOutputs(scopeOutputName0, {2});
            }
            if (MatchedSubScopes(scope, {"map/", "TensorArrayStack_4/"})) {
                std::string scopeOutputName0 = scope->Name() + "map/TensorArrayStack_4/TensorArrayGatherV3";
                fusion_rlt->InsertOutputs(scopeOutputName0, {3});
            }
            fusion_rlt->SetType(kOpType);
            fusion_rlt->SetDescription("");
            std::string scopeName = scope->Name();
            fusion_rlt->SetName(scopeName.substr(0, scopeName.length() - 1));
        }
        OP_LOGI(kOpType.c_str(), "ScopeBatchMultiClassNonMaxSuppressionPass Scope fusion success.");
        return;
    }

    ScopeFusionPatterns ScopeBatchMultiClassNMSPass::GenWhileScopePatterns() {
        ScopePattern *while_cell = new(std::nothrow) ScopePattern();
        if (while_cell == nullptr) {
            OP_LOGE(kOpType.c_str(), "Alloc an object failed.");
            return ScopeFusionPatterns();
        }

        while_cell->SetSubType(kScopeResultType);
        while_cell->AddScopeFeature(ScopeFeature("", 1, "", "map"));
        while_cell->AddScopeFeature(ScopeFeature("", 1, "", "while"));
        while_cell->AddScopeFeature(ScopeFeature("", 1, "", "MultiClassNonMaxSuppression"));

        ScopeFusionPatterns while_scope_pattern = {{while_cell}};
        return while_scope_pattern;
    }

    string ScopeBatchMultiClassNMSPass::to_string(const vector<Scope *> &scopes) const {
        string result;
        for (auto &scope : scopes) {
            result += scope->Name();
            result += " ";
        }
        return result;
    }

    bool ScopeBatchMultiClassNMSPass::MatchedSubScopes(const Scope *root_scope,
                                              const vector<string> scopes2check) const {
        string full_name;
        auto root = root_scope;
        for (auto &scope_name : scopes2check) {
            full_name = root->Name();
            full_name += scope_name;
            auto sub_scope = root->GetSubScope(full_name);
            if (sub_scope == nullptr) {
                OP_LOGI(kOpType.c_str(),
                "Get sub scope:%s failed, %s's sub scopes:%s",
                full_name.c_str(),
                root->Name().c_str(),
                to_string(root->GetAllSubScopes()).c_str());
                return false;
            }
            root = sub_scope;
        }

        OP_LOGI(kOpType.c_str(), "MatchedSubScopes:%s success.",
        root_scope->Name().c_str());
        return true;
    }

} // namespace ge

