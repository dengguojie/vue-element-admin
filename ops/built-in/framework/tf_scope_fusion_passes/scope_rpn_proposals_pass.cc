/**
 * Copyright 2020 Huawei Technologies Co., Ltd

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

#include "scope_rpn_proposals_pass.h"
#include "op_log.h"
#include "external/register/scope/scope_fusion_pass_register.h"

namespace ge {
    namespace {
        static const char *const kScopeType = "RpnProposals";
        static const char *const kRpnProposalsType = "rpn_proposals";
        static const char *const kOpType = "RpnProposals";
    }  // namespace

    // define the scope patterns
    std::vector<ScopeFusionPatterns> ScopeRpnProposalsPass::DefinePatterns() {
        std::vector<ScopeFusionPatterns> patternsList;
        ScopeFusionPatterns patterns;
        GenScopePatterns(patterns);   // recognize rpn_proposals scope
        patternsList.push_back(patterns);
        return patternsList;
    }

    // return the name of the pass
    std::string ScopeRpnProposalsPass::PassName() {
        return std::string("ScopeRpnProposalsPass");
    }

    // distinguish the scpoe pattern
    void ScopeRpnProposalsPass::GenScopePatterns(ScopeFusionPatterns &patterns) {
        OP_LOGI(kOpType, "GenScopePatterns start.");
        std::vector<ScopePattern *> batch;
        // alloc an new object of ScopePattern
        ScopePattern* rpnProposalsPattern = new (std::nothrow) ScopePattern();
        if (rpnProposalsPattern == nullptr) {
            ScopeUtil::FreeScopePatterns(patterns);
            OP_LOGE(kOpType, "Alloc an object failed.");
            return;
        }
        rpnProposalsPattern->SetSubType(kRpnProposalsType);
        rpnProposalsPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("NonMaxSuppressionV2", 1, 0)); // The number of NonMaxSuppressionV2 is 1.
        rpnProposalsPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("TopKV2", 1, 0)); // The number of TopKV2 is 1.
        rpnProposalsPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Where", 0, 4)); // The number of Where is a multiple of 2.  actual 4
        rpnProposalsPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Gather", 0, 6)); // The number of Gather is a multiple of 2. actual  6

        // rpnProposalsPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Size", -1, 0)); // The Size op is not included
        rpnProposalsPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("ExpandDims", -1, 0)); // The ExpandDims op is not included
        rpnProposalsPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Switch", -1, 0)); // The Switch op is not included
        rpnProposalsPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("transpose", -1, 0)); // The transpose op is not included

        batch.push_back(rpnProposalsPattern);
        patterns.push_back(batch);
    }

    // validation the scope pattern
    Status ScopeRpnProposalsPass::LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph> &scopeGraph,
    std::vector<ScopesResult>   &results) {
        OP_LOGI(kOpType, "LastMatchScopesAndOPs start.");
        if (scopeGraph == nullptr) {
            OP_LOGE(kOpType, "Input params is nullptr.");
            return domi::PARAM_INVALID;
        }
        const ScopeTree *scopeTree = scopeGraph->GetScopeTree();
        if (scopeTree == nullptr) {
            OP_LOGE(kOpType, "Scope tree is nullptr.");
            return domi::PARAM_INVALID;
        }
        std::vector<Scope *> scopes = scopeTree->GetAllScopes();
        for (auto &scope : scopes) {
            if (scope->SubType() == kRpnProposalsType && scope->Name().find("generate_rpn_proposals")
                != std::string::npos) {
                OP_LOGI(kOpType, "RpnProposals LastMatchScopesAndOPs match SubType.");
                ScopesResult result;
                std::vector<Scope *> result_scopes;
                result_scopes.push_back(scope);
                result.SetScopes(result_scopes);
                results.push_back(result);
            }
        }
        return (!(results.empty())) ? SUCCESS : FAILED;
    }

    // define the input and output of this scope
    void ScopeRpnProposalsPass::GenerateFusionResult(const std::vector<Scope *> &scopes, FusionScopesResult *fusionRlt)
    {
        if (fusionRlt == nullptr) {
            OP_LOGE(kOpType, "Input fusionRlt is nullptr.");
            return;
        }

        for (auto &scope : scopes) {
            // The upper call guarantees that the scope is not empty.
            if (scope->SubType() != kRpnProposalsType) {
                continue;
            }

            fusionRlt->InsertInputs("filtered_boxes", {1, kFusionDisableIndex});	// Input index 1 of rpn_propsoals
            fusionRlt->InsertInputs("filtered_scores", {1, kFusionDisableIndex});	  // Input index 1 of rpn_propsoals
            fusionRlt->InsertInputs("Gather_1", {1, kFusionDisableIndex});	  // Input index 1 of rpn_propsoals
            fusionRlt->InsertInputs("transpose", {0, kFusionDisableIndex});		// Input index 0 of rpn_propsoals
            fusionRlt->InsertInputs("clip_boxes/ReverseV2", {2, kFusionDisableIndex});	  // Input index 2 of rpn_propsoals

            fusionRlt->InsertOutputs("boxes", {0});  // Output index 0 of rpn_propsoals

            fusionRlt->SetType(kScopeType);
            fusionRlt->SetDescription("");
            std::string scopeName = scope->Name();
            OP_LOGI(kOpType, "RpnProposals Scope name in TF is: %s", scopeName.c_str());
            fusionRlt->SetName(scopeName.substr(0, scopeName.length() - 1));
        }
        OP_LOGI(kOpType, "RpnProposals Scope fusion success.");
        return;
    }

}  // namespace ge
