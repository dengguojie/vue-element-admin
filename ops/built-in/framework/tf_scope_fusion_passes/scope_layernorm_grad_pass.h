/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file scope_layernorm_grad_pass.h
 * \brief
 */
#ifndef OPS_BUILT_IN_FRAMEWORK_TF_SCOPE_FUSION_PASSES_SCOPE_LAYERNORM_GRAD_PASS_H_
#define OPS_BUILT_IN_FRAMEWORK_TF_SCOPE_FUSION_PASSES_SCOPE_LAYERNORM_GRAD_PASS_H_

#include <string>
#include <vector>
#include "register/scope/scope_fusion_pass_register.h"

namespace ge {
/**
 * @ingroup domi_omg
 * @brief ScopeLayerNormGradPass
 */
class ScopeLayerNormGradPass : public ScopeBasePass {
 protected:
  std::vector<ScopeFusionPatterns> DefinePatterns() override;

  std::string PassName() override;

  Status LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph>& scope_graph, std::vector<ScopesResult>& results) override;

  void GenerateFusionResult(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt) override;

  void GenScopePatterns(ScopeFusionPatterns& patterns) const;

 private:
  std::vector<ge::OperatorPtr> FindOutNodesShouldInScope(const Scope* scope, std::shared_ptr<ScopeGraph>& scope_graph,
                                                         const std::string& mask = "");

  void FindInputXIndex(const Scope* scope, int& index) const;

  void FindInputIndex(const Scope* scope, int& index, const std::string& name, const std::string& base_name) const;
  void IsConnectReshape(const Scope* scope, const std::string& name, bool& is_connected_reshape) const;
  void FindOutputdXNode(const std::vector<ge::OperatorPtr>& nodes, std::string& node_def_name, int &index);
  void ProcessInputDy(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt);
  void ProcessInputX(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt);
  void ProcessInputGamma(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt);
  void OutputGammaBetaProcess(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt);
};
}  // namespace ge

#endif  // OPS_BUILT_IN_FRAMEWORK_TF_SCOPE_FUSION_PASSES_SCOPE_LAYERNORM_GRAD_PASS_H_
