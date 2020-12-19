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

#ifndef OPS_BUILT_IN_FRAMEWORK_TF_SCOPE_FUSION_PASSES_SCOPE_DYNAMIC_RNN_PASS_H_
#define OPS_BUILT_IN_FRAMEWORK_TF_SCOPE_FUSION_PASSES_SCOPE_DYNAMIC_RNN_PASS_H_

#include <string>
#include <vector>

#include "register/scope/scope_fusion_pass_register.h"

namespace ge {
class ScopeDynamicRNNPass : public ScopeBasePass {
 protected:
  std::vector<ScopeFusionPatterns> DefinePatterns() override;
  std::string PassName() override;

  /**
   * @brief ScopeDynamicRNNPass multi scope
   */
  Status LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph>& scope_graph, std::vector<ScopesResult>& results) override;
  void GenerateFusionResult(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt) override;
  void GenScopePatterns(ScopeFusionPatterns& patterns);
  void GenTacotronScopePatterns(ScopeFusionPatterns& patterns);
};
}  // namespace ge

#endif  // OPS_BUILT_IN_FRAMEWORK_TF_SCOPE_FUSION_PASSES_SCOPE_DYNAMIC_RNN_PASS_H_
