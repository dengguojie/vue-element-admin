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
 * \file scope_dynamic_lstm_pass.h
 * \brief
 */
#ifndef OPS_BUILT_IN_FRAMEWORK_TF_SCOPE_FUSION_PASSES_SCOPE_DYNAMIC_LSTM_PASS_H_
#define OPS_BUILT_IN_FRAMEWORK_TF_SCOPE_FUSION_PASSES_SCOPE_DYNAMIC_LSTM_PASS_H_

#include <string>
#include <vector>

#include "register/scope/scope_fusion_pass_register.h"

namespace ge {
class ScopeDynamicLSTMPass : public ScopeBasePass {
 protected:
  std::vector<ScopeFusionPatterns> DefinePatterns() override;
  std::string PassName() override;

  /**
   * @brief ast match, scope multi scope
   */
  Status LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph>& scope_graph, std::vector<ScopesResult>& results) override;
  void GenerateFusionResult(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt) override;
  void GenScopePatterns(ScopeFusionPatterns& patterns);
};
}  // namespace ge

#endif  // OPS_BUILT_IN_FRAMEWORK_TF_SCOPE_FUSION_PASSES_SCOPE_DYNAMIC_LSTM_PASS_H_
