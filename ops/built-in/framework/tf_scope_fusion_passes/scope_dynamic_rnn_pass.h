/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

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
#include <string.h>
#include "register/scope/scope_fusion_pass_register.h"
#include "graph/operator.h"

namespace ge {

#define GET_DEQUANT_SCALE_DEQ(dequantScaleData)                                \
  ((dequantScaleData & 0x00000000ffffffff))

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
  void GenLTCRNNScopePatterns(ScopeFusionPatterns& patterns);
  void GenChinaMobileScopePatterns(ScopeFusionPatterns& patterns);
  void GenQuantScopePatterns(ScopeFusionPatterns& patterns);
  void DynamicRNNPassParserParams(const std::unordered_map<std::string, ge::OperatorPtr>& nodes_map, const std::string &origin_node_name, const std::string &op_type, ge::Operator* inner_node);
  void QuantWeightRollBack(const std::unordered_map<std::string, ge::OperatorPtr>& nodes_map,
                           const std::string &origin_node_name, const std::string &type, ge::Operator* inner_node,
                           const float scale_x, const float scale_b);
  void GetQuantScale(const std::unordered_map<std::string, ge::OperatorPtr>& nodes_map,
                     float &scale_x, float &scale_b);
  void GenerateFusionResultForQuant(const Scope* scope, FusionScopesResult* fusion_rlt, const std::string sub_type);
  void GenerateFusionResultForLTCRNN(const Scope* scope, FusionScopesResult* fusion_rlt);
  void GenerateFusionResultForMultiLSTM(const Scope* scope, FusionScopesResult* fusion_rlt);
  void GenerateFusionResultForMultiNetease(const Scope* scope, FusionScopesResult* fusion_rlt);
  void ConcatParserParams(const std::string &origin_node_name, const std::string &op_type, ge::Operator* inner_node, FusionScopesResult *fusion_rlt);
  std::string GetNodeNameFromScope(const std::unordered_map<std::string, ge::OperatorPtr>& nodes_map, const std::string &sub_name);
};
}  // namespace ge

#endif  // OPS_BUILT_IN_FRAMEWORK_TF_SCOPE_FUSION_PASSES_SCOPE_DYNAMIC_RNN_PASS_H_
