/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file scanpqcodes_fusion_pass.h
 * \brief ScanPQCodes fusion pass(ScanPQCodes --> ScanPQCodes(Aicore)+ScanPQCodes(VectorCore))
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_A_SCANPQCODES_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_A_SCANPQCODES_FUSION_PASS_H_

#include <string>
#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ScanPQCodesFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& new_nodes) override;

 private:
  const std::string FUSED_OP_TYPE = "ScanPQCodes";
  Status SplitScanPQCodesNode(ge::ComputeGraph& graph, ge::NodePtr& scanPQCodesNode, ge::NodePtr& scanPQCodesNodeAicore,
                              ge::NodePtr& scanPQCodesNodeVectorCore);
  Status CreateNewTopKPQDistanceNode(ge::ComputeGraph& graph, ge::NodePtr& scanPQCodesNode,
                                     vector<ge::NodePtr>& newScanPQCodesNodes, ge::NodePtr& topKPQDistanceNode,
                                     ge::NodePtr& fusedTopKPQDistanceNode);
  Status ClearFusedNode(ge::ComputeGraph& graph, ge::NodePtr& node);
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_A_SCANPQCODES_FUSION_PASS_H_
