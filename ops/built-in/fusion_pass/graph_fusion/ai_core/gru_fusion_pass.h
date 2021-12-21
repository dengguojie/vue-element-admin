/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
 * \file gru_fusion_pass.h
 * \brief GRU fusion pass
 *   (CommonGRU --> DynamicGRUV2)
 */

#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_GRU_FUSION_PASS_H
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_GRU_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class GRUFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  void ProcessNZFormat(std::vector<int64_t>& dims);
  void ProcessZFormat(std::vector<int64_t>& dims);
  std::vector<int64_t> RemoveNumDirectionsDim(const std::vector<int64_t>& dims, bool isReverse);
  std::vector<int64_t> ProcessOutputDim(const std::vector<int64_t>& dims);
  Status AddTransposNode(ge::NodePtr gruNode, int anchorIndex, ge::ComputeGraph& graph);
  Status CreateSliceNode(ge::ComputeGraph& graph, ge::NodePtr& gru_node, ge::NodePtr& new_node);
  Status AddBiasSplitNode(ge::ComputeGraph& graph, ge::NodePtr& fused_node, ge::NodePtr& splitNode);
  const string FUSED_OP_TYPE = "SplitD_DynamicGRUV2";
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_GRU_FUSION_PASS_H