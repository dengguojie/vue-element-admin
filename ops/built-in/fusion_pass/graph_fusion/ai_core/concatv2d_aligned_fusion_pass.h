/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file concatv2d_aligned_fusion_pass.h
 * \brief concatv2d inputs shape aligned fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONCATV2D_ALIGNED_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONCATV2D_ALIGNED_FUSION_PASS_H_

#include "graph/tensor.h"
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ConcatV2DAlignedFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  Status CheckFusedNodes(vector<ge::NodePtr>& fusedNodes) const;
  Status UpdateShapes(ge::NodePtr begin_node, const ge::NodePtr end_node) const;
  Status UpdateEdges(vector<ge::NodePtr>& fusedNodes, vector<ge::NodePtr>& newNodes) const;
  Status CreatePadDNode(ge::ComputeGraph& graph, const ge::OutDataAnchorPtr& preAnchor, size_t padDim,
                        ge::NodePtr& padDNode) const;
  Status CreateSplitVNode(ge::ComputeGraph& graph, const ge::OutDataAnchorPtr& preAnchor, size_t splitDim,
                          ge::NodePtr& splitVNode) const;
  Status CreateConcatV2Node(ge::ComputeGraph& graph, const vector<ge::OutDataAnchorPtr>& preAnchors, size_t concatDim,
                            ge::NodePtr& concatV2Node) const;
  const string FUSED_OP_TYPE = "FusedConcatV2DAligned";
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONCATV2D_ALIGNED_FUSION_PASS_H_
