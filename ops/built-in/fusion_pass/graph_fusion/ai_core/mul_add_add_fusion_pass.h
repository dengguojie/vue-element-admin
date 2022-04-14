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
 * \file mul_add_add_fusion_pass.h
 * \brief mul+add+transData+add fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_MUL_ADD_ADD_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_MUL_ADD_ADD_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "graph/tensor.h"

namespace fe {
class MulAddAddFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  Status RemoveFusedNode(ge::ComputeGraph& graph, ge::NodePtr& fusedNode);
  Status CheckFusedNode(vector<ge::NodePtr>& fusedNodes, ge::NodePtr& transdataDstNode);
  Status CheckFusedControlAnchor(ge::NodePtr& fusedNode, ge::NodePtr& mulAddAddNode);
  Status GetTransdataNode(ge::NodePtr& srcNode, ge::NodePtr& mulDstNode);
  Status AddMulPadDNode(ge::ComputeGraph& graph, ge::NodePtr& mulNode, ge::NodePtr& mulPadDNode);
  Status AddPadDNode(ge::ComputeGraph& graph, ge::NodePtr& addNode, ge::NodePtr& padDNode);
  Status AddAndDeleteEdge(vector<ge::NodePtr>& fusedNodes, ge::NodePtr& fusedMulAddAddNode, ge::NodePtr& mulPadDNode,
                          ge::NodePtr& padDNode, ge::NodePtr& transdataDstNode);
  Status FusionUnalignedScense(ge::ComputeGraph& graph, vector<ge::NodePtr>& fusedNodes, vector<ge::NodePtr>& newNodes,
                               ge::NodePtr& transdataDstNode);
  const string FUSED_OP_TYPE = "FusedMulAddAdd";
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_MUL_ADD_ADD_FUSION_PASS_H_
