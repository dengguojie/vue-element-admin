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
  std::vector<int64_t> RemoveNumDirectionsDim(const std::vector<int64_t>& dims, bool isReverse) const;
  std::vector<int64_t> ProcessOutputDim(const std::vector<int64_t>& dims);
  Status AddTransposNode(ge::NodePtr gruNode, int anchorIndex, ge::ComputeGraph& graph);
  Status CreateSliceNode(ge::ComputeGraph& graph, ge::NodePtr& gru_node, ge::NodePtr& new_node);
  Status AddBiasSplitNode(ge::ComputeGraph& graph, const ge::NodePtr& fusedNode, ge::NodePtr& splitNode) const;

  Status CheckParams(const ge::OpDescPtr& fusedDesc);
  void SetTensorDescription(ge::GeTensorDesc &tensorDesc, vector<int64_t> &dims, const ge::Format &format,
                            const ge::DataType &dtype);
  Status ProcessBidiFusion(ge::ComputeGraph& graph, ge::NodePtr& fusedNode, ge::OpDescPtr& fusedDesc,
                           vector<ge::NodePtr>& newNodes);
  Status SetSplitVNodeInfo(ge::GeTensorDesc& tensorDesc, ge::OpDescPtr& outOpDesc, vector<int64_t>& dimIn,
                           vector<int32_t>& inputDims);
  Status AddSplitVNode(const std::string& nodeName, ge::GeTensorDesc& inputDesc, ge::GeTensorDesc& outputDesc,
                       ge::NodePtr& splitNode, ge::NodePtr& peerOutNode, vector<int32_t>& splitDimAxis,
                       vector<int32_t>& sizeSplitAxis, int splitIndex, vector<ge::NodePtr>& newNodes,
                       ge::ComputeGraph& graph);
  Status AddBidiWeightSplitNode(ge::ComputeGraph& graph, const ge::NodePtr& fusedNode,
                                int weightIndex, ge::NodePtr& splitNode, vector<ge::NodePtr>& newNodes);
  Status AddBidiBiasSplitNode(ge::ComputeGraph& graph, const ge::NodePtr& fusedNode, int biasIndex,
                              ge::NodePtr& forwardSplitNode, ge::NodePtr& reverseSplitNode,
                              vector<ge::NodePtr>& newNodes);
  Status AddBidiInitHSplitNode(ge::ComputeGraph& graph, const ge::NodePtr& fusedNode,
                               int initHIndex, ge::NodePtr& splitNode, vector<ge::NodePtr>& newNodes);
  Status AddExpandDimsAndConcatNode(ge::ComputeGraph& graph, ge::NodePtr& fusedNode, ge::NodePtr& forwardNode,
                                    ge::NodePtr& reverseNode, ge::GeTensorDesc& outputDesc,
                                    vector<ge::NodePtr>& newNodes);
  Status AddSliceAndConcatNode(ge::ComputeGraph& graph, ge::NodePtr& fusedNode, ge::NodePtr& forwardNode,
                               ge::NodePtr& reverseNode, ge::GeTensorDesc& outputDesc, vector<ge::NodePtr>& newNodes,
                               int nodeIndex);

  const string FUSED_OP_TYPE = "SplitD_DynamicGRUV2";
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_GRU_FUSION_PASS_H