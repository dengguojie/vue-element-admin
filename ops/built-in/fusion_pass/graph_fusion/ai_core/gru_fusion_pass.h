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
  struct SplitInfo {
    const std::string& nodeName;
    ge::GeTensorDesc& inputDesc;
    ge::GeTensorDesc& outputDesc;
    vector<int32_t>& splitDimAxis;
    vector<int32_t>& sizeSplitAxis;
    int splitIndex;
  };
  struct InputNodes {
    ge::NodePtr dynamicGruV2ForwardNode;
    ge::NodePtr dynamicGRUV2ReverseNode;
    ge::NodePtr weightInputSplitNode;
    ge::NodePtr weightHiddenSplitNode;
    ge::NodePtr inputBiasSplitNode;
    ge::NodePtr hiddenBiasSplitNode;
    ge::NodePtr initHSplitNode;
  };

  void ProcessNZFormat(std::vector<int64_t>& dims);
  void ProcessZFormat(std::vector<int64_t>& dims);
  std::vector<int64_t> RemoveNumDirectionsDim(const std::vector<int64_t>& dims, bool isReverse) const;
  std::vector<int64_t> ProcessOutputDim(const std::vector<int64_t>& dims);
  Status AddTransposNode(int anchorIndex, int nodeNum, ge::NodePtr& transposeNode);
  Status CreateSliceNode(ge::NodePtr gru_node, ge::NodePtr& new_node);
  Status AddBiasSplitNode(ge::NodePtr& splitNode);
  Status CheckParams();
  void SetTensorDescription(ge::GeTensorDesc& tensorDesc, vector<int64_t>& dims, const ge::Format& format,
                            const ge::DataType& dtype);
  Status ProcessBidiFusion();
  Status SetSplitVNodeInfo(ge::GeTensorDesc& tensorDesc, ge::OpDescPtr& outOpDesc, vector<int64_t>& dimIn,
                           vector<int32_t>& inputDims);
  Status AddSplitVNode(SplitInfo splitInfo, ge::NodePtr& splitNode, ge::NodePtr peerOutNode);
  Status AddBidiWeightSplitNode(int weightIndex, ge::NodePtr& splitNode);
  Status AddBidiBiasSplitNode(int biasIndex, ge::NodePtr& inputSplitNode, ge::NodePtr& hiddenSplitNode);
  Status AddBidiInitHSplitNode(int initHIndex, ge::NodePtr& splitNode);
  Status AddExpandDimsAndConcatNode(ge::NodePtr forwardNode, ge::NodePtr reverseNode);
  Status AddSliceAndConcatNode(ge::NodePtr forwardNode, ge::NodePtr reverseNode, int nodeIndex);
  template<typename T>
  ge::NodePtr AttrToConstNode(vector<ge::OpDescPtr>& tensorDesc, T& inputDims, const std::string& nodeName);
  Status AddConcatNode(vector<ge::NodePtr>& fusedNodes, const std::string& nodeName, vector<int64_t>& nodeDims);
  Status AddBidiInputNodes(InputNodes nodes);
  Status AddInputNodes(ge::NodePtr splitNode, ge::NodePtr gruNode);
  void UpdateOutputDesc(ge::OpDescPtr gruOpDesc, ge::GeTensorDesc& tensorDesc);
  void UnlinkAllAnchors();
  Status InitParams(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes);

  const string FUSED_OP_TYPE = "SplitD_DynamicGRUV2";
  ge::ComputeGraph* graph_;
  vector<ge::NodePtr>* newNodes_;
  ge::NodePtr fusedNode_;
  ge::OpDescPtr fusedDesc_;
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_GRU_FUSION_PASS_H