/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file dynamic_rnn_grad_fusion_pass.h
 * \brief DynamicRNNGrad fusion pass(DynamicRNNGrad --> LSTMIInputGrad & LSTMWeightGrad(Split&Concat&Matmul&Reduce))
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_RNN_GRAD_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_RNN_GRAD_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class DynamicRNNGradFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  Status AddEdgeForNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr node, int unlinkIndex, int anchorIndex) const;

  Status AddEdgeForCell(ge::NodePtr dynamicRNNGradNode,
                        vector<ge::NodePtr>& newNodes, bool& failStatus,
                        vector<vector<ge::NodePtr>> resultNode, ge::NodePtr lstmSplitC,
                        ge::NodePtr lstmSplitDy, ge::NodePtr lstmSplitI,
                        ge::NodePtr lstmSplitJ, ge::NodePtr lstmSplitF,
                        ge::NodePtr lstmSplitO, ge::NodePtr lstmSplitTanh,
                        ge::NodePtr lstmXConcatD, ge::NodePtr& lstmGageConcatD);

  vector<vector<ge::NodePtr>> AddTLoopNode(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph,
                                           vector<ge::NodePtr>& newNodes, bool& failStatus);

  ge::NodePtr AddLSTMInputGradNode(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph,
                                   vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddSplitNode(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                           bool& failStatus);
  ge::NodePtr AddHConcatNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr splitNode, ge::ComputeGraph& graph,
                             vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddConcatNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr hConcatNode, ge::ComputeGraph& graph,
                            vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddConcatNodeT_1(ge::NodePtr dynamicRNNGradNode, ge::ComputeGraph& graph,
                            vector<ge::NodePtr>& newNodes, bool& failStatus);
  ge::NodePtr AddMatmulNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr concatNode, ge::NodePtr lstmInputGradNode,
                            ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes, bool& failStatus);
  Status AddDwReduceSumNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr matmulNode, ge::ComputeGraph& graph,
                            vector<ge::NodePtr>& newNodes);
  Status AddDbReduceSumNode(ge::NodePtr dynamicRNNGradNode, ge::NodePtr lstmInputGradNode, ge::ComputeGraph& graph,
                            vector<ge::NodePtr>& newNodes, ge::NodePtr const_one_node);
  const string FUSED_OP_TYPE = "LSTMInputGrad_Split_Concat_Matmul_Reduce";
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_RNN_GRAD_FUSION_PASS_H_
