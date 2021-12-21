/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
 * \file dynamic_rnn_fusion_pass.h
 * \brief DynamicRNN fusion pass
 * (DynamicRNN(BIDIRECTIONAL) --> split + [DynamicRNN + DynamicRNN(REDIRECTIONAL)] + concat)
 */

/*
 * x  w   b seq_len init_h init_c      x  seq_len  w  b init_h init_c
 * \  \   |   /    /      /             \    \      \  \  |    /
 *                                                       split
 *  \  \  |  /   /      /                \     \    /  /  |    \
 * DynamicRNN(BIDIRECTIONAL)  --> [DynamicRNN  +   DynamicRNN(REDIRECTIONAL)]
 *   / / / | \ \ \ \                        \ \ \  |  / / /  /
 *                                              concat
 *  / / /  |  \ \ \  \                       / / /  |  \ \ \  \
 * y  h c  i  j  f o tanhc                  y  h c  i  j f o tanhc
 */

#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_RNN_FUSION_PASS_H
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_RNN_FUSION_PASS_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {

// input index map
struct InputIndexInfo {
  int32_t xIndex = 0;
  int32_t wIndex = 1;
  int32_t bIndex = 2;
  int32_t sIndex = 3;
  int32_t hIndex = 4;
  int32_t cIndex = 5;
};

class DynamicRNNFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  ge::OpDescPtr CreateRNNDesc(ge::OpDescPtr RNNDesc, ge::OpDescPtr dynamicRNNDesc, string direction, bool has_seq,
                              bool has_h0, bool has_c0);
  ge::OpDescPtr CreateSplitDesc(ge::OpDescPtr splitDesc, ge::OpDescPtr dynamicRNNDesc, string tensorName,
                                int64_t splitDim);
  ge::OpDescPtr CreateConcatDesc(ge::OpDescPtr concatDesc, ge::OpDescPtr dynamicRNNDesc, string tensorName,
                                 int64_t concatDim);

  Status AddInputEdge(ge::NodePtr dynamicRNNNode, ge::NodePtr forwardRNNNode, ge::NodePtr backwardRNNNode,
                      ge::NodePtr splitWNode, ge::NodePtr splitBNode, ge::NodePtr splitHNode, ge::NodePtr splitCNode,
                      InputIndexInfo inputIndexInfo, bool has_seq, bool has_h0, bool has_c0);
  Status AddSplitEdge(ge::NodePtr splitNode, ge::NodePtr forwardRNNNode, ge::NodePtr backwardRNNNode, string nodeName,
                      int64_t index);
  Status AddConcatEdge(ge::NodePtr concatNode, ge::NodePtr dynamicRNNNode, ge::NodePtr forwardRNNNode,
                       ge::NodePtr backwardRNNNode, string nodeName, int64_t index);

  const string FUSED_OP_TYPE = "DynamicRNN";
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_RNN_FUSION_PASS_H
