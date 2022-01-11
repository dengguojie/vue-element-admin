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
 * \file rnn_fusion_pass.h
 * \brief LayerNormGrad fusion pass
 *   (LayerNormGrad --> LayerNormXBackprop & LayerNormBetaGammaBackprop)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_RNN_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_RNN_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class RNNFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  ge::GeTensorDesc ProcessStatic(const ge::NodePtr& fusedNode, const int32_t num_output, ge::NodePtr& innerproductNode,
                                 ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes, bool& failStatus);
  vector<ge::NodePtr> ProcessRnnCell(const ge::NodePtr& fusedNode, ge::ComputeGraph& graph,
                                     const ge::GeTensorDesc& outInnerProductTensorDesc, vector<ge::NodePtr>& newNodes,
                                     bool& failStatus, const bool has_static);
  const string FUSED_OP_TYPE = "SplitVD_BasicRNNCell_ConcatD";
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_RNN_FUSION_PASS_H_
