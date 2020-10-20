/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file deconv_weight_trans_fusion_pass.h
 * \brief deconv weight trans fusion pass(weight -> deconv ===> weight ->
 *   reshape -> transpose -> reshape -> reverse -> reshape -> deconv)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DECONV_WEIGHT_TRANS_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DECONV_WEIGHT_TRANS_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class DeconvWeightTransFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) override;

 private:
  void GetShapeUsedByIntermediateProcessInDeconvWeightTrans(const ge::Format& filterFormat,
                                                            const vector<int64_t>& shapeNCHW, vector<int64_t>& dimComp,
                                                            vector<int64_t>& reshapeIn, vector<int64_t>& transPerm,
                                                            vector<int64_t>& reverseAxis, vector<int64_t>& reshapeOut);
  Status Relink(ge::NodePtr filterNode, ge::NodePtr dimCompNode, ge::NodePtr transposeNode, ge::NodePtr reformatNode,
                ge::NodePtr reshapeInNode, ge::NodePtr reverseNode, ge::NodePtr reshapeOutNode, ge::NodePtr deconvNode);
  const string FUSED_OP_TYPE = "Deconvolution";
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DECONV_WEIGHT_TRANS_FUSION_PASS_H_
