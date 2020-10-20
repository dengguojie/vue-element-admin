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
 * \file conv2d_group_fusion_pass.h
 * \brief split fusion pass(conv2d --> conv2d/splited conv2d/depthwise conv2d)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV2D_GROUP_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV2D_GROUP_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class Conv2DGroupFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  Status SwapNumChn(ge::OpDescPtr opDesc, bool bInput, uint32_t index);
  Status ProcessDepthwiseConv(ge::NodePtr convNode);
  const string FUSED_OP_TYPE = "Conv2D";
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV2D_GROUP_FUSION_PASS_H_
