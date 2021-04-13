/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file roialign_fusion_pass.h
 * \brief  fusion pass
 *
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_ROIALIGN_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_ROIALIGN_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ROIAlignFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& new_nodes) override;

 private:
  Status MakeCastNode(ge::ComputeGraph& graph, const ge::NodePtr& fused_node, ge::NodePtr& new_node);

  Status MakeConcatNode(ge::ComputeGraph& graph, const ge::NodePtr& fused_node, const ge::NodePtr& broad_node,
                        ge::NodePtr& new_node);
  const string FUSED_OP_TYPE = "ROIAlign_Concat_Broadcast_Fusion";
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_ROIALIGN_FUSION_PASS_H_
