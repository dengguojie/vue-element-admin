/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
 * \file group_norm_squeeze_fusion_pass.h
 * \brief squeeze + instance_norm & const + mul + add fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_GROUP_NORM_SQUEEZE_FUSION_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_GROUP_NORM_SQUEEZE_FUSION_H_

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class GroupNormSqueezeFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
 private:
  const string FUSED_OP_TYPE = "GroupNorm";
  Status CheckNode(const ge::NodePtr &squeeze_node, const ge::NodePtr &instance_norm_node,
                   const ge::NodePtr &mul_node, const ge::NodePtr &add_node);

  Status GenGroupNorm(const ge::NodePtr &squeeze_node, const ge::NodePtr &instance_norm_node,
                      const ge::NodePtr &mul_node, const ge::NodePtr &add_node,
                      ge::OpDescPtr &group_norm_desc);

  Status AddNodeEdge(ge::NodePtr &squeeze_node, ge::NodePtr &instance_norm_node,
                     ge::NodePtr &mul_node, ge::NodePtr &add_node, ge::NodePtr &group_norm_node);
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_GROUP_NORM_SQUEEZE_FUSION_H_
