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
 * \file conv3d_bp_filter_group_fusion_pass.h
 * \brief fusion pass(conv3d_bp_filter_group -->conv3d_bp_filter + mul)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV3D_BP_FILTER_GROUP_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV3D_BP_FILTER_GROUP_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class Conv3DBpFilterGroupFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& new_nodes) override;

 private:
  bool Relink(ge::NodePtr& conv_node, ge::NodePtr& mul_node, ge::NodePtr& const_node);
  bool GenerateMulNode(ge::ComputeGraph& graph,
                       const ge::OpDescPtr& conv_desc,
                       ge::NodePtr& mul_node);
  bool GenMultiplier(ge::ComputeGraph& graph, const std::vector<int64_t>& dims,
                     const std::map<std::string, int64_t>& group_map,
                     const ge::OpDescPtr& conv_desc,
                     ge::NodePtr& const_node);
  void SetMultiplierValue(float* data, const std::vector<int64_t>& dims,
                          const std::map<std::string, int64_t>& group_map);
  Status TransOutDims2dhwcn(const ge::OpDescPtr& dw_desc, std::vector<int64_t>& dims);
  Status CalculateGroup(int64_t in_channel, int64_t out_channel, int64_t groups,
                        std::map<std::string, int64_t>& group_map);
  int64_t LCM(int64_t numL, int64_t numR);
  Status GetChannelValue(const ge::OpDescPtr& dw_desc, const std::string& name, int64_t& channel);

  const string FUSED_OP_TYPE = "Conv3d_bp_filter_group";
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV3D_BP_FILTER_GROUP_FUSION_PASS_H_