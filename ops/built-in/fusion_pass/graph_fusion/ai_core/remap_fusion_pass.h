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
 * \file remap_fusion_pass.h
 * \brief remap fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_REMAP_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_REMAP_FUSION_PASS_H_

#include <vector>
#include <string>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {

class RemapFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;
  Status CreateSplitNode(ge::NodePtr& split_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                         vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& temp_desc) const;
  Status CreateFloorxNode(ge::NodePtr& floorx_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                          vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& temp_desc) const;
  Status CreateCeilxNode(ge::NodePtr& ceilx_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                         vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& temp_desc) const;
  Status CreateFlooryNode(ge::NodePtr& floory_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                          vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& temp_desc) const;
  Status CreateCeilyNode(ge::NodePtr& ceily_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                         vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& temp_desc) const;
  Status CreateMulsx1Node(ge::NodePtr& mulsx1_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                          vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& temp_desc, float& val) const;
  Status CreateMulsx2Node(ge::NodePtr& mulsx2_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                          vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& temp_desc, float& val) const;
  Status CreateMulsy1Node(ge::NodePtr& mulsy1_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                          vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& temp_desc, float& val) const;
  Status CreateMulsy2Node(ge::NodePtr& mulsy2_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                          vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& temp_desc, float& val) const;
  Status CreateAddNode(const std::string name, ge::NodePtr& add_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                       vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& temp_desc) const;
  Status CreateCastNode(const std::string name, ge::NodePtr& add_node, ge::NodePtr& fused_node,
                        ge::ComputeGraph& graph,  vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& temp_desc) const;
  Status CreateConcatNode(ge::NodePtr& concat_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                          vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& temp_desc) const;
  Status CreateRemapOffsetsNode(ge::NodePtr& offsets_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& input0_desc,
                                ge::GeTensorDesc& input1_desc) const;
  Status CreateRemapResizeNode(ge::NodePtr& resize_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                               vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& input0_desc,
                               ge::GeTensorDesc& input1_desc, ge::GeTensorDesc& output_desc) const;

 private:
  const string FUSED_OP_TYPE = "Remap";
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_REMAP_FUSION_PASS_H_
