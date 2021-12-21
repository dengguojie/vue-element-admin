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
 * \file non_max_suppression_fusion_pass.h
 * \brief non_max_suppression fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_IMGWARP_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_IMGWARP_FUSION_PASS_H_

#include <vector>
#include <string>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {

class IMGWarpFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;
  int64_t GetDimNum(const vector<int64_t>& shapes) const;
  void SetAssitValue(float* data, const std::vector<int64_t>& shape) const;
  Status CreateConstNode(vector<int64_t>& assit_tensor_shape, ge::NodePtr& fuse_node, ge::ComputeGraph& graph,
                         ge::NodePtr& const_node) const;
  Status CreateAddNode(ge::NodePtr& add_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                       vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& add_input0_desc) const;
  Status CreateIMGWarpOffsetsNode(ge::NodePtr& offsets_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                  vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& input0_desc,
                                  ge::GeTensorDesc& input1_desc) const;
  Status CreateIMGWarpResizeNode(ge::NodePtr& resize_node, ge::NodePtr& fused_node, ge::ComputeGraph& graph,
                                 vector<ge::NodePtr>& new_nodes, ge::GeTensorDesc& input0_desc,
                                 ge::GeTensorDesc& input1_desc, ge::GeTensorDesc& output_desc) const;

 private:
  const string FUSED_OP_TYPE = "IMGWarp";
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_IMGWARP_FUSION_PASS_H_
