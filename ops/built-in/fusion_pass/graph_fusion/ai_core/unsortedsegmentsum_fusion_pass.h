/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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
 * \file unsortedsegmentsum_fusion_pass.h
 * \brief
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_UNSORTEDSEGMENTSUM_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_UNSORTEDSEGMENTSUM_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class UnsortedSegmentSumFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) override;

 private:
  const string FUSED_OP_TYPE = "Yolo";
  Status AddConcatConstNode(const int32_t pad_dim_size,
                            int32_t const_value,
                            ge::NodePtr& const_node,
                            std::shared_ptr<ge::OpDesc>& concat_desc);
  Status AddSliceConstNode(const std::vector<int64_t>& const_shape,
                           std::vector<int32_t>& const_value,
                           ge::NodePtr& const_node,
                           std::shared_ptr<ge::OpDesc>& slice_desc);
  Status CheckUnsortedSegmentSumNode(const ge::NodePtr& unsorted_segment_sum_node);
  Status AddEdges(const ge::NodePtr& unsorted_segment_sum_node,
                  const int32_t pad_dim_size,
                  ge::NodePtr& concat_node,
                  ge::NodePtr& unsorted_segment_sum_pad_node,
                  ge::NodePtr& slice_node);
  Status AddNodes(const ge::NodePtr& unsorted_segment_sum_node,
                  const ge::OpDescPtr& unsorted_segment_sum_desc,
                  const int32_t pad_dim_size,
                  ge::ComputeGraph& graph,
                  vector<ge::NodePtr>& fusion_nodes);
  Status CheckDims(const std::vector<int64_t>& x_dims);
  Status GetPadDimSize(const ge::OpDescPtr& unsorted_segment_sum_desc, int32_t& pad_dim_size);
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_UNSORTEDSEGMENTSUM_FUSION_PASS_H_
