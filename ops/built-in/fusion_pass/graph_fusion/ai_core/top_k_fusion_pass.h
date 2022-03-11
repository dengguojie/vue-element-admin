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
 * \file top_k_fusion_pass.h
 * \brief TopK fusion pass(TopKV2 --> TopK)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_TOP_K_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_TOP_K_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
struct SegmentCalcParams {
  int64_t merge_channel = 4; // 4: max number of proposal arrays
  int64_t core_align_num = 1984; // 1984: max proposal num for one array
  int64_t core_min_num = 7936; // 7936: max proposal num limited by ub size
  int64_t pro_repeat_num = 16; // 16: proposal num processed for each repeat
  int64_t pro_data_num = 8; // 8: data num for one proposal
  int64_t k_num = 0;
  int64_t data_size = 0;
  int64_t ai_core_num = 0;
  string soc_version;
};

class TopKFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusion_nodes) override;

 private:
  const string kFusedOpType = "TopKD";
  bool CheckMultiCoreSegment(ge::NodePtr& topk_node, SegmentCalcParams& calcParams);
  Status AddMultiMergeNode(ge::ComputeGraph& graph, ge::NodePtr& topk_node, ge::NodePtr& segmentsort_node,
                           int64_t segment_num, SegmentCalcParams& calcParams, vector<NodePtr>& fusion_nodes);
  Status AddSegmentSortAndMergeNode(ge::ComputeGraph& graph, ge::NodePtr& topk_node, SegmentCalcParams& calcParams,
                                    vector<ge::NodePtr>& fusion_nodes);
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_TOP_K_FUSION_PASS_H_
