/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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
 * \file continuation_indicator_fusion_pass.h
 * \brief for caffe (ContinuationIndicator)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONTINUATION_INDICATOR_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONTINUATION_INDICATOR_FUSION_PASS_H_

#include "pattern_fusion_util.h"
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ContinuationIndicatorFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusion_nodes) override;
 private:
  Status CreatConstNode(ge::ComputeGraph& graph, ge::NodePtr &const_node, int64_t time_step, int64_t batch_size);
  const string FUSED_OP_TYPE = "ContinuationIndicator";
};

}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONTINUATION_INDICATOR_FUSION_PASS_H_
