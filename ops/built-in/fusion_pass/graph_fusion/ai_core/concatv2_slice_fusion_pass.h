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
 * \file concatv2_slice_fusion_pass.h
 * \brief concatv2 slice fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONCATV2_SLICE_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONCATV2_SLICE_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "graph/tensor.h"

namespace fe {
class Concatv2SliceFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

private:
  bool IsVectorZeros(const vector<int64_t>& vec) const;
  Status GetConstValue(const ge::NodePtr &node_name, const string &attr_name, vector<int64_t> &const_data);
  Status GetSortedFusedNodes(const ge::ComputeGraph& graph, const Mapping& mapping, vector<ge::NodePtr>& fusedNodes);
  Status CheckSliceInfo(const ge::NodePtr& concatV2Node, const ge::NodePtr& sliceNode0, const ge::NodePtr& sliceNode1);
  const string FUSED_OP_TYPE = "ConcatV2Slice";
  const int64_t OUTPUTSIZE_LIMIT_TWO = 2;
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONCATV2_SLICE_FUSION_PASS_H_

