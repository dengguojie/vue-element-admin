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
 * \file pad_v3_fusion_pass.h
 * \brief split fusion pass(padv3 --> padv3 + strideslice)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_PADV3_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_PADV3_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "graph/tensor.h"

namespace fe {
class PadV3FusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) override;

 private:
  bool GetConstValue(const ge::Tensor& constTensor, const ge::DataType& dtype,
                     std::vector<int64_t>& constData);
  bool SplitPadding(std::vector<int64_t>& padValue, std::vector<int64_t>& leftPad, std::vector<int64_t>& rightPad);
  Status CreateStrideSliceDNode(ge::ComputeGraph& graph, ge::NodePtr& padNode, ge::NodePtr& newNode,
                               std::vector<int64_t>& outputDims, std::vector<int64_t>& begin,
                               std::vector<int64_t>& end, std::vector<int64_t>& strides);
  Status UpdatePadding(ge::NodePtr& padNode, std::vector<int64_t>& pads, ge::DataType dtype);
  Status AddEdge(ge::NodePtr& padNode, ge::NodePtr& sliceNode) const;
  Status Infer(const ge::Operator& op, std::vector<int64_t>& paddings, bool paddingsContiguous);
  bool CheckDynamic(std::vector<int64_t>& dims);
  void CacuBegin(std::vector<int64_t>& pads, bool paddingsContiguous, std::vector<int64_t>& begin);
  void CacuEnd(std::vector<int64_t>& pads, std::vector<int64_t>& end, std::vector<int64_t>& outputDims, bool paddingsContiguous);
  const string FUSED_OP_TYPE = "PadV3";
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_PADV3_FUSION_PASS_H_
