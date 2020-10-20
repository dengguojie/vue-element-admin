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
 * \file unpack_fusion_pass.h
 * \brief Unpack fusion pass(Unpack --> Split + Unpack)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_UNPACK_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_UNPACK_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class UnpackFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  Status AddUnpackOps(ge::OpDescPtr fused_desc, ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                      std::vector<ge::GeTensorDesc> output_desc, ge::NodePtr fused_node, ge::NodePtr splitvd_base_node,
                      int64_t num, int64_t axis, int64_t i, int64_t j, int64_t mini_out);
  const string FUSED_OP_TYPE = "SplitVD_Unpack";
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_UNPACK_FUSION_PASS_H_