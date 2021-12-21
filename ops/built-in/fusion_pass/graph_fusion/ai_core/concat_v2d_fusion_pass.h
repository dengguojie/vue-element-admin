/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file concat_v2d_fusion_pass.h
 * \brief Concatv2d fusion pass(multi Concatv2d --> single Concatv2d)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONCAT_V2D_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONCAT_V2D_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class Concatv2dFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) override;

 private:
  bool CheckConcatValid(ge::NodePtr node, ge::Format format, ge::GeShape shape, int32_t dimNum);
  static bool HasUnKnowInputShape(const std::vector<ge::NodePtr>& input_nodes);
  Status PatternParse(ge::NodePtr concatv2dNode, vector<ge::NodePtr>& fusedInputNodes,
                      vector<ge::NodePtr>& concatNodes);
  const string FUSED_OP_TYPE = "ConcatV2D";
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONCAT_V2D_FUSION_PASS_H_
