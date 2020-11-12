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
 * \file dynamic_gru_v2_fusion_pass.h
 * \brief DynamicGRUV2 fusion pass(DynamicGRUV2 --> BatchMatmulV2 & DynamicGRUV2Hidden)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_GRU_V2_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_GRU_V2_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class DynamicGRUV2FusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  void SetAttr(ge::OpDescPtr gru_desc, ge::OpDescPtr gru_split_desc);
  ge::NodePtr AddSplitNode(ge::NodePtr gru_node, ge::NodePtr matmul_node,
                           ge::ComputeGraph& graph, vector<ge::NodePtr>& new_nodes);
  ge::NodePtr AddMatmulNode(ge::NodePtr gru_node, ge::ComputeGraph& graph,
                            vector<ge::NodePtr>& new_nodes);
  bool JudgeSplit(ge::NodePtr gru_node, bool& result);
  const string FUSED_OP_TYPE = "DynamicGRUV2FusionPass";
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DYNAMIC_GRU_V2_FUSION_PASS_H_
