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
 * \file swin_attention_ffn_fusion_pass.h
 * \brief batchmatmulv2 + add + reshape + reshape + transpose + reshape + reshape fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_SWIN_ATTENTION_FFN_FUSION_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_SWIN_ATTENTION_FFN_FUSION_H_

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class SwinAttentionFFNFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph,
                Mapping &mapping,
                vector<ge::NodePtr> &fusionNodes) override;
 private:
  const string FUSED_OP_TYPE = "SwinAttentionFFN";
  Status GetNodeFromPattern(Mapping& mapping, vector<ge::NodePtr>& node_ptr_all);
  bool CheckNodeShape(vector<int64_t>& label_shape, vector<int64_t>& check_shape);
  Status CheckBatchMatmulNode(ge::NodePtr& batchmatmul_node);
  Status CheckReshapeNode(vector<ge::NodePtr>& node_ptr_all);
  Status CheckRollNode(vector<ge::NodePtr>& node_ptr_all);
  Status CheckConcatNode(vector<ge::NodePtr>& node_ptr_all);
  Status CheckPatternNode(vector<ge::NodePtr>& node_ptr_all);
  Status NewNodeAddEdge(ge::NodePtr& attention_matmul_node, vector<ge::NodePtr>& node_ptr_all);
  Status SetInputOutputDesc(vector<ge::NodePtr>& node_ptr_all, std::shared_ptr<ge::OpDesc>& attention_matmul_desc);
  Status SetAttrPatternNode(vector<ge::NodePtr>& node_ptr_all, ge::NodePtr& attention_matmul_node);
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_SWIN_ATTENTION_FFN_FUSION_H_
