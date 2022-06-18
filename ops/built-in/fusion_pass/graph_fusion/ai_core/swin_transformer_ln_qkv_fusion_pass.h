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
 * \file swin_transformer_ln_qkv_fusion_pass.h
 * \brief swin_transformer_ln_qkv_fusion_pass
 */

#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_SWIN_TRANSFORMER_LN_QKV_FUSION_PASS_H
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_SWIN_TRANSFORMER_LN_QKV_FUSION_PASS_H

#include <string>

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
namespace fe {
class SwinTransformerLnQKVFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusion_nodes) override;

 private:
  const string FUSED_OP_TYPE = "SwinTransformerLnQkv";
  vector<FusionPattern*> DefineFirstPatterns(vector<FusionPattern*>& patterns);
  vector<FusionPattern*> DefineSecondPatterns(vector<FusionPattern*>& patterns);
  vector<FusionPattern*> DefineThirdPatterns(vector<FusionPattern*>& patterns);
  vector<FusionPattern*> DefineFourthPatterns(vector<FusionPattern*>& patterns);
  bool CheckNodeShape(vector<int64_t>& label_shape, vector<int64_t>& check_shape);
  Status GetNodeFromPatten(Mapping& mapping, vector<ge::NodePtr>& node_ptr_all);
  Status NewNodeAddEdge(ge::NodePtr& ln_qkv_node, vector<ge::NodePtr>& node_ptr_all);
  Status CheckLayerNormNode(ge::NodePtr& ln_node);
  Status CheckBatchMatmulNode(ge::NodePtr& batchmatmul_node);
  Status CheckReshapeNode(vector<ge::NodePtr>& node_ptr_all);
  Status CheckConfusionTransposeNode(ge::NodePtr& confuse_node);
  Status CheckRollNode(vector<ge::NodePtr>& node_ptr_all);
  Status CheckConcatNode(vector<ge::NodePtr>& node_ptr_all);
  Status SetInputOutputDesc(vector<ge::NodePtr>& node_ptr_all, std::shared_ptr<ge::OpDesc>& ln_qkv_desc);
  Status CheckPattenNode(vector<ge::NodePtr>& node_ptr_all);
  Status SetAttrPattenNode(vector<ge::NodePtr>& node_ptr_all, ge::NodePtr& ln_qkv_node);
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_ATTENTION_LN_QKV_FUSION_PASS_H

