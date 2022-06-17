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
 * \file layernorm_training_fusion_pass.h
 * \brief fuse matched scope to target LayerNorm and LayerNormGrad.
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LAYERNORM_TRAINING_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LAYERNORM_TRAINING_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "graph_optimizer/fusion_common/fusion_turbo.h"

namespace fe {
class LayerNormTrainingFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusion_nodes) override;

 private:
  Status CheckNullPtr(const std::vector<ge::NodePtr>& pattern_nodes);
  Status CheckLNGNodes(const ge::NodePtr& add_n_node, const ge::NodePtr& mul_node,
                       const ge::NodePtr& mul_node_1, const uint32_t input_size);
  Status GetReduceOpAttr(std::vector<int64_t>& axes, bool& keep_dims, const ge::NodePtr& node);
  Status CheckReduceOpAttr(const std::vector<int64_t>& axes_0, const bool keep_dims_1,
                           const ge::NodePtr& mean_node_1, const ge::NodePtr& mean_node_2);
  Status GetInput24PeerInNodes(const ge::NodePtr& grad_rsqrt_node, const ge::NodePtr& rsqrt_node,
                               ge::NodePtr& mul_node_5, ge::NodePtr& add_n_node_0, ge::NodePtr& mul_node_8);
  Status GetInput13PeerInNodes(const ge::NodePtr& add_n_node_0, const ge::NodePtr& grad_sub_node,
                               ge::NodePtr& mul_node_4, ge::NodePtr& mul_4_up_node, ge::NodePtr& mul_node_2);
  Status GetInput0PeerInNodes(const ge::NodePtr& grad_mul_node_2, const ge::NodePtr& grad_sub_node,
                              ge::NodePtr& sum_node_2, ge::NodePtr& mul_node_1);
  Status AddTargetNode(FusionTurbo& turbo_instance, const ge::NodePtr& mean_node_1, const ge::NodePtr& sum_node_2,
                       ge::NodePtr& layer_norm_node, ge::NodePtr& layer_norm_grad_node);
  Status SetLayerNormAttr(const ge::NodePtr& node, const std::vector<int64_t>& axes_vec, const ge::NodePtr& add_1);
  void GetInputRelations(Relations& input_relations,
                         const std::vector<std::pair<ge::NodePtr, ge::NodePtr>>& pairs);
  void GetOutputRelations(Relations& output_relations, const std::vector<ge::NodePtr>& nodes);
  Status UpdateAddN(FusionTurbo& turbo_instance, const ge::NodePtr& layer_norm_grad_node,
                    const ge::NodePtr& add_n_node_1, const uint32_t input_size);

  const std::string FUSED_OP_TYPE = "LayerNorm";
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LAYERNORM_TRAINING_FUSION_PASS_H_
