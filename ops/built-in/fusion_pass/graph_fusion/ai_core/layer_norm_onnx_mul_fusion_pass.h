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
 * \file layer_norm_fusion_pass.h
 * \brief layer norm fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LAYER_NORM_ONNX_MUL_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LAYER_NORM_ONNX_MUL_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class LayerNormONNXMULFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) override;

 private:
  bool with_affine = true;
  bool with_pow = false;
  const string FUSED_OP_TYPE = "LayerNorm";
  std::vector<int64_t> axes;

  template <class T>
  Status CreatNode(ge::ComputeGraph& graph, const ge::NodePtr& previous_node, ge::NodePtr& cur_node, std::string opname,
                   std::string optype, T value, vector<ge::NodePtr>& fusionNodes);
  Status CreateMulAndAddNode(ge::ComputeGraph& graph, const ge::NodePtr div0_node, ge::NodePtr& mul0_node,
                             ge::NodePtr& add1_node, vector<ge::NodePtr>& fusionNodes);
  Status AddEdge(const ge::NodePtr& pre_node, int pre_idx, const ge::NodePtr& cur_node, int cur_idx);
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LAYER_NORM_FUSION_PASS_H_
