/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
 * \file layer_norm_onnx_fusion_pass.h
 * \brief layer norm onnx fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LAYER_NORM_ONNX_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LAYER_NORM_ONNX_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "graph/tensor.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
namespace fe {

class LayerNormONNXFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  const std::string FUSED_OP_TYPE = "LayerNorm";
  bool with_affine = true;
  bool x_dynamic = false;
  bool gamma_dynamic = false;
  bool beta_dynamic = false;
  float epsilon = 0.0000001;
  int begin_norm_axis = 0;
  std::vector<int64_t> axes;

  Status CheckEdges(std::map<std::string, ge::NodePtr>& nodes_map);
  Status CheckValue(std::map<std::string, ge::NodePtr>& nodes_map);
  template <class T>
  Status CreatNode(ge::ComputeGraph& graph, const ge::NodePtr& previous_node, ge::NodePtr& cur_node, std::string opname,
                   std::string optype, T value, vector<ge::NodePtr>& fusionNodes);
  Status CreateMulAndAddNode(ge::ComputeGraph& graph, const ge::NodePtr div0_node, ge::NodePtr& mul0_node,
                             ge::NodePtr& add1_node, vector<ge::NodePtr>& fusionNodes);
  Status AddEdge(const ge::NodePtr& pre_node, int pre_idx, const ge::NodePtr& cur_node, int cur_idx) const;
  bool GetConstValue(const Operator& op, const Tensor& const_tensor, const DataType& dtype,
                     std::vector<int64_t>& const_data);
  bool CheckDynamic(const ge::NodePtr node, int32_t index) const;
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LAYER_NORM_ONNX_FUSION_PASS_H_
