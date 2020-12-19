/* Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_THRESHOLD_RELU_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_THRESHOLD_RELU_FUSION_PASS_H_
#include <string>
#include <vector>

#include "graph/tensor.h"
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "pattern_fusion_util.h"

namespace fe {
class ThresholdReluPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                vector<ge::NodePtr> &fusion_nodes) override;
  Status FusionNode(ge::NodePtr &fused_node, ge::NodePtr &threshold_node,
                    ge::NodePtr &mul_node) const;
  Status CreateThresholdNode(ge::ComputeGraph &graph, ge::NodePtr &fused_node,
                             ge::NodePtr &new_node) const;
  Status CreateMulNode(ge::ComputeGraph &graph, ge::NodePtr &fused_node,
                       ge::NodePtr &new_node) const;
  Status AddEdgeToMulForOut(ge::NodePtr &fused_node,
                            ge::NodePtr &mul_node) const;
  Status RemoveFusedNode(ge::ComputeGraph &graph,
                         ge::NodePtr &fused_node) const;

 private:
  static const std::string PATTERN_FUSEDNODE;
};
}  //  namespace fe

#endif  //   OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_THRESHOLD_RELU_FUSION_PASS_H_
