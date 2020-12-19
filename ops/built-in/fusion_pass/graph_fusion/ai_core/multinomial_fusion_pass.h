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

#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_MULTINOMIAL_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_MULTINOMIAL_FUSION_PASS_H_
#include <string>
#include <vector>

#include "graph/tensor.h"
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "pattern_fusion_util.h"

namespace fe {
class MultinomialPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                vector<ge::NodePtr> &fusion_nodes) override;
  Status CreateMultinomialNode(ge::ComputeGraph &graph, ge::NodePtr &fused_node,
                               ge::NodePtr &multinomial_node) const;
  Status CreateConstNode(ge::ComputeGraph &graph, ge::NodePtr &fused_node,
                         ge::NodePtr &const_node) const;
  Status AddEdgeForOut(ge::NodePtr &fused_node,
                       ge::NodePtr &multinomial_node) const;
  Status RemoveFusedNode(ge::ComputeGraph &graph,
                         ge::NodePtr &fused_node) const;

 private:
  static const std::string PATTERN_FUSEDNODE;
  static const int DATA_TYPE_INT64_ONNX;
};
}  // namespace fe
#endif  //  OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_MULTINOMIAL_FUSION_PASS_H_
