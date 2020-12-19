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

#ifndef MAX_FUSION_PASS_H
#define MAX_FUSION_PASS_H
#include <string>
#include <vector>

#include "graph/tensor.h"
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "pattern_fusion_util.h"

namespace fe {
class MaxToMaximumPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                vector<ge::NodePtr>& fusion_nodes) override;
  Status CreateNode(ge::ComputeGraph& graph, ge::NodePtr& new_node,
                    const int index) const;
  Status AddEdgeForRoot(ge::NodePtr& fused_node, ge::NodePtr& new_node) const;
  Status AddEdgeForList(ge::NodePtr& fused_node, ge::NodePtr& new_node,
                        ge::NodePtr& curr_node, const int index) const;
  Status AddEdgeForEnd(ge::NodePtr& fused_node, ge::NodePtr& end_node) const;
  Status RemoveFusedNode(ge::ComputeGraph& graph,
                         ge::NodePtr& fused_node) const;
  void GetDataFormatAndType(ge::NodePtr& fused_node);
  Status UpdateDescForRoot(ge::NodePtr& fused_node,
                           ge::NodePtr& new_node) const;
  Status UpdateDescForList(ge::NodePtr& fused_node, ge::NodePtr& new_node,
                           ge::NodePtr& curr_node, const int index) const;
  void GetOutputShape(std::vector<int64_t>& dims,
                      const ge::GeShape input_shape) const;
  Status FusionNode(ge::ComputeGraph& graph, ge::NodePtr& fused_node,
                    vector<ge::NodePtr>& fusion_nodes,
                    const int input_size) const;

 private:
  ge::Format data_format = ge::FORMAT_NCHW;
  ge::DataType data_type = ge::DT_FLOAT;
  static const std::string PATTERN_FUSEDNODE;
  static const int MIN_INPUT_SIZE;
};
}  // namespace fe
#endif  //  OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_MIN_FUSION_PASS_H_
