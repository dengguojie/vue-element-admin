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
 * \file softmax_onnx_fusion.h
 * \brief SoftmaxWithDropOutDoMask fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_SOFTMAXV2_ONNX_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_SOFTMAXV2_ONNX_FUSION_PASS_H_

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ASoftmaxFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;

 private:
  bool CheckIsNeedFusion(ge::NodePtr& fused_node);
  Status CreateFlattenNode(ge::ComputeGraph& graph, ge::NodePtr& fused_node, ge::NodePtr& new_node);
  Status CreateReshapeNode(ge::ComputeGraph& graph, ge::NodePtr& fused_node, ge::NodePtr& flatten_node,
                           ge::NodePtr& new_node);
  Status CreateConstNode(ge::ComputeGraph& graph, ge::NodePtr& fused_node, ge::NodePtr& new_node);
  const string FUSED_OP_TYPE = "SoftmaxV2Flatten";
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_SOFTMAXV2_ONNX_FUSION_PASS_H_