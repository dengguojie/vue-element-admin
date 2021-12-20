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
 * \file lstmp_fusion_pass.h
 * \brief lstmp fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LSTMP_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LSTMP_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class LSTMPFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) override;
 private:
  Status ProcessLSTMWxh(ge::NodePtr fused_node, vector<ge::GeTensorPtr> &tensorPtr);
  void SetTensorDescription(ge::GeTensorDesc &tensorDesc, vector<int64_t> &dims, const ge::Format &format,
                            const ge::DataType &dtype);
  Status CreateConcatNode(ge::ComputeGraph& graph, ge::OpDescPtr& fused_desc, ge::NodePtr& new_node);
  Status CreateTransposeNode(ge::ComputeGraph& graph, ge::GeTensorDesc& input_desc, ge::NodePtr& new_node,
                             std::vector<int32_t>& perm, const std::string& name);
  Status CreateDynamicV3Node(ge::ComputeGraph& graph, ge::OpDescPtr& fused_desc, ge::NodePtr& fused_node,
                             ge::NodePtr& new_node);
  Status CreateConstNode(ge::ComputeGraph& graph, ge::OpDescPtr& fused_desc, ge::NodePtr& new_node);
  Status CreateSplitNode(ge::ComputeGraph& graph, ge::OpDescPtr& dynamicv3_desc, ge::NodePtr& new_node, const std::string& output_name);
  Status AddEdgeForOptionInput(ge::ComputeGraph& graph, ge::NodePtr& fused_node, ge::NodePtr& dynamicv3_node);
  Status AddEdgeForOutput(ge::NodePtr& fused_node, ge::NodePtr& dynamicv3_node, ge::NodePtr& splith_node, ge::NodePtr& splitc_node);
  Status RemoveFusedNode(ge::ComputeGraph& graph, ge::NodePtr& fused_node);
  int CalcInputWxIndex(ge::OpDescPtr& fused_desc);
  const string FUSED_OP_TYPE = "LSTMP";
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_LSTMP_FUSION_PASS_H_
