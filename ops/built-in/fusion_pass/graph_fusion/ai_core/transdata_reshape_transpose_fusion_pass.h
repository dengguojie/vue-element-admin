/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
 * \file transdata_reshape_transdata_transpose_fusion_pass.h
 * \brief transdataTranspose fusion pass(Transdata-Reshape-Transdata->Transpose)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_TRANSDATA_RESHAPE_TRANSPOSE_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_TRANSDATA_RESHAPE_TRANSPOSE_FUSION_PASS_H_

#include <string>
#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class TransdataReshapeTransposeFusionPass : public PatternFusionBasePass {
 protected:
  std::vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) override;

 private:
  bool VerifyFusedNode(const ge::NodePtr &transdata_node1, const ge::NodePtr &reshape_node,
                       const ge::NodePtr &transdata_node2) const;

  ge::NodePtr CreateTransposeNode(const ge::NodePtr &transdata1_node, const ge::NodePtr &transdata2_node,
                                  ge::ComputeGraph &graph);

  bool RelinkEdges(ge::NodePtr &transdata_node1, ge::NodePtr &reformat_node1, ge::NodePtr &reshape_node,
                   ge::NodePtr &reformat_node2, ge::NodePtr &transdata_node2, ge::NodePtr &transpose_node,
                   ge::ComputeGraph &graph);

  bool UnLinkDataEdges(const ge::NodePtr &transdata_node1, const ge::NodePtr &reformat_node1,
                       const ge::NodePtr &reshape_node, const ge::NodePtr &reformat_node2,
                       ge::ComputeGraph &graph);

  bool RelinkControlEdges(const ge::NodePtr &src_node, const ge::NodePtr &dst_node) const;

  const string FUSED_OP_TYPE = "TransdataReshapeTransposeFusionPass";
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_TRANSDATA_RESHAPE_TRANSPOSE_FUSION_PASS_H_
