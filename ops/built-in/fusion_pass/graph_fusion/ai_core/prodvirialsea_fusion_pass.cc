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
 * \file prodvirialsea_fusion_pass.cc
 * \brief ProdVirialSeA fusion pass(Parallel ProdVirialSeA)
 */
#include <stdint.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../../../op_proto/util/error_util.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "deep_md_fusion_pass_util.h"
#include "prodvirialsea_fusion_pass.h"

namespace fe {
static const std::string PATTERN_PRODVIRIALSEA = "ProdVirialSeA";
static const std::string OP_TYPE_PRODVIRIALSEA = "ProdVirialSeA";

/*!
 * @brief Define pattern.
 * The graph struct need to adapt and target is shown as follows:
 *
 *        inputs                          inputs
 *          |                             /   \
 *    ProdVirialSeA    ==>    ProdVirialSeA   ProdVirialSeA
 *          |                           | \   / |
 *       outputs                        |  \ /  |
 *                                      |  / \  |
 *                                      add   add
 *                                        \   /
 *                                       outputs
 * @return vector<FusionPattern*> All valid patterns.
 */
vector<FusionPattern*> ProdVirialSeAFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  string passName = "ProdVirialSeAFusionPass";
  FusionPattern* pattern = new (std::nothrow) FusionPattern(passName);
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to new a pattern object."),
                    return patterns);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define pass pattern");
  pattern->AddOpDesc(PATTERN_PRODVIRIALSEA, {OP_TYPE_PRODVIRIALSEA}).SetOutput(PATTERN_PRODVIRIALSEA);
  patterns.push_back(pattern);
  return patterns;
}

/*
 * @brief: parse nodes matched in mapping and call graph DoFusion
 * @param [in] graph: original graph
 * @param [in] mapping: matched pattern
 * @param [out] newNodes: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status ProdVirialSeAFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into ProdVirialSeA fusion pass");

  ge::NodePtr virialNode = GetNodeFromMapping(PATTERN_PRODVIRIALSEA, mapping);
  FUSION_PASS_CHECK(virialNode == nullptr,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Failed to get ProdVirialSeA Node"),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(DeepMdFusionPassUtil::CheckSplitInitInfo(FUSED_OP_TYPE, virialNode) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Failed to check {split_count, split_index}"),
                    return NOT_CHANGED);

  ge::NodePtr virialNodeAic = nullptr;
  ge::NodePtr virialNodeVec = nullptr;
  FUSION_PASS_CHECK(DeepMdFusionPassUtil::SplitNodeToAICoreAndVectorCore(FUSED_OP_TYPE, graph, virialNode,
                                                                         virialNodeAic, virialNodeVec) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to split ProdVirialSeA node"),
                    return FAILED);
  newNodes.push_back(virialNodeAic);
  newNodes.push_back(virialNodeVec);

  ge::NodePtr addVirialNode = nullptr;
  vector<ge::NodePtr> virialNodes = {virialNode, virialNodeAic, virialNodeVec};
  FUSION_PASS_CHECK(DeepMdFusionPassUtil::CreateAddNodeAfterSplitNode(FUSED_OP_TYPE, graph, addVirialNode,
                                                                      virialNodes, 0) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create Add(virial) node"),
                    return FAILED);
  newNodes.push_back(addVirialNode);

  ge::NodePtr addAtomVirialNode = nullptr;
  FUSION_PASS_CHECK(DeepMdFusionPassUtil::CreateAddNodeAfterSplitNode(FUSED_OP_TYPE, graph, addAtomVirialNode,
                                                                      virialNodes, 1) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create Add(atom_virial) node"),
                    return FAILED);
  newNodes.push_back(addAtomVirialNode);

  for (auto inAnchor : virialNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    inAnchor->UnlinkAll();
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(addVirialNode->GetOutDataAnchor(0), inAnchor) != ge::GRAPH_SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from output virial to output node"),
        return FAILED);
  }
  for (auto inAnchor : virialNode->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {
    inAnchor->UnlinkAll();
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(addAtomVirialNode->GetOutDataAnchor(0), inAnchor) != ge::GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Failed to add edge from output atom_virial to output node"),
                      return FAILED);
  }

  FUSION_PASS_CHECK(DeepMdFusionPassUtil::ClearFusedNode(FUSED_OP_TYPE, graph, virialNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to clear ProdVirialSeA node"),
                    return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to ProdVirialSeA fusion pass");
  return SUCCESS;
}
REGISTER_PASS("ProdVirialSeAFusionPass", BUILT_IN_GRAPH_PASS, ProdVirialSeAFusionPass);
}  // namespace fe
