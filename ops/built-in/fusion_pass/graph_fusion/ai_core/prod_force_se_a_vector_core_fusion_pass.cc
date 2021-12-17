/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file prod_force_se_a_vector_core_fusion_pass.cc
 * \brief ProdSeA fusion pass(Parallel ProdForceSeA)
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
#include "prod_force_se_a_vector_core_fusion_pass.h"

namespace fe {

static const std::string PATTERN_PRODVIRIALSEA = "ProdForceSeA";
static const std::string OP_TYPE_PRODVIRIALSEA = "ProdForceSeA";
static const std::string ATTR_OP_SPECIFIED_ENGINE_NAME = "_specified_engine_name";
static const std::string ATTR_OP_SPECIFIED_KERNEL_LIB_NAME = "_specified_kernel_lib_name";
static const std::string ATTR_NAME_STREAM_LABEL = "_stream_label";
static const std::string ATTR_SPLIT_COUNT = "split_count";
static const std::string ATTR_SPLIT_INDEX = "split_index";
static const std::string ATTR_INPUT_NAME_KEY = "_input_name_key";
static const std::string ATTR_INPUT_NAME_VALUE = "_input_name_value";
static const uint32_t VIRIAL_INPUT_SIZE = 5;

/*!
 * @brief Define pattern.
 * The graph struct need to adapt and target is shown as follows:
 *
 *        inputs                       inputs
 *          |                          /   \
 *    ProdForceSeA    ==>    ProdForceSeA   ProdForceSeA
 *          |                           \   /
 *       outputs                         add
 *                                        |
 *                                        |
 *                                      outputs
 * @return vector<FusionPattern*> All valid patterns.
 */
vector<FusionPattern*> ProdForceSeAVectorFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  string passName = "ProdForceSeAVectorFusionPass";
  FusionPattern* pattern = new (std::nothrow) FusionPattern(passName);
  FUSION_PASS_CHECK(pattern == nullptr, CommonRuntimeErrLog(FUSED_OP_TYPE, "Failed to new a pattern object."),
                    return patterns);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define pass pattern");
  pattern->AddOpDesc(PATTERN_PRODVIRIALSEA, {OP_TYPE_PRODVIRIALSEA}).SetOutput(PATTERN_PRODVIRIALSEA);
  patterns.push_back(pattern);
  return patterns;
}

Status ProdForceSeAVectorFusionPass::SplitForceNode(ge::ComputeGraph& graph, ge::NodePtr& forceNode,
                                                ge::NodePtr& forceNodeAiCore, ge::NodePtr& forceNodeVectorCore) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into split ProdForceSeA node.");
  ge::OpDescPtr forceDesc = forceNode->GetOpDesc();
  FUSION_PASS_CHECK(forceDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to get OpDesc of ProdForceSeA."),
                    return PARAM_INVALID);

  ge::OpDescPtr forceDescAic = AttrUtils::CopyOpDesc(forceDesc);
  FUSION_PASS_CHECK(
      forceDescAic == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Faile to create ProdForceSeA(AI Core) op desc."),
      return FAILED);
  forceDescAic->SetName(forceDesc->GetName() + "_aicore");

  // Set attribute of AICore flag.
  ge::AttrUtils::SetStr(forceDescAic, ATTR_OP_SPECIFIED_ENGINE_NAME, "AIcoreEngine");
  ge::AttrUtils::SetStr(forceDescAic, ATTR_OP_SPECIFIED_KERNEL_LIB_NAME, "AIcoreEngine");
  ge::AttrUtils::SetInt(forceDescAic, ATTR_SPLIT_COUNT, (int64_t)2);
  ge::AttrUtils::SetInt(forceDescAic, ATTR_SPLIT_INDEX, 0);

  forceNodeAiCore = graph.AddNode(forceDescAic);
  FUSION_PASS_CHECK(forceNodeAiCore == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add ProdForceSeA(AI Core) to graph"),
                    return FAILED);

  ge::OpDescPtr forceDescVec = AttrUtils::CopyOpDesc(forceDesc);
  FUSION_PASS_CHECK(
      forceDescVec == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Faile to create ProdForceSeA(Vector Core) op desc."),
      return FAILED);
  forceDescVec->SetName(forceDesc->GetName() + "_vectorcore");

  // Set attribute of VectorCore flag.
  ge::AttrUtils::SetStr(forceDescVec, ATTR_OP_SPECIFIED_ENGINE_NAME, "VectorEngine");
  ge::AttrUtils::SetStr(forceDescVec, ATTR_OP_SPECIFIED_KERNEL_LIB_NAME, "VectorEngine");
  ge::AttrUtils::SetInt(forceDescVec, ATTR_SPLIT_COUNT, (int64_t)2);
  ge::AttrUtils::SetInt(forceDescVec, ATTR_SPLIT_INDEX, (int64_t)1);
  ge::AttrUtils::SetStr(forceDescVec, ATTR_NAME_STREAM_LABEL, "VectorEngine");

  forceNodeVectorCore = graph.AddNode(forceDescVec);
  FUSION_PASS_CHECK(forceNodeVectorCore == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add ProdForceSeA(Vector Core) to graph"),
                    return FAILED);
  uint32_t index = 0;
  for (auto inputDesc : forceDesc->GetAllInputsDesc()) {
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(forceNode->GetInDataAnchor(index)->GetPeerOutAnchor(),
                                           forceNodeAiCore->GetInDataAnchor(index)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "Failed to add edge from ProdForceSeA node to ProdForceSeA(AI Core) node."),
        return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(forceNode->GetInDataAnchor(index)->GetPeerOutAnchor(),
                                           forceNodeVectorCore->GetInDataAnchor(index)),
        VECTOR_FUSION_INNER_ERR_REPORT(
            FUSED_OP_TYPE.c_str(), "Failed to add edge from ProdForceSeA node to ProdForceSeA(Vector Core) node."),
        return FAILED);
    index++;
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to split ProdForceSeA node.");
  return SUCCESS;
}

Status ProdForceSeAVectorFusionPass::CreateAddNodes(ge::ComputeGraph& graph, ge::NodePtr& forceNode,
                                               vector<ge::NodePtr>& newForceNodes, ge::NodePtr& addForceNode) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into create Add nodes.");

  std::string addForceNodeName = forceNode->GetName() + "/Add/Force";
  std::shared_ptr<ge::OpDesc> addForceDesc = std::make_shared<ge::OpDesc>(addForceNodeName, "Add");
  FUSION_PASS_CHECK(addForceDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create Add force Node."),
                    return FAILED);

  ge::GeTensorDesc outputDescForce = forceNode->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(addForceDesc->AddInputDesc(0, outputDescForce) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add x1 desc for add force."),
                    return FAILED);
  FUSION_PASS_CHECK(addForceDesc->AddInputDesc(1, outputDescForce) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add x2 desc for add force."),
                    return FAILED);
  FUSION_PASS_CHECK(addForceDesc->AddOutputDesc(outputDescForce) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add y desc for add force."),
                    return FAILED);

  addForceNode = graph.AddNode(addForceDesc);
  FUSION_PASS_CHECK(addForceNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add Add(force) Node to graph"),
                    return FAILED);

  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(newForceNodes[0]->GetOutDataAnchor(0), addForceNode->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from output force to x1."),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(newForceNodes[1]->GetOutDataAnchor(0), addForceNode->GetInDataAnchor(1)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from output force to x2."),
      return FAILED);

  for (auto inAnchor : forceNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    inAnchor->UnlinkAll();
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(addForceNode->GetOutDataAnchor(0), inAnchor),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from output force to output node."),
        return FAILED);
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to create Add nodes.");
  return SUCCESS;
}

Status ProdForceSeAVectorFusionPass::ClearFusedNode(ge::ComputeGraph& graph, ge::NodePtr& node) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into clear fused node.");

  std::string nodeName = node->GetName();
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Node name: %s", nodeName.c_str());
  for (auto inAnchor : node->GetAllInDataAnchors()) {
    if (inAnchor == nullptr) {
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to UnlinkAll of %s for inAnchor is nullptr.", nodeName.c_str());
    } else {
      inAnchor->UnlinkAll();
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Finished to UnlinkAll inDataAnchor of node[%s].", nodeName.c_str());
    }
  }

  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to remove node[%s].", nodeName.c_str()), return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to clear fused node.");
  return SUCCESS;
}

/*
 * @brief: parse nodes matched in mapping and call graph DoFusion
 * @param [in] graph: original graph
 * @param [in] mapping: matched pattern
 * @param [out] newNodes: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status ProdForceSeAVectorFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into ProdForceSeA fusion pass.");

  ge::NodePtr forceNode = GetNodeFromMapping(PATTERN_PRODVIRIALSEA, mapping);
  FUSION_PASS_CHECK(forceNode == nullptr, CommonRuntimeErrLog(FUSED_OP_TYPE, "Failed to get ProdForceSeA Node."),
                    return PARAM_INVALID);

  ge::NodePtr forceNodeAiCore;
  ge::NodePtr forceNodeVectorCore;
  Status virialRet = SplitForceNode(graph, forceNode, forceNodeAiCore, forceNodeVectorCore);
  FUSION_PASS_CHECK(SUCCESS != virialRet,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to split ProdForceSeA node."),
                    return virialRet);

  ge::NodePtr addForceNode;
  vector<ge::NodePtr> newForceNodes = {forceNodeAiCore, forceNodeVectorCore};
  Status addRet = CreateAddNodes(graph, forceNode, newForceNodes, addForceNode);
  FUSION_PASS_CHECK(SUCCESS != addRet,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create Add nodes."),
                    return addRet);

  Status clearFusedNodeRet = ClearFusedNode(graph, forceNode);
  FUSION_PASS_CHECK(SUCCESS != clearFusedNodeRet,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to clear ProdForceSeA node."),
                    return clearFusedNodeRet);

  newNodes.push_back(forceNodeAiCore);
  newNodes.push_back(forceNodeVectorCore);
  newNodes.push_back(addForceNode);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to ProdForceSeA vector fusion pass.");
  return SUCCESS;
}
REGISTER_PASS("ProdForceSeAVectorFusionPass", BUILT_IN_GRAPH_PASS, ProdForceSeAVectorFusionPass);
}  // namespace fe
