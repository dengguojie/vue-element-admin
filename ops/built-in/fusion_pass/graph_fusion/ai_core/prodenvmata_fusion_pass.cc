/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file prodenvmata_fusion_pass.cc
 * \brief ProdEnvMatA fusion pass(Parallel ProdEnvMatA)
 */
#include <cstdint>

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
#include "prodenvmata_fusion_pass.h"

namespace fe {
static const std::string PATTERN_PRODENVMATA = "ProdEnvMatA";
static const std::string OP_TYPE_PRODENVMATA = "ProdEnvMatA";
static const std::string ATTR_OP_SPECIFIED_ENGINE_NAME = "_specified_engine_name";
static const std::string ATTR_OP_SPECIFIED_KERNEL_LIB_NAME = "_specified_kernel_lib_name";
static const std::string ATTR_NAME_STREAM_LABEL = "_stream_label";
static const std::string ATTR_SPLIT_COUNT = "split_count";
static const std::string ATTR_SPLIT_INDEX = "split_index";
const std::string FUSED_OP_TYPE = "ProdEnvMatA";

int32_t nall = 0;
int32_t nloc = 0;
int32_t nnei = 0;
int32_t nsample = 0;

const int64_t THIRD_CONCAT_NODE_INDEX = 2;
const int64_t FORTH_CONCAT_NODE_INDEX = 3;
const int64_t SPLIT_NODE_COUNT_VALUE = 2;
const int64_t THIRD_OUTPUT_INDEX = 2;
const int64_t FORTH_OUTPUT_INDEX = 3;
const int64_t CONCAT_NUMS = 2;
const int64_t DESCRIPT_LAST_DIM = 4;
const int64_t DERIV_LAST_DIM = 12;
const int64_t RIJ_LAST_DIM = 3;
/*!
 * @brief Define pattern.
 * The graph struct need to adapt and target is shown as follows:
 *
 *        inputs                          inputs
 *          |                             /   \
 *    ProdEnvMatA    ==>    ProdEnvMatA   ProdEnvMatA
 *          |                           | \   / |
 *       outputs                        |  \ /  |
 *                                      |  / \  |
 *                                    concat   concat
 *                                        \   /
 *                                       outputs
 * @return vector<FusionPattern*> All valid patterns.
 */
vector<FusionPattern*> ProdEnvMatAFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  string passName = "ProdEnvMatAFusionPass";
  FusionPattern* pattern = new (std::nothrow) FusionPattern(passName);
  FUSION_PASS_CHECK(pattern == nullptr, CommonRuntimeErrLog(FUSED_OP_TYPE, "Failed to new a pattern object."),
                    return patterns);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define pass pattern");
  pattern->AddOpDesc(PATTERN_PRODENVMATA, {OP_TYPE_PRODENVMATA}).SetOutput(PATTERN_PRODENVMATA);
  patterns.push_back(pattern);
  return patterns;
}

Status ProdEnvMatAFusionPass::SplitEnvmatNode(ge::ComputeGraph& graph, ge::NodePtr& envmatNode,
                                              ge::NodePtr& envmatNodeAiCore, ge::NodePtr& envmatNodeVectorCore) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into split ProdEnvMatA node.");
  ge::OpDescPtr envmatDesc = envmatNode->GetOpDesc();
  FUSION_PASS_CHECK(envmatDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to get OpDesc of ProdEnvMatA."),
                    return PARAM_INVALID);

  ge::OpDescPtr envmatDescAic = AttrUtils::CopyOpDesc(envmatDesc);

  FUSION_PASS_CHECK(
      envmatDescAic == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Faile to create ProdEnvMatA(AI Core) op desc."),
      return FAILED);

  if (nall != -1) {
    int32_t aic_nloc = (nloc / SPLIT_NODE_COUNT_VALUE) + (nloc % SPLIT_NODE_COUNT_VALUE);
    ge::GeShape output1Shape({nsample, aic_nloc * nnei * DESCRIPT_LAST_DIM});
    ge::GeTensorDesc output1TensorDesc = ge::GeTensorDesc(output1Shape, ge::FORMAT_ND, ge::DT_FLOAT);
    output1TensorDesc.SetOriginShape(output1Shape);
    output1TensorDesc.SetOriginFormat(ge::FORMAT_ND);

    ge::GeShape output2Shape({nsample, aic_nloc * nnei * DERIV_LAST_DIM});
    ge::GeTensorDesc output2TensorDesc = ge::GeTensorDesc(output2Shape, ge::FORMAT_ND, ge::DT_FLOAT);
    output2TensorDesc.SetOriginShape(output2Shape);
    output2TensorDesc.SetOriginFormat(ge::FORMAT_ND);

    ge::GeShape output3Shape({nsample, aic_nloc * nnei * RIJ_LAST_DIM});
    ge::GeTensorDesc output3TensorDesc = ge::GeTensorDesc(output3Shape, ge::FORMAT_ND, ge::DT_FLOAT);
    output3TensorDesc.SetOriginShape(output3Shape);
    output3TensorDesc.SetOriginFormat(ge::FORMAT_ND);

    ge::GeShape output4Shape({nsample, aic_nloc * nnei});
    ge::GeTensorDesc output4TensorDesc = ge::GeTensorDesc(output4Shape, ge::FORMAT_ND, ge::DT_INT32);
    output4TensorDesc.SetOriginShape(output4Shape);
    output4TensorDesc.SetOriginFormat(ge::FORMAT_ND);
    FUSION_PASS_CHECK(SUCCESS != envmatDescAic->UpdateOutputDesc("descrpt", output1TensorDesc),
                      OP_LOGE(FUSED_OP_TYPE.c_str(),
                              "Update aicore node descrpt desc failed!"),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != envmatDescAic->UpdateOutputDesc("descrpt_deriv", output2TensorDesc),
                      OP_LOGE(FUSED_OP_TYPE.c_str(),
                              "Update aicore node descrpt_deriv desc failed!"),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != envmatDescAic->UpdateOutputDesc("rij", output3TensorDesc),
                      OP_LOGE(FUSED_OP_TYPE.c_str(),
                              "Update aicore node rij desc failed!"),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != envmatDescAic->UpdateOutputDesc("nlist", output4TensorDesc),
                      OP_LOGE(FUSED_OP_TYPE.c_str(),
                              "Update aicore node nlist desc failed!"),
                      return FAILED);
  }
  envmatDescAic->SetName(envmatDesc->GetName() + "_aicore");

  // Set attribute of AICore flag.
  ge::AttrUtils::SetStr(envmatDescAic, ATTR_OP_SPECIFIED_ENGINE_NAME, "AIcoreEngine");
  ge::AttrUtils::SetStr(envmatDescAic, ATTR_OP_SPECIFIED_KERNEL_LIB_NAME, "AIcoreEngine");
  ge::AttrUtils::SetInt(envmatDescAic, ATTR_SPLIT_COUNT, SPLIT_NODE_COUNT_VALUE);
  ge::AttrUtils::SetInt(envmatDescAic, ATTR_SPLIT_INDEX, 0);

  envmatNodeAiCore = graph.AddNode(envmatDescAic);
  FUSION_PASS_CHECK(envmatNodeAiCore == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add ProdEnvMatA(AI Core) to graph"),
                    return FAILED);

  ge::OpDescPtr envmatDescVec = AttrUtils::CopyOpDesc(envmatDesc);
  FUSION_PASS_CHECK(
      envmatDescVec == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Faile to create ProdEnvMatA(Vector Core) op desc."),
      return FAILED);

  if (nall != -1) {
    int32_t vec_nloc = nloc / SPLIT_NODE_COUNT_VALUE;
    ge::GeShape output1Shape({nsample, vec_nloc * nnei * DESCRIPT_LAST_DIM});
    ge::GeTensorDesc output1TensorDesc = ge::GeTensorDesc(output1Shape, ge::FORMAT_ND, ge::DT_FLOAT);
    output1TensorDesc.SetOriginShape(output1Shape);
    output1TensorDesc.SetOriginFormat(ge::FORMAT_ND);

    ge::GeShape output2Shape({nsample, vec_nloc * nnei * DERIV_LAST_DIM});
    ge::GeTensorDesc output2TensorDesc = ge::GeTensorDesc(output2Shape, ge::FORMAT_ND, ge::DT_FLOAT);
    output2TensorDesc.SetOriginShape(output2Shape);
    output2TensorDesc.SetOriginFormat(ge::FORMAT_ND);

    ge::GeShape output3Shape({nsample, vec_nloc * nnei * RIJ_LAST_DIM});
    ge::GeTensorDesc output3TensorDesc = ge::GeTensorDesc(output3Shape, ge::FORMAT_ND, ge::DT_FLOAT);
    output3TensorDesc.SetOriginShape(output3Shape);
    output3TensorDesc.SetOriginFormat(ge::FORMAT_ND);

    ge::GeShape output4Shape({nsample, vec_nloc * nnei});
    ge::GeTensorDesc output4TensorDesc = ge::GeTensorDesc(output4Shape, ge::FORMAT_ND, ge::DT_INT32);
    output4TensorDesc.SetOriginShape(output4Shape);
    output4TensorDesc.SetOriginFormat(ge::FORMAT_ND);
    FUSION_PASS_CHECK(SUCCESS != envmatDescVec->UpdateOutputDesc("descrpt", output1TensorDesc),
                      OP_LOGE(FUSED_OP_TYPE.c_str(),
                              "Update vector node descrpt desc failed!"),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != envmatDescVec->UpdateOutputDesc("descrpt_deriv", output2TensorDesc),
                      OP_LOGE(FUSED_OP_TYPE.c_str(),
                              "Update vector node descrpt_deriv desc failed!"),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != envmatDescVec->UpdateOutputDesc("rij", output3TensorDesc),
                      OP_LOGE(FUSED_OP_TYPE.c_str(),
                              "Update vector node rij desc failed!"),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != envmatDescVec->UpdateOutputDesc("nlist", output4TensorDesc),
                      OP_LOGE(FUSED_OP_TYPE.c_str(),
                              "Update vector node nlist desc failed!"),
                      return FAILED);
  }

  envmatDescVec->SetName(envmatDesc->GetName() + "_vectorcore");

  // Set attribute of VectorCore flag.
  ge::AttrUtils::SetStr(envmatDescVec, ATTR_OP_SPECIFIED_ENGINE_NAME, "VectorEngine");
  ge::AttrUtils::SetStr(envmatDescVec, ATTR_OP_SPECIFIED_KERNEL_LIB_NAME, "VectorEngine");
  ge::AttrUtils::SetInt(envmatDescVec, ATTR_SPLIT_COUNT, SPLIT_NODE_COUNT_VALUE);
  ge::AttrUtils::SetInt(envmatDescVec, ATTR_SPLIT_INDEX, (int64_t)1);
  ge::AttrUtils::SetStr(envmatDescVec, ATTR_NAME_STREAM_LABEL, "VectorEngine");

  envmatNodeVectorCore = graph.AddNode(envmatDescVec);
  FUSION_PASS_CHECK(envmatNodeVectorCore == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add ProdEnvMatA(Vector Core) to graph"),
                    return FAILED);
  uint32_t index = 0;
  for (auto inputDesc : envmatDesc->GetAllInputsDesc()) {
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(envmatNode->GetInDataAnchor(index)->GetPeerOutAnchor(),
                                           envmatNodeAiCore->GetInDataAnchor(index)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "Failed to add edge from ProdEnvMatA node to ProdEnvMatA(AI Core) node."),
        return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(envmatNode->GetInDataAnchor(index)->GetPeerOutAnchor(),
                                           envmatNodeVectorCore->GetInDataAnchor(index)),
        VECTOR_FUSION_INNER_ERR_REPORT(
            FUSED_OP_TYPE.c_str(), "Failed to add edge from ProdEnvMatA node to ProdEnvMatA(Vector Core) node."),
        return FAILED);
    index++;
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to split ProdEnvMatA node.");
  return SUCCESS;
}

Status ProdEnvMatAFusionPass::CreateConcatNode(ge::ComputeGraph& graph, ge::NodePtr& envmatNode,
                                               vector<ge::NodePtr>& newEnvmatNodes,
                                               ge::NodePtr& cocatNode,
                                               const std::string& nodeName, int32_t outputIndex){
  ge::OpDescPtr concatDesc = nullptr;
  std::string concatNodeName = envmatNode->GetName() + "/" + nodeName + "/Concat";
  FUSION_PASS_MAKE_SHARED(concatDesc = std::make_shared<ge::OpDesc>(concatNodeName,
                                                                    "ConcatD"),
                                                                    return FAILED);
  FUSION_PASS_CHECK(concatDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create Concat envmat Node."),
                    return FAILED);
  ge::GeTensorDesc outputDescEnvmat = envmatNode->GetOpDesc()->GetOutputDesc(outputIndex);
  ge::GeTensorDesc inputX1Desc = newEnvmatNodes[0]->GetOpDesc()->GetOutputDesc(outputIndex);
  ge::GeTensorDesc inputX2Desc = newEnvmatNodes[1]->GetOpDesc()->GetOutputDesc(outputIndex);

  FUSION_PASS_CHECK(concatDesc->AddInputDesc("x1", inputX1Desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add x1 desc for Concat envmat."),
                    return FAILED);
  FUSION_PASS_CHECK(concatDesc->AddInputDesc("x2", inputX2Desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add x2 desc for Concat envmat."),
                    return FAILED);
  FUSION_PASS_CHECK(concatDesc->AddOutputDesc(outputDescEnvmat) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add y desc for Concat envmat."),
                    return FAILED);
  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 1);
  ge::AttrUtils::SetInt(concatDesc, "N", CONCAT_NUMS);
  cocatNode = graph.AddNode(concatDesc);
  FUSION_PASS_CHECK(cocatNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add Concat(envmat) Node to graph"),
                    return FAILED);

  return SUCCESS;
}

Status ProdEnvMatAFusionPass::CreateConcatNodes(ge::ComputeGraph& graph, ge::NodePtr& envmatNode,
                                                vector<ge::NodePtr>& newEnvmatNodes,
                                                vector<ge::NodePtr>& newConcatNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into create Concat nodes.");
  std::string concatEnvmatDescrptName = "concatEnvmatDescrpt";
  CreateConcatNode(graph, envmatNode, newEnvmatNodes, newConcatNodes[0],
                   concatEnvmatDescrptName, 0);

  std::string concatEnvmatDescrptDerivName = "concatEnvmatDescrptDeriv";
  CreateConcatNode(graph, envmatNode, newEnvmatNodes, newConcatNodes[1],
                   concatEnvmatDescrptDerivName, 1);

  std::string concatEnvmatRijName = "concatEnvmatRij";
  CreateConcatNode(graph, envmatNode, newEnvmatNodes,
                   newConcatNodes[THIRD_CONCAT_NODE_INDEX], concatEnvmatRijName, THIRD_CONCAT_NODE_INDEX);

  std::string concatEnvmatNlistName = "concatEnvmatNlist";
  CreateConcatNode(graph, envmatNode, newEnvmatNodes, newConcatNodes[FORTH_CONCAT_NODE_INDEX],
                   concatEnvmatNlistName, FORTH_CONCAT_NODE_INDEX);

  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(newEnvmatNodes[0]->GetOutDataAnchor(0),
                                         newConcatNodes[0]->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from output descrpt to x1."),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(newEnvmatNodes[1]->GetOutDataAnchor(0),
                                         newConcatNodes[0]->GetInDataAnchor(1)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from output descrpt to x2."),
      return FAILED);

  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(newEnvmatNodes[0]->GetOutDataAnchor(1),
                                         newConcatNodes[1]->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from output descrpt_deriv to x1."),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(newEnvmatNodes[1]->GetOutDataAnchor(1),
                                         newConcatNodes[1]->GetInDataAnchor(1)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from output descrpt_deriv to x2."),
      return FAILED);

  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(newEnvmatNodes[0]->GetOutDataAnchor(THIRD_OUTPUT_INDEX),
                                         newConcatNodes[THIRD_CONCAT_NODE_INDEX]->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from output rij to x1."),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(newEnvmatNodes[1]->GetOutDataAnchor(THIRD_OUTPUT_INDEX),
                                         newConcatNodes[THIRD_CONCAT_NODE_INDEX]->GetInDataAnchor(1)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from output rij to x2."),
      return FAILED);

  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(newEnvmatNodes[0]->GetOutDataAnchor(FORTH_OUTPUT_INDEX),
                                         newConcatNodes[FORTH_CONCAT_NODE_INDEX]->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from output nlist to x1."),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(newEnvmatNodes[1]->GetOutDataAnchor(FORTH_OUTPUT_INDEX),
                                         newConcatNodes[FORTH_CONCAT_NODE_INDEX]->GetInDataAnchor(1)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from output nlist to x2."),
      return FAILED);

  for (auto inAnchor : envmatNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    inAnchor->UnlinkAll();
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(newConcatNodes[0]->GetOutDataAnchor(0), inAnchor),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "Failed to add edge from output concatEnvmatDescrptNode to output node."),
        return FAILED);
  }
  for (auto inAnchor : envmatNode->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {
    inAnchor->UnlinkAll();
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(newConcatNodes[1]->GetOutDataAnchor(0), inAnchor),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Failed to add edge from output concatNode to output node."),
                      return FAILED);
  }
  for (auto inAnchor : envmatNode->GetOutDataAnchor(THIRD_OUTPUT_INDEX)->GetPeerInDataAnchors()) {
    inAnchor->UnlinkAll();
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(newConcatNodes[THIRD_CONCAT_NODE_INDEX]->GetOutDataAnchor(0), inAnchor),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from concatNode to output node."),
        return FAILED);
  }
  for (auto inAnchor : envmatNode->GetOutDataAnchor(FORTH_OUTPUT_INDEX)->GetPeerInDataAnchors()) {
    inAnchor->UnlinkAll();
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(newConcatNodes[FORTH_CONCAT_NODE_INDEX]->GetOutDataAnchor(0),
                                                         inAnchor),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Failed to add edge from output concatNode to output node."),
                      return FAILED);
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to create Concat nodes.");
  return SUCCESS;
}

Status ProdEnvMatAFusionPass::ClearFusedNode(ge::ComputeGraph& graph, ge::NodePtr& node) {
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
Status ProdEnvMatAFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into ProdEnvMatA fusion pass.");

  ge::NodePtr envmatNode = GetNodeFromMapping(PATTERN_PRODENVMATA, mapping);
  FUSION_PASS_CHECK(envmatNode == nullptr, CommonRuntimeErrLog(FUSED_OP_TYPE, "Failed to get ProdEnvMatA Node."),
                    return PARAM_INVALID);

  nall = envmatNode->GetOpDesc()->GetInputDesc(1).GetShape().GetDim(1);
  nsample = envmatNode->GetOpDesc()->GetInputDesc(1).GetShape().GetDim(0);
  int32_t nlistDim = envmatNode->GetOpDesc()->GetOutputDesc(FORTH_OUTPUT_INDEX).GetShape().GetDim(1);
  vector<int32_t> sel_a = {};
  ge::AttrUtils::GetListInt(envmatNode->GetOpDesc(), "sel_a", sel_a);
  for (size_t i = 0; i < sel_a.size(); i++) {
     nnei = nnei + sel_a[i];
  }
  nloc = nlistDim / nnei;

  ge::NodePtr envmatNodeAiCore = nullptr;
  ge::NodePtr envmatNodeVectorCore = nullptr;

  Status envmatRet = SplitEnvmatNode(graph, envmatNode, envmatNodeAiCore, envmatNodeVectorCore);
  FUSION_PASS_CHECK(SUCCESS != envmatRet,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to split ProdEnvMatA node."),
                    return envmatRet);

  ge::NodePtr concatEnvmatDescrptNode = nullptr;
  ge::NodePtr concatEnvmatDescrptDerivNode = nullptr;
  ge::NodePtr concatEnvmatRijNode = nullptr;
  ge::NodePtr concatEnvmatNlistNode = nullptr;

  vector<ge::NodePtr> newEnvmatNodes = {envmatNodeAiCore, envmatNodeVectorCore};
  vector<ge::NodePtr> newConcatNodes = {concatEnvmatDescrptNode, concatEnvmatDescrptDerivNode,
                                        concatEnvmatRijNode, concatEnvmatNlistNode};
  Status addRet = CreateConcatNodes(graph, envmatNode, newEnvmatNodes,
                                    newConcatNodes);
  FUSION_PASS_CHECK(SUCCESS != addRet,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create Concat nodes."),
                    return addRet);

  Status clearFusedNodeRet = ClearFusedNode(graph, envmatNode);
  FUSION_PASS_CHECK(SUCCESS != clearFusedNodeRet,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to clear ProdEnvMatA node."),
                    return clearFusedNodeRet);

  newNodes.push_back(envmatNodeAiCore);
  newNodes.push_back(envmatNodeVectorCore);
  newNodes.push_back(newConcatNodes[0]);
  newNodes.push_back(newConcatNodes[1]);
  newNodes.push_back(newConcatNodes[THIRD_CONCAT_NODE_INDEX]);
  newNodes.push_back(newConcatNodes[FORTH_CONCAT_NODE_INDEX]);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to ProdEnvMatA fusion pass.");
  return SUCCESS;
}
REGISTER_PASS("ProdEnvMatAFusionPass", BUILT_IN_GRAPH_PASS, ProdEnvMatAFusionPass);
}  // namespace fe
