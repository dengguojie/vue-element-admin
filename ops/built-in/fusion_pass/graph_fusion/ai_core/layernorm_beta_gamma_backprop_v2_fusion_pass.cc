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
 * \file layernorm_beta_gamma_backprop_fusion_pass.cpp
 * \brief clip fusion pass(min --> max)
 */
#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "register/graph_optimizer/fusion_common/graph_pass_util.h"
#include "pattern_fusion_util.h"
#include "layernorm_beta_gamma_backprop_v2_fusion_pass.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"

namespace fe {
static const string PASS_OP_TYPE_LAYERNORM_BETA_GAMMA_BACKPROP = "LayerNormBetaGammaBackpropV2";
static const string PASS_OP_TYPE_CAST = "Cast";

vector<FusionPattern*> LayerNormBetaGammaBackpropV2FusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  return patterns;
}

Status LayerNormBetaGammaBackpropV2FusionPass::Run(ge::ComputeGraph& graph) {
  vector<LayerNormMatchResult> passMatchResultVec;
  fe::Status status = MatchPass(graph, passMatchResultVec);
  if (status != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Fail to match LayerNormBetaGammaBackpropV2FusionPass.");
  }

  if (passMatchResultVec.empty()) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "This graph does not match with pass LayerNormBetaGammaBackpropV2FusionPass.");
    return SUCCESS;
  }
  int32_t effectTimes = 0;
  for (LayerNormMatchResult& matchResult : passMatchResultVec) {
    status = FusionGraphWithPass(graph, matchResult);
    if (status == SUCCESS) {
      effectTimes++;
    }
    OpDescPtr lnOpDesc = matchResult.layerNormBetaGammaBackpropPtr->GetOpDesc();
    OP_LOGI(FUSED_OP_TYPE.c_str(), "LayerNormBetaGammaBackpropV2's 1st output datatype is %d.",
            lnOpDesc->GetOutputDesc(0).GetDataType());
    OP_LOGI(FUSED_OP_TYPE.c_str(), "LayerNormBetaGammaBackpropV2's 1st output origin datatype is %d.",
            lnOpDesc->GetOutputDesc(0).GetOriginDataType());
    OP_LOGI(FUSED_OP_TYPE.c_str(), "LayerNormBetaGammaBackpropV2's 2nd output datatype is %d.",
            lnOpDesc->GetOutputDesc(1).GetDataType());
    OP_LOGI(FUSED_OP_TYPE.c_str(), "LayerNormBetaGammaBackpropV2's 2nd output origin datatype is %d.",
            lnOpDesc->GetOutputDesc(1).GetOriginDataType());
  }
  FusionInfo fusionInfo(graph.GetSessionID(), to_string(graph.GetGraphID()), GetName(),
                        static_cast<int32_t>(passMatchResultVec.size()), effectTimes);
  FusionStatisticRecorder& fusionStatisticInst = FusionStatisticRecorder::Instance();
  fusionStatisticInst.UpdateGraphFusionMatchTimes(fusionInfo);
  fusionStatisticInst.UpdateGraphFusionEffectTimes(fusionInfo);
  OP_LOGD(FUSED_OP_TYPE.c_str(),
          "SessionId[%d], GraphId[%d], GraphFusionPass[%s]: pattern=undefined, matchedTimes=%zu, effectTimes=%zu.",
          graph.GetSessionID(), graph.GetGraphID(), GetName().c_str(), passMatchResultVec.size(), effectTimes);

  return SUCCESS;
}

Status LayerNormBetaGammaBackpropV2FusionPass::MatchPass(ge::ComputeGraph& graph,
                                                       vector<LayerNormMatchResult>& passMatchResultVec) {
  vector<NodePtr> layerNormBetaGammaBackpropNodeVec;
  Status status = GetAllLayerNormBetaGammaBackpropV2Nodes(graph, layerNormBetaGammaBackpropNodeVec);
  if (status != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Fail to get all LayerNormBetaGammaBackpropV2 nodes from graph.");
  }
  if (layerNormBetaGammaBackpropNodeVec.empty()) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "This graph does not contain LayerNormBetaGammaBackpropV2 node.");
    return SUCCESS;
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "This graph has [%d] LayerNormBetaGammaBackpropV2 node.",
          layerNormBetaGammaBackpropNodeVec.size());
  for (NodePtr lnNodePtr : layerNormBetaGammaBackpropNodeVec) {
    LayerNormMatchResult matchResult;
    if (MatchLayerNormBetaGammaBackpropV2Node(lnNodePtr, matchResult) == SUCCESS) {
      passMatchResultVec.push_back(matchResult);
    }
  }
  return SUCCESS;
}

Status LayerNormBetaGammaBackpropV2FusionPass::GetAllLayerNormBetaGammaBackpropV2Nodes(
    ge::ComputeGraph& graph, vector<NodePtr>& layerNormBetaGammaBackpropNodeVec) {
  for (NodePtr node : graph.GetDirectNode()) {
    if (node->GetType() == PASS_OP_TYPE_LAYERNORM_BETA_GAMMA_BACKPROP) {
      layerNormBetaGammaBackpropNodeVec.push_back(node);
    }
  }
  return SUCCESS;
}

Status LayerNormBetaGammaBackpropV2FusionPass::MatchLayerNormBetaGammaBackpropV2Node(NodePtr lnNodePtr,
                                                                                 LayerNormMatchResult& matchResult) {
  // LayerNormBetaGammaBackpropV2 node has epsilon attr
  OpDescPtr lnOpDescPtr = lnNodePtr->GetOpDesc();
  if (!ge::AttrUtils::HasAttr(lnOpDescPtr, "shape_gamma")) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The LayerNormBetaGammaBackpropV2 node does not have shape_gamma attr.");
    return FAILED;
  }

  // ln have 2 input
  if (lnNodePtr->GetAllInDataAnchors().size() != 2) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The LayerNormBetaGammaBackpropV2 node must have 2 input anchor.");
    return FAILED;
  }

  // ln have 2 output
  if (lnNodePtr->GetAllOutDataAnchors().size() != 2) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The LayerNormBetaGammaBackpropV2 node must have 2 output anchor.");
    return FAILED;
  }
  
  ge::GeTensorDesc inputDesc = lnOpDescPtr->GetInputDesc(0);
  if (inputDesc.GetDataType() != DT_FLOAT16) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The LayerNormBetaGammaBackpropV2 node's inputDesc[0] should be float16.");
    return FAILED;
  }

  for (NodePtr nodePtr : lnNodePtr->GetOutDataNodes()) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Out Node Type is [%s]", nodePtr->GetType().c_str());
    if (nodePtr->GetType() == PASS_OP_TYPE_CAST) {
      matchResult.castNodeVec.push_back(nodePtr);
    }
  }

  if (matchResult.castNodeVec.size() != 2) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The LayerNormBetaGammaBackpropV2 node's output  must be 2 Cast node.");
    return FAILED;
  }

  matchResult.layerNormBetaGammaBackpropPtr = lnNodePtr;
  OP_LOGI(
      FUSED_OP_TYPE.c_str(),
      "The LayerNormBetaGammaBackpropV2FusionPass has been matched, name of node is [%s]",
      lnNodePtr->GetName().c_str());
  return SUCCESS;
}

Status LayerNormBetaGammaBackpropV2FusionPass::FusionGraphWithPass(ge::ComputeGraph& graph,
                                                                 LayerNormMatchResult& matchResult) {
  NodePtr lnNodePtr = matchResult.layerNormBetaGammaBackpropPtr;
  FUSION_PASS_CHECK(lnNodePtr == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "lnNodePtr is null, fusion failed."),
                    return PARAM_INVALID);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Begin to fusion pass, the name of LayerNormBetaGammaBackpropV2 is [%s].",
          lnNodePtr->GetName().c_str());

  OpDescPtr layerNormBetaGammaDesc = lnNodePtr->GetOpDesc();
  int index = 0;
  for (NodePtr castNode : matchResult.castNodeVec) {
    FUSION_PASS_CHECK(graph.RemoveNode(castNode) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove cast node[%s] failed.", castNode->GetName().c_str()),
                      return FAILED);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Remove cast node[%s].", castNode->GetName().c_str());
    ge::GeTensorDesc tensorDesc = layerNormBetaGammaDesc->GetOutputDesc(index);
    tensorDesc.SetDataType(ge::DT_FLOAT);
    tensorDesc.SetOriginDataType(ge::DT_FLOAT);
    layerNormBetaGammaDesc->UpdateOutputDesc(index, tensorDesc);
    index++;
  }
  return SUCCESS;
}

Status LayerNormBetaGammaBackpropV2FusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                    vector<ge::NodePtr>& newNodes) {
  return SUCCESS;
}

REGISTER_PASS("LayerNormGradFusionPassBetaGammaV2", BUILT_IN_GRAPH_PASS, LayerNormBetaGammaBackpropV2FusionPass);
}  // namespace fe
