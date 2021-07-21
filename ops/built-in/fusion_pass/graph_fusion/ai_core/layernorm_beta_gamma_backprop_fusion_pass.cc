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
#include "layernorm_beta_gamma_backprop_fusion_pass.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"

namespace fe {
static const string PASS_OP_TYPE_LAYERNORM_BETA_GAMMA_BACKPROP = "LayerNormBetaGammaBackprop";
static const string PASS_OP_TYPE_CAST = "Cast";

vector<FusionPattern*> LayerNormBetaGammaBackpropFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  return patterns;
}

Status LayerNormBetaGammaBackpropFusionPass::Run(ge::ComputeGraph& graph) {
  vector<LayerNormMatchResult> passMatchResultVec;
  fe::Status status = MatchPass(graph, passMatchResultVec);
  if (status != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Fail to match LayerNormBetaGammaBackpropFusionPass.");
  }

  if (passMatchResultVec.empty()) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "This graph does not match with pass LayerNormBetaGammaBackpropFusionPass.");
    return SUCCESS;
  }
  int32_t effectTimes = 0;
  for (LayerNormMatchResult& matchResult : passMatchResultVec) {
    status = FusionGraphWithPass(graph, matchResult);
    if (status == SUCCESS) {
      effectTimes++;
    }
    OpDescPtr lnOpDesc = matchResult.layerNormBetaGammaBackpropPtr->GetOpDesc();
    OP_LOGI(FUSED_OP_TYPE.c_str(), "LayerNormBetaGammaBackprop's 1st output datatype is %d.",
            lnOpDesc->GetOutputDesc(0).GetDataType());
    OP_LOGI(FUSED_OP_TYPE.c_str(), "LayerNormBetaGammaBackprop's 1st output origin datatype is %d.",
            lnOpDesc->GetOutputDesc(0).GetOriginDataType());
    OP_LOGI(FUSED_OP_TYPE.c_str(), "LayerNormBetaGammaBackprop's 2nd output datatype is %d.",
            lnOpDesc->GetOutputDesc(1).GetDataType());
    OP_LOGI(FUSED_OP_TYPE.c_str(), "LayerNormBetaGammaBackprop's 2nd output origin datatype is %d.",
            lnOpDesc->GetOutputDesc(1).GetOriginDataType());
  }
  FusionInfo fusionInfo(graph.GetSessionID(), to_string(graph.GetGraphID()), GetName(),
                        static_cast<int32_t>(passMatchResultVec.size()), effectTimes);
  FusionStatisticRecorder& fusionStatisticInst = FusionStatisticRecorder::Instance();
  fusionStatisticInst.UpdateGraphFusionMatchTimes(fusionInfo);
  fusionStatisticInst.UpdateGraphFusionEffectTimes(fusionInfo);
  OP_LOGD(FUSED_OP_TYPE.c_str(),
          "SessionId[%d], GraphId[%d], GraphFusionPass[%s]: pattern=undefined, matchedTimes=%zu, effectTimes=%d.",
          graph.GetSessionID(), graph.GetGraphID(), GetName().c_str(), passMatchResultVec.size(), effectTimes);

  return SUCCESS;
}

Status LayerNormBetaGammaBackpropFusionPass::MatchPass(ge::ComputeGraph& graph,
                                                       vector<LayerNormMatchResult>& passMatchResultVec) {
  vector<NodePtr> layerNormBetaGammaBackpropNodeVec;
  Status status = GetAllLayerNormBetaGammaBackpropNodes(graph, layerNormBetaGammaBackpropNodeVec);
  if (status != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Fail to get all LayerNormBetaGammaBackprop nodes from graph.");
  }
  if (layerNormBetaGammaBackpropNodeVec.empty()) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "This graph does not contain LayerNormBetaGammaBackprop node.");
    return SUCCESS;
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "This graph has [%d] LayerNormBetaGammaBackprop node.",
          layerNormBetaGammaBackpropNodeVec.size());
  for (NodePtr lnNodePtr : layerNormBetaGammaBackpropNodeVec) {
    LayerNormMatchResult matchResult;
    if (MatchLayerNormBetaGammaBackpropNode(lnNodePtr, matchResult) == SUCCESS) {
      passMatchResultVec.push_back(matchResult);
    }
  }
  return SUCCESS;
}

Status LayerNormBetaGammaBackpropFusionPass::GetAllLayerNormBetaGammaBackpropNodes(
    ge::ComputeGraph& graph, vector<NodePtr>& layerNormBetaGammaBackpropNodeVec) {
  for (NodePtr node : graph.GetDirectNode()) {
    if (node->GetType() == PASS_OP_TYPE_LAYERNORM_BETA_GAMMA_BACKPROP) {
      layerNormBetaGammaBackpropNodeVec.push_back(node);
    }
  }
  return SUCCESS;
}

Status LayerNormBetaGammaBackpropFusionPass::MatchLayerNormBetaGammaBackpropNode(NodePtr lnNodePtr,
                                                                                 LayerNormMatchResult& matchResult) {
  // LayerNormBetaGammaBackprop node has epsilon attr
  OpDescPtr lnOpDescPtr = lnNodePtr->GetOpDesc();
  if (!ge::AttrUtils::HasAttr(lnOpDescPtr, "shape_gamma")) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The LayerNormBetaGammaBackprop node does not have shape_gamma attr.");
    return FAILED;
  }

  // ln have 4 input
  if (lnNodePtr->GetAllInDataAnchors().size() != 4) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The LayerNormBetaGammaBackprop node must have 4 input anchor.");
    return FAILED;
  }

  // bn have 3 output
  if (lnNodePtr->GetAllOutDataAnchors().size() != 2) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The LayerNormBetaGammaBackprop node must have 2 output anchor.");
    return FAILED;
  }

  for (auto inputDesc : lnOpDescPtr->GetAllInputsDesc()) {
    if (inputDesc.GetDataType() != DT_FLOAT16) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "The LayerNormBetaGammaBackprop node's inputDesc should be float16.");
      return FAILED;
    }
  }

  for (NodePtr nodePtr : lnNodePtr->GetOutDataNodes()) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Out Node Type is [%s]", nodePtr->GetType().c_str());
    if (nodePtr->GetType() == PASS_OP_TYPE_CAST) {
      matchResult.castNodeVec.push_back(nodePtr);
    }
  }

  if (matchResult.castNodeVec.size() != 2) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The LayerNormBetaGammaBackprop node's output  must be 2 Cast node.");
    return FAILED;
  }

  matchResult.layerNormBetaGammaBackpropPtr = lnNodePtr;
  OP_LOGI(
      FUSED_OP_TYPE.c_str(),
      "The LayerNormBetaGammaBackpropFusionPass has been matched, the name of LayerNormBetaGammaBackprop node is [%s]",
      lnNodePtr->GetName().c_str());
  return SUCCESS;
}

Status LayerNormBetaGammaBackpropFusionPass::FusionGraphWithPass(ge::ComputeGraph& graph,
                                                                 LayerNormMatchResult& matchResult) {
  NodePtr lnNodePtr = matchResult.layerNormBetaGammaBackpropPtr;
  FUSION_PASS_CHECK(lnNodePtr == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "lnNodePtr is null, fusion failed."),
                    return PARAM_INVALID);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Begin to fusion pass, the name of LayerNormBetaGammaBackprop is [%s].",
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

Status LayerNormBetaGammaBackpropFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                    vector<ge::NodePtr>& newNodes) {
  return SUCCESS;
}

REGISTER_PASS("LayerNormGradFusionPassBetaGamma", BUILT_IN_GRAPH_PASS, LayerNormBetaGammaBackpropFusionPass);
}  // namespace fe
