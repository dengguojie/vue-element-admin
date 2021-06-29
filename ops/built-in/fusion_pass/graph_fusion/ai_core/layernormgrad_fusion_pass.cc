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
 * \file layernormgrad_fusion_pass.cpp
 * \brief LayerNormGrad fusion pass
 *   (LayerNormGrad --> LayerNormXBackpropV2 & LayerNormBetaGammaBackpropV2 in known shape or
 *    LayerNormGrad --> LayerNormXBackprop & LayerNormBetaGammaBackprop in unknown shape)
 */
#include "layernormgrad_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;
namespace fe {

static const char *FUSED_NODE = "LayerNormGrad";

static const std::string PATTERN_FUSEDNODE = "LayerNormGrad";

vector<FusionPattern *> LayerNormGradFusionPass::DefinePatterns() {
  vector < FusionPattern * > patterns;

  FusionPattern *pattern = new(std::nothrow) FusionPattern("LayerNormGradFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
  return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

// vector<ge::NodePtr> &fusionNodes: Store fusion nodes,
//       including newly added nodes and fused but not deleted nodes
Status LayerNormGradFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector <ge::NodePtr> &fusionNodes) {
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
  return PARAM_INVALID);
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode's OpDesc is null, fusion failed."),
  return PARAM_INVALID);
  ge::OpDescPtr layerNormXDesc = AttrUtils::CloneOpDesc(fusedDesc);
  FUSION_PASS_CHECK(
      layerNormXDesc == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.", fusedNode->GetName().c_str()),
  return PARAM_INVALID);
  ge::OpDescPtr layerNormBetaGammaDesc = AttrUtils::CloneOpDesc(fusedDesc);
  FUSION_PASS_CHECK(
      layerNormBetaGammaDesc == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.", fusedNode->GetName().c_str()),
  return PARAM_INVALID);

  if (layerNormXDesc->GetOutputsSize() < 3) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Index is beyond the size[%zu] of output desc", layerNormXDesc->GetInputsSize());
    return NOT_CHANGED;
  }
  if (layerNormBetaGammaDesc->GetOutputsSize() < 3) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Index is beyond the size[%zu] of output desc",
            layerNormBetaGammaDesc->GetInputsSize());
    return NOT_CHANGED;
  }
  if (layerNormBetaGammaDesc->GetInputsSize() < 5) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Index is beyond the size[%zu] of input desc",
            layerNormBetaGammaDesc->GetInputsSize());
    return NOT_CHANGED;
  }

  layerNormXDesc->SetName(fusedDesc->GetName() + "/LayerNormXBackpropV2");
  layerNormBetaGammaDesc->SetName(fusedDesc->GetName() + "/LayerNormBetaGammaBackpropV2");
  layerNormXDesc->SetType("LayerNormXBackpropV2");
  layerNormBetaGammaDesc->SetType("LayerNormBetaGammaBackpropV2");

  OpDescUtils::ClearOutputDesc(layerNormXDesc, 2);
  OpDescUtils::ClearOutputDesc(layerNormBetaGammaDesc, 0);
  ge::GeTensorDesc res_for_gamma_tensor = fusedNode->GetOpDesc()->GetInputDesc(0).Clone();
  res_for_gamma_tensor.SetDataType(ge::DT_FLOAT);
  FUSION_PASS_CHECK(layerNormXDesc->UpdateOutputDesc(1, res_for_gamma_tensor) != GRAPH_SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to update output desc[1] of op[%s]",
                            layerNormXDesc->GetName().c_str()),
  return FAILED);
  std::map <string, uint32_t> output_name_idx_0 = {{"pd_x",          0},
                                                    {"res_for_gamma", 1}};
  std::map <string, uint32_t> output_name_idx_1 = {{"pd_gamma", 0},
                                                    {"pd_beta",  1}};
  layerNormXDesc->UpdateOutputName(output_name_idx_0);
  layerNormBetaGammaDesc->UpdateOutputName(output_name_idx_1);

  // get shape of layerNormBetaGammaDesc's 5th Input, and convert it to attr
  OpDescUtils::ClearInputDesc(layerNormBetaGammaDesc, 4);
  OpDescUtils::ClearInputDesc(layerNormBetaGammaDesc, 3);
  OpDescUtils::ClearInputDesc(layerNormBetaGammaDesc, 2);
  std::vector <int64_t> inputDescShapeData = fusedDesc->GetInputDesc(4).GetShape().GetDims();
  if (!ge::AttrUtils::SetListInt(layerNormBetaGammaDesc, "shape_gamma", inputDescShapeData)) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Set attr shape_gamma for LayerNormBetaGammaBackpropV2 failed.");
  }
  FUSION_PASS_CHECK(layerNormBetaGammaDesc->UpdateInputDesc(1, res_for_gamma_tensor) != GRAPH_SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to update input desc[1] of op[%s]",
                            layerNormBetaGammaDesc->GetName().c_str()),
  return FAILED);
  std::map <string, uint32_t> input_name_idx_0 = {{"dy",            0},
                                                  {"x",             1},
                                                  {"variance",      2},
                                                  {"mean",          3},
                                                  {"gamma",         4}};
  layerNormXDesc->UpdateInputName(input_name_idx_0);
  std::map <string, uint32_t> input_name_idx_1 = {{"dy",            0},
                                                  {"res_for_gamma", 1}};
  layerNormBetaGammaDesc->UpdateInputName(input_name_idx_1);
  ge::NodePtr layerNormXNode = graph.AddNode(layerNormXDesc);
  ge::NodePtr layerNormBetaGammaNode = graph.AddNode(layerNormBetaGammaDesc);
  FUSION_PASS_CHECK(
      layerNormXNode == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.", layerNormXNode->GetName().c_str()),
  return PARAM_INVALID);
  FUSION_PASS_CHECK(layerNormBetaGammaNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                            layerNormBetaGammaNode->GetName().c_str()),
  return PARAM_INVALID);
  for (unsigned int i = 0; i < fusedNode->GetAllInDataAnchors().size(); i++) {
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(i)->GetPeerOutAnchor(),
                                            layerNormXNode->GetInDataAnchor(i)),
        OP_LOGE(FUSED_OP_TYPE.c_str(),
                "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d] failed.",
                fusedNode->GetName().c_str(), i, layerNormXNode->GetName().c_str(), i),
    return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[%d] to fusion node:%s's index[%d].",
            fusedNode->GetName().c_str(), i, layerNormXNode->GetName().c_str(), i);
  }
  for (unsigned int i = 0; i < fusedNode->GetInControlAnchor()->GetPeerOutControlAnchors().size(); i++) {
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInControlAnchor()->GetPeerOutControlAnchors().at(i),
                                            layerNormXNode->GetInControlAnchor()),
        OP_LOGE(FUSED_OP_TYPE.c_str(),
                "Add edge from fused node:%s's control index[%d] to fusion node:%s's control index failed.",
                fusedNode->GetName().c_str(), i, layerNormXNode->GetName().c_str()),
    return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(),
            "Add edge from fused node:%s's control index[%d] to fusion node:%s's control index.",
            fusedNode->GetName().c_str(), i, layerNormXNode->GetName().c_str());
  }
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                          layerNormBetaGammaNode->GetInDataAnchor(0)),
      OP_LOGE(FUSED_OP_TYPE.c_str(),
              "Add edge from fused node:%s's index[0] to fusion node:%s's index[0] failed.",
              fusedNode->GetName().c_str(), layerNormBetaGammaNode->GetName().c_str()),
  return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's index[0] to fusion node:%s's index[0].",
          fusedNode->GetName().c_str(), layerNormBetaGammaNode->GetName().c_str());
  for (unsigned int i = 0; i < fusedNode->GetInControlAnchor()->GetPeerOutControlAnchors().size(); i++) {
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInControlAnchor()->GetPeerOutControlAnchors().at(i),
                                            layerNormBetaGammaNode->GetInControlAnchor()),
        OP_LOGE(FUSED_OP_TYPE.c_str(),
                "Add edge from fused node:%s's control index[%d] to fusion node:%s's control index failed.",
                fusedNode->GetName().c_str(), i, layerNormXNode->GetName().c_str()),
    return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(),
            "Add edge from fused node:%s's control index[%d] to fusion node:%s's control index.",
            fusedNode->GetName().c_str(), i, layerNormXNode->GetName().c_str());
  }
  // add input res_for_gamma_tensor for layerNormBetaGammaNode
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(layerNormXNode->GetOutDataAnchor(1),
                                            layerNormBetaGammaNode->GetInDataAnchor(1)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(),
                            "add adge from layerNormXNode's 2st index output to %s 's 2st index input failed.",
                            layerNormBetaGammaNode->GetName().c_str()),
  return FAILED);

  if (fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The size of layerNormXNode is [%d].",
            fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size());
    for (InDataAnchorPtr &inAnchorPtr : fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(layerNormXNode->GetOutDataAnchor(0), inAnchorPtr),
          OP_LOGE(FUSED_OP_TYPE.c_str(),
                  "Add edge from fused node:%s's index to fusion node:%s's 1st index failed.",
                  fusedNode->GetName().c_str(), layerNormXNode->GetName().c_str()),
      return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's 1st index to fusion node:%s's 1st index.",
              fusedNode->GetName().c_str(), layerNormXNode->GetName().c_str());
    }
  }
  if (fusedNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size() > 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The size of layerNormXNode is [%d].",
            fusedNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size());
    for (InDataAnchorPtr &inAnchorPtr : fusedNode->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(layerNormBetaGammaNode->GetOutDataAnchor(0),
                                                            inAnchorPtr),
                        OP_LOGE(FUSED_OP_TYPE.c_str(),
                                "Add edge from fused node:%s's 2nd index to fusion node:%s's 1st index failed.",
                                fusedNode->GetName().c_str(), layerNormBetaGammaNode->GetName().c_str()),
      return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's 2nd index to fusion node:%s's 1st index.",
              fusedNode->GetName().c_str(), layerNormBetaGammaNode->GetName().c_str());
    }
  }
  if (fusedNode->GetOutDataAnchor(2)->GetPeerInDataAnchors().size() > 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The size of layerNormXNode is [%d].",
            fusedNode->GetOutDataAnchor(2)->GetPeerInDataAnchors().size());
    for (InDataAnchorPtr &inAnchorPtr : fusedNode->GetOutDataAnchor(2)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(layerNormBetaGammaNode->GetOutDataAnchor(1),
                                                            inAnchorPtr),
                        OP_LOGE(FUSED_OP_TYPE.c_str(),
                                "Add edge from fused node:%s's 3rd index to fusion node:%s's 2nd index failed.",
                                fusedNode->GetName().c_str(), layerNormBetaGammaNode->GetName().c_str()),
      return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's 3rd index to fusion node:%s's 2nd index.",
              fusedNode->GetName().c_str(), layerNormBetaGammaNode->GetName().c_str());
    }
  }

  if (fusedNode->GetInControlAnchor() != nullptr) {
    fusedNode->GetInControlAnchor()->UnlinkAll();
  }
  for (auto &inAnchor : fusedNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(fusedNode),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed",
                            fusedNode->GetName().c_str()),
  return FAILED);
  fusionNodes.push_back(layerNormXNode);
  fusionNodes.push_back(layerNormBetaGammaNode);
  return SUCCESS;
}

REGISTER_PASS("LayerNormGradFusionPass", BUILT_IN_GRAPH_PASS, LayerNormGradFusionPass);
}  // namespace fe

