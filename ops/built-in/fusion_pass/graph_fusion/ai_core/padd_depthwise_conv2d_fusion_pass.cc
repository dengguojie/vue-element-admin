/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file padd_depthwise_conv2d_fusion_pass.cpp
 * \brief padd depthwise_conv2d fusion pass
 */
#include <memory>
#include <string>
#include "padd_depthwise_conv2d_fusion_pass.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph/utils/graph_utils.h"

namespace fe {

static const string PATTERN_INPUTS1 = "input1";
static const string PATTERN_PADD = "padd";
static const string PATTERN_DEPTHWISECONV2D = "depthwise_conv2d";
static const string PADD = "PadD";
static const string PADDINGS = "paddings";
static const string PADS = "pads";
static const string PADDING = "padding";
static const string DEPTHWISECONV2D = "DepthwiseConv2D";
static const int DIM_NUM4 = 4;
static const int DIRECTION_COUNT = 2;
vector<FusionPattern*> PaddDepthwiseConv2dFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define PaddDepthwiseConv2dFusionPass pattern begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("PaddDepthwiseConv2dFusionPass");

  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_PADD, {PADD})
      .AddOpDesc(PATTERN_DEPTHWISECONV2D, {DEPTHWISECONV2D})
      .AddOpDesc(PATTERN_INPUTS1)
      .SetInputs(PATTERN_DEPTHWISECONV2D, {PATTERN_PADD, PATTERN_INPUTS1})
      .SetOutput(PATTERN_DEPTHWISECONV2D);
  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define PaddDepthwiseConv2dFusionPass pattern end");
  return patterns;
}

Status PaddDepthwiseConv2dFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                             vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define PaddDepthwiseConv2dFusionPass fusion begin");
  ge::NodePtr paddNode = GetNodeFromMapping(PATTERN_PADD, mapping);
  FUSION_PASS_CHECK(paddNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "padD Node is null, fusion failed."),
                    return PARAM_INVALID);

  ge::NodePtr depthwiseConv2dNode = GetNodeFromMapping(PATTERN_DEPTHWISECONV2D, mapping);
  FUSION_PASS_CHECK(depthwiseConv2dNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "DepthwiseConv2D Node is null, fusion failed."),
                    return PARAM_INVALID);

  Operator op = ge::OpDescUtils::CreateOperatorFromNode(depthwiseConv2dNode);
  std::string paddingMode;
  if (op.GetAttr("padding", paddingMode) != ge::GRAPH_SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "GetOpAttr DepthwiseConv2D padding failed!");
    return PARAM_INVALID;
  }

  if (paddingMode != "VALID") {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "PaddDepthwiseConv2dFusion can only support VALID padding mode.");
    return NOT_CHANGED;
  }

  int64_t convCount = 0;
  for (auto peerInDataAnchor : paddNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    ge::NodePtr nextNode = peerInDataAnchor->GetOwnerNode();
    if (nextNode->GetType() == DEPTHWISECONV2D) {
      convCount++;
    }
  }
  FUSION_PASS_CHECK(convCount > 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(),
                            "Padnode have multiple depthwise_conv2d outputs,"
                            " can not fusion."),
                    return NOT_CHANGED);
  vector<vector<int64_t>> paddings;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListListInt(paddNode->GetOpDesc(), PADDINGS, paddings),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Get paddNode paddings attr failed."), return NOT_CHANGED);

  if (paddings.size() < DIM_NUM4 || paddings[0].size() < DIRECTION_COUNT || paddings[1].size() < DIRECTION_COUNT ||
      paddings[2].size() < DIRECTION_COUNT || paddings[3].size() < DIRECTION_COUNT) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The number of paddings not valid, can not fusion.");
    return NOT_CHANGED;
  }

  int64_t paddingsT;
  int64_t paddingsB;
  int64_t paddingsL;
  int64_t paddingsR;
  if (paddNode->GetOpDesc()->GetInputDesc(0).GetFormat() == ge::FORMAT_NCHW) {
    if (paddings[0][0] != 0 || paddings[0][1] != 0 || paddings[1][0] != 0 || paddings[1][1] != 0) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Padd and DepthwiseConv2d fusion can only on H and W.");
      return NOT_CHANGED;
    }
    paddingsT = paddings[2][0];
    paddingsB = paddings[2][1];
    paddingsL = paddings[3][0];
    paddingsR = paddings[3][1];
  } else if (paddNode->GetOpDesc()->GetInputDesc(0).GetFormat() == ge::FORMAT_NHWC) {
    if (paddings[0][0] != 0 || paddings[0][1] != 0 || paddings[3][0] != 0 || paddings[3][1] != 0) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Padd and DepthwiseConv2d fusion can only on H and W.");
      return NOT_CHANGED;
    }
    paddingsT = paddings[1][0];
    paddingsB = paddings[1][1];
    paddingsL = paddings[2][0];
    paddingsR = paddings[2][1];
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Padd intput Format is not NCHW or NHWC, can not fusion.");
    return NOT_CHANGED;
  }

  if (paddingsT < 0 || paddingsT > 255 || paddingsB < 0 || paddingsB > 255 || paddingsL < 0 || paddingsL > 255 ||
      paddingsR < 0 || paddingsR > 255) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Paddings value not in [0,255], can not fusion.");
    return NOT_CHANGED;
  }
  ge::NodePtr kernelNode = depthwiseConv2dNode->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  if (kernelNode->GetOpDesc()->GetOutputDesc(0).GetFormat() == ge::FORMAT_NCHW &&
      (kernelNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(2) <= paddingsT ||
       kernelNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(2) <= paddingsB)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Filter_H more than pad_H, can not fusion.");
    return NOT_CHANGED;
  }

  if (kernelNode->GetOpDesc()->GetOutputDesc(0).GetFormat() == ge::FORMAT_HWCN &&
      (kernelNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(0) <= paddingsT ||
       kernelNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(0) <= paddingsB)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Filter_H more than pad_H, can not fusion.");
    return NOT_CHANGED;
  }

  vector<int64_t> pads;
  pads.push_back(paddingsT);
  pads.push_back(paddingsB);
  pads.push_back(paddingsL);
  pads.push_back(paddingsR);
  if (!paddNode->GetOutControlAnchor()->GetPeerInControlAnchors().empty()) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "PaddNode has control edge, can not fusion.");
    return NOT_CHANGED;
  }

  for (auto inDataAnchor : paddNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(inDataAnchor->GetOwnerNode()->GetOpDesc()->GetType() != DEPTHWISECONV2D,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "Output node is not DepthwiseConv2D, can not fusion."),
                      return NOT_CHANGED);
  }
  vector<ge::NodePtr> nodeVector;
  // maybe fuse dw in the future.
  nodeVector.push_back(depthwiseConv2dNode);
  for (ge::NodePtr nodePtr : nodeVector) {
    string nodeName = nodePtr->GetOpDesc()->GetType();
    // update input desc
    FUSION_PASS_CHECK(nodePtr->GetOpDesc()->UpdateInputDesc(0, paddNode->GetOpDesc()->GetInputDesc(0)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Update %s input failed.", nodeName.c_str()), return FAILED);
    // change input edge of padd to depthwise_conv2d
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(nodePtr->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                 nodePtr->GetInDataAnchor(0)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove %s input0 edge error", nodeName.c_str()), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(paddNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                              nodePtr->GetInDataAnchor(0)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                              paddNode->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                              nodePtr->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(nodePtr->GetOpDesc(), PADS, pads),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Set paddings to %s failed.", nodeName.c_str()), return FAILED);
    FUSION_PASS_CHECK(!ge::AttrUtils::SetStr(nodePtr->GetOpDesc(), PADDING, "SAME"),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Set padding attr failed."), return FAILED);
  }
  // remove paddNode output
  for (auto inDataAnchor : paddNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(paddNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
  }
  if (paddNode->GetOutControlAnchor()) {
    for (auto inControlAnchor : paddNode->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(paddNode->GetOutControlAnchor(), inControlAnchor) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out control edge failed."), return FAILED);
    }
  }
  FUSION_PASS_CHECK(graph.RemoveNode(paddNode) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove PadD node failed."),
                    return FAILED);
  fusionNodes.push_back(depthwiseConv2dNode);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define PaddDepthwiseConv2dFusionPass fusion end");
  return SUCCESS;
}
REGISTER_PASS("PaddDepthwiseConv2dFusionPass", BUILT_IN_GRAPH_PASS, PaddDepthwiseConv2dFusionPass);
}  // namespace fe
