/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file spp_fusion_pass.cpp
 * \brief
 */
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
#include "error_util.h"
#include "fp16_t.hpp"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "pattern_fusion_util.h"

#include "spp_fusion_pass.h"

using namespace std;
using namespace ge;

namespace fe {
static const int64_t INT_NUM_TWO = 2;
static const int64_t INT_NUM_THREE = 3;
static const int64_t INT_NUM_FOUR = 4;
static const int64_t INT_LESS_C0 = 15;
static const string PATTERN_SPP = "SPP";
static const char* SPP = "SPP";

Status SPPPass::MakePoolingLayer(ge::OpDescPtr& poolingOpDesc, const ge::GeTensorDesc& inputDesc, int64_t hyramidLevel,
                                 int64_t poolMethod) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter SPP make pooling layer");
  vector<int64_t> shapeDims = inputDesc.GetOriginShape().GetDims();
  if (shapeDims.empty() || shapeDims.size() < INT_NUM_FOUR) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "SPP input shapedims is less than 4");
    return PARAM_INVALID;
  }
  for (size_t i = 1; i <= INT_NUM_THREE; i++) {
    auto dim = shapeDims[i];
    if (PatternFusionUtil::IsUnknownShape(dim)) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "SPPPass cannot be applied for unknown shape.");
      return FAILED;
    }
  }
  int64_t num_bins = pow(INT_NUM_TWO, hyramidLevel);
  bool globalPooling = (hyramidLevel == 0) ? true : false;
  int64_t bottomH = shapeDims[INT_NUM_TWO];
  int64_t bottomW = shapeDims[INT_NUM_THREE];
  int64_t windowH = bottomH;
  int64_t windowW = bottomW;
  int64_t strideH = 1;
  int64_t strideW = 1;
  int64_t padT = 0;
  int64_t padB = 0;
  int64_t padL = 0;
  int64_t padR = 0;
  int64_t poolMode = 0;
  int64_t ceilMode = 0;

  if (!globalPooling) {
    windowH = ceil(bottomH / static_cast<double>(num_bins));
    int64_t remainderH = windowH * num_bins - bottomH;
    padT = (remainderH + 1) / INT_NUM_TWO;
    padB = (remainderH + 1) / INT_NUM_TWO;
    strideH = windowH;

    windowW = ceil(bottomW / static_cast<double>(num_bins));
    int64_t remainderW = windowW * num_bins - bottomW;
    padL = (remainderW + 1) / INT_NUM_TWO;
    padR = (remainderW + 1) / INT_NUM_TWO;
    strideW = windowW;
  }

  ge::AttrUtils::SetBool(poolingOpDesc, "global_pooling", globalPooling);
  vector<int64_t> windowVec;
  windowVec.push_back(windowH);
  windowVec.push_back(windowW);
  ge::AttrUtils::SetListInt(poolingOpDesc, "window", windowVec);
  vector<int64_t> strideVec;
  strideVec.push_back(strideH);
  strideVec.push_back(strideW);
  ge::AttrUtils::SetListInt(poolingOpDesc, "stride", strideVec);
  vector<int64_t> padVec;
  padVec.push_back(padT);
  padVec.push_back(padB);
  padVec.push_back(padL);
  padVec.push_back(padR);
  ge::AttrUtils::SetListInt(poolingOpDesc, "pad", padVec);
  poolMode = poolMethod;
  ge::AttrUtils::SetInt(poolingOpDesc, "mode", poolMode);
  ceilMode = 0;
  ge::AttrUtils::SetInt(poolingOpDesc, "ceil_mode", ceilMode);

  ge::GeTensorDesc poolingInputDesc;
  vector<int64_t> inputShape;
  int64_t c0 = 16;
  int64_t c1 = (shapeDims[1] + INT_LESS_C0) / c0;
  inputShape.push_back(shapeDims[0]);
  inputShape.push_back(c1);
  inputShape.push_back(shapeDims[INT_NUM_TWO]);
  inputShape.push_back(shapeDims[INT_NUM_THREE]);
  inputShape.push_back(c0);
  poolingInputDesc.SetShape(ge::GeShape(inputShape));
  poolingInputDesc.SetFormat(ge::FORMAT_NC1HWC0);
  poolingInputDesc.SetDataType(inputDesc.GetDataType());
  poolingInputDesc.SetOriginShape(inputDesc.GetOriginShape());
  poolingInputDesc.SetOriginFormat(ge::FORMAT_NCHW);
  poolingInputDesc.SetOriginDataType(inputDesc.GetOriginDataType());
  poolingOpDesc->AddInputDesc("x", poolingInputDesc);

  ge::GeTensorDesc poolingOutputDesc;
  vector<int64_t> outputShape;
  outputShape.push_back(shapeDims[0]);
  outputShape.push_back(c1);
  outputShape.push_back(num_bins);
  outputShape.push_back(num_bins);
  outputShape.push_back(c0);
  poolingOutputDesc.SetShape(ge::GeShape(outputShape));
  poolingOutputDesc.SetFormat(ge::FORMAT_NC1HWC0);
  poolingOutputDesc.SetDataType(inputDesc.GetDataType());
  vector<int64_t> oriOutputShape;
  oriOutputShape.push_back(shapeDims[0]);
  oriOutputShape.push_back(shapeDims[1]);
  oriOutputShape.push_back(num_bins);
  oriOutputShape.push_back(num_bins);
  poolingOutputDesc.SetOriginShape(ge::GeShape(oriOutputShape));
  poolingOutputDesc.SetOriginFormat(ge::FORMAT_NCHW);
  poolingOutputDesc.SetOriginDataType(inputDesc.GetOriginDataType());
  poolingOpDesc->AddOutputDesc("y", poolingOutputDesc);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "SPP make pooling layer success");
  return SUCCESS;
}

Status SPPPass::MakeConcatLayer(ge::OpDescPtr& concatOpDesc, vector<ge::OpDescPtr> fatherOp, int64_t concatDims) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter SPP make concat layer");
  uint64_t bottomSize = fatherOp.size();
  int64_t batchNum = 0;
  int64_t totalDimNum = 0;
  concatOpDesc->AddDynamicInputDesc("x", bottomSize);
  for (uint64_t i = 0; i < bottomSize; i++) {
    ge::GeTensorDesc bottomOutputDesc = fatherOp[i]->GetOutputDesc(0);
    vector<int64_t> shapeDims = bottomOutputDesc.GetOriginShape().GetDims();
    FUSION_PASS_CHECK(shapeDims.size() < INT_NUM_FOUR,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "SPP output shape dims is less than 4."),
                      return PARAM_INVALID;);
    for (size_t i = 1; i <= INT_NUM_THREE; i++) {
      auto dim = shapeDims[i];
      if (PatternFusionUtil::IsUnknownShape(dim)) {
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "SPPPass cannot be applied for unknown shape.");
        return FAILED;
      }
    }
    batchNum = shapeDims[0];
    totalDimNum += shapeDims[1] * shapeDims[INT_NUM_TWO] * shapeDims[INT_NUM_THREE];
    ge::GeTensorDesc inputDesc;
    vector<int64_t> inputShapeDims;
    inputShapeDims.push_back(shapeDims[0]);
    inputShapeDims.push_back(shapeDims[1] * shapeDims[INT_NUM_TWO] * shapeDims[INT_NUM_THREE]);
    inputShapeDims.push_back(1);
    inputShapeDims.push_back(1);
    inputDesc.SetShape(ge::GeShape(inputShapeDims));
    inputDesc.SetFormat(ge::FORMAT_NCHW);
    inputDesc.SetDataType(bottomOutputDesc.GetDataType());
    inputDesc.SetOriginShape(ge::GeShape(inputShapeDims));
    inputDesc.SetOriginFormat(ge::FORMAT_NCHW);
    inputDesc.SetOriginDataType(bottomOutputDesc.GetOriginDataType());
    concatOpDesc->UpdateInputDesc(i, inputDesc);
  }

  ge::AttrUtils::SetInt(concatOpDesc, "concat_dim", concatDims);
  ge::AttrUtils::SetInt(concatOpDesc, "N", bottomSize);

  ge::GeTensorDesc outputDesc;
  vector<int64_t> outputShapeDims;
  outputShapeDims.push_back(batchNum);
  outputShapeDims.push_back(totalDimNum);
  outputShapeDims.push_back(1);
  outputShapeDims.push_back(1);
  outputDesc.SetShape(ge::GeShape(outputShapeDims));
  outputDesc.SetFormat(ge::FORMAT_NCHW);
  outputDesc.SetDataType(concatOpDesc->GetInputDesc(0).GetDataType());
  outputDesc.SetOriginShape(ge::GeShape(outputShapeDims));
  outputDesc.SetOriginFormat(ge::FORMAT_NCHW);
  outputDesc.SetOriginDataType(concatOpDesc->GetInputDesc(0).GetOriginDataType());
  concatOpDesc->AddOutputDesc("y", outputDesc);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "SPP make concat layer success");
  return SUCCESS;
}

vector<FusionPattern*> SPPPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  // define Fusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("SPPPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  // define origin graph
  pattern->AddOpDesc(PATTERN_SPP, {SPP}).SetOutput(PATTERN_SPP);

  patterns.push_back(pattern);

  return patterns;
}

Status SPPPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "enter into SPPPass");
  // diag node
  ge::NodePtr sppNode = GetNodeFromMapping(PATTERN_SPP, mapping);
  FUSION_PASS_CHECK(sppNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "sppNode is null, fusion failed."),
                    return PARAM_INVALID);

  // Get Input Node
  ge::InDataAnchorPtr oriInAnchorPtr0 = sppNode->GetInDataAnchor(0);
  ge::OutDataAnchorPtr oriBottomPeerAnchorPtr0 = oriInAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr inputNode = oriBottomPeerAnchorPtr0->GetOwnerNode();
  // Get Output Node
  ge::OutDataAnchorPtr oriOutAnchorPtr0 = sppNode->GetOutDataAnchor(0);
  auto oriTopPeerAnchors = oriOutAnchorPtr0->GetPeerInDataAnchors();

  ge::OpDescPtr sppOpDesc = sppNode->GetOpDesc();
  ge::GeTensorDesc sppInputDesc = sppOpDesc->GetInputDesc(0);

  for (auto inAnchor : sppNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }

  if (sppNode->GetInControlAnchor() != nullptr) {
    sppNode->GetInControlAnchor()->UnlinkAll();
  }

  for (auto outAnchor : sppNode->GetAllOutDataAnchors()) {
    if (outAnchor != nullptr) {
      outAnchor->UnlinkAll();
    }
  }
  if (sppNode->GetOutControlAnchor() != nullptr) {
    sppNode->GetOutControlAnchor()->UnlinkAll();
  }

  int64_t sppHyramidHeight = 1;
  int64_t sppPoolMethod = 0;
  if (!ge::AttrUtils::GetInt(sppOpDesc, "pyramid_height", sppHyramidHeight)) {
    sppHyramidHeight = 1;
  }
  if (!ge::AttrUtils::GetInt(sppOpDesc, "pool_method", sppPoolMethod)) {
    sppPoolMethod = 0;
  }

  if (sppHyramidHeight == 1) {
    ge::OpDescPtr singlePoolingOp;
    ge::NodePtr singlePoolingNode;
    FUSION_PASS_MAKE_SHARED((singlePoolingOp = std::make_shared<ge::OpDesc>(sppNode->GetName() +
                                                                            "_spp_pooling", "SppPooling")),
                            return INTERNAL_ERROR);

    FUSION_PASS_CHECK(SUCCESS != MakePoolingLayer(singlePoolingOp, sppInputDesc, 0, sppPoolMethod),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "make pooling layer failed."),
                      return FAILED);

    singlePoolingNode = graph.AddNode(singlePoolingOp);

    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(oriBottomPeerAnchorPtr0, singlePoolingNode->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                inputNode->GetName().c_str(), singlePoolingNode->GetName().c_str()),
        return FAILED);

    for (uint64_t i = 0; i < oriTopPeerAnchors.size(); i++) {
      ge::InDataAnchorPtr oriTopPeerAnchorPtri = oriTopPeerAnchors.at(i);
      FUSION_PASS_CHECK(oriTopPeerAnchorPtri == nullptr,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                          "SPP output anchor ptr is null, fusion failed."),
                        return FAILED;);
      ge::NodePtr outputNode = oriTopPeerAnchorPtri->GetOwnerNode();
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(singlePoolingNode->GetOutDataAnchor(0), oriTopPeerAnchorPtri),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                  singlePoolingNode->GetName().c_str(), outputNode->GetName().c_str()),
          return FAILED);
    }

    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(sppNode),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove spp node failed"), return FAILED);

    return SUCCESS;
  }

  vector<ge::OpDescPtr> poolingOp;
  vector<ge::NodePtr> poolingNode;
  for (int64_t i = 0; i < sppHyramidHeight; i++) {
    string strName = sppNode->GetName() + "_spp_pooling" + std::to_string(i);
    ge::OpDescPtr tmpOp;
    ge::NodePtr tmpNode;
    FUSION_PASS_MAKE_SHARED((tmpOp = std::make_shared<ge::OpDesc>(strName, "SppPooling")), return INTERNAL_ERROR);
    poolingOp.push_back(tmpOp);

    FUSION_PASS_CHECK(SUCCESS != MakePoolingLayer(poolingOp[i], sppInputDesc, i, sppPoolMethod),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "make pooling layer failed."),
                      return FAILED);
    tmpNode = graph.AddNode(poolingOp[i]);
    FUSION_PASS_CHECK(tmpNode == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                        "poolingNode[%ld] is null, fusion failed.", i),
                      return PARAM_INVALID);
    poolingNode.push_back(tmpNode);

    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(oriBottomPeerAnchorPtr0, poolingNode[i]->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                        "add edge from src node[%s] to dst node[%s] failed.",
                              inputNode->GetName().c_str(), poolingNode[i]->GetName().c_str()),
                      return FAILED);
  }

  ge::OpDescPtr concatOp;
  ge::NodePtr concatNode;
  FUSION_PASS_MAKE_SHARED((concatOp = std::make_shared<ge::OpDesc>(sppNode->GetName() + "_concat", "ConcatD")),
                          return INTERNAL_ERROR);
  int64_t concatDim = 1;
  FUSION_PASS_CHECK(SUCCESS != MakeConcatLayer(concatOp, poolingOp, concatDim),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "make pooling layer failed."), return FAILED);

  concatNode = graph.AddNode(concatOp);

  for (uint64_t i = 0; i < poolingNode.size(); i++) {
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(poolingNode[i]->GetOutDataAnchor(0), concatNode->GetInDataAnchor(i)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                poolingNode[i]->GetName().c_str(), concatNode->GetName().c_str()),
        return FAILED);
  }

  for (uint64_t i = 0; i < oriTopPeerAnchors.size(); i++) {
    ge::InDataAnchorPtr oriTopPeerAnchorPtri = oriTopPeerAnchors.at(i);
    FUSION_PASS_CHECK(oriTopPeerAnchorPtri == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                        "SPP output anchor ptr is null, fusion failed."),
                      return FAILED;);
    ge::NodePtr outputNode = oriTopPeerAnchorPtri->GetOwnerNode();
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(concatNode->GetOutDataAnchor(0), oriTopPeerAnchorPtri),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                        "add edge from src node[%s] to dst node[%s] failed.",
                              concatNode->GetName().c_str(), outputNode->GetName().c_str()),
                      return FAILED);
  }

  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(sppNode),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove spp node failed"), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "SPPPass success!!!!");

  return SUCCESS;
}

REGISTER_PASS("SPPPass", BUILT_IN_GRAPH_PASS, SPPPass);
}  // namespace fe
