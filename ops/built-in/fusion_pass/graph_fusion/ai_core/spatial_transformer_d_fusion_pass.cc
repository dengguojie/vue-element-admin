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
 * \file spatial_transformer_d_fusion_pass.cpp
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
#include "fp16_t.hpp"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "pattern_fusion_util.h"
#include "common/util/platform_info.h"

#include "spatial_transformer_d_fusion_pass.h"

using namespace std;
using namespace ge;

namespace fe {
static const string PATTERN_SPATIAL_TRANSFORMER_D = "SpatialTransformerD";
static const char* SPATIAL_TRANSFORMER_D = "SpatialTransformerD";

Status SpatialTransformerDPass::StnWIndexFP16(const int32_t h, const int32_t w, uint16_t* output1) {
  for (int32_t i = 0; i < h; ++i) {
    for (int32_t j = 0; j < w; ++j) {
      fp16_t t;
      t.val = 0;
      float normnize = j * 1.0 / w * 2.0 - 1.0;
      t = normnize;
      output1[i * w + j] = t.val;
    }
  }
  return SUCCESS;
}

Status SpatialTransformerDPass::StnHIndexFP16(const int32_t h, const int32_t w, uint16_t* output1) {
  for (int32_t i = 0; i < h; ++i) {
    for (int32_t j = 0; j < w; ++j) {
      fp16_t t;
      t.val = 0;
      float normnize = i * 1.0 / h * 2.0 - 1.0;
      t = normnize;
      output1[i * w + j] = t.val;
    }
  }
  return SUCCESS;
}

Status SpatialTransformerDPass::StnPreAddConst(ge::NodePtr& thisNode, ge::OpDescPtr& thisOpDesc) {
  OP_LOGI(TBE_FUSED_OP_TYPE.c_str(), "Enter SpatialTransformerD StnPre Add Const layer");
  vector<int64_t> size;
  ge::AttrUtils::GetListInt(thisOpDesc, "size", size);
  FUSION_PASS_CHECK(size.empty(), OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "StnPre size shape is NULL."), return FAILED);
  FUSION_PASS_CHECK(size.size() < 4, OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "StnPre size dims must 4."), return FAILED);

  ge::GeTensorPtr assitPtrW = nullptr;
  ge::GeTensorPtr assitPtrH = nullptr;

  vector<int64_t> assitDimInfo;
  assitDimInfo.push_back(size[2]);
  assitDimInfo.push_back(size[3]);

  unique_ptr<uint16_t[]> inputAssitW(new (std::nothrow) uint16_t[assitDimInfo[0] * assitDimInfo[1]]());
  FUSION_PASS_CHECK(inputAssitW.get() == nullptr, OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "inputAssitW is NULL"),
                    return PARAM_INVALID);
  unique_ptr<uint16_t[]> inputAssitH(new (std::nothrow) uint16_t[assitDimInfo[0] * assitDimInfo[1]]());
  FUSION_PASS_CHECK(inputAssitH.get() == nullptr, OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "inputAssitH is NULL"),
                    return PARAM_INVALID);

  Status ret = StnWIndexFP16(assitDimInfo[0], assitDimInfo[1], inputAssitW.get());
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "GenerateWIndex failed."), return ret);
  ret = StnHIndexFP16(assitDimInfo[0], assitDimInfo[1], inputAssitH.get());
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "GenerateHIndex failed."), return ret);

  ge::GeShape assitShape(assitDimInfo);
  ge::GeTensorDesc tensorDescW(GeShape(), ge::FORMAT_ND, ge::DT_FLOAT16);
  tensorDescW.SetShape(assitShape);
  tensorDescW.SetOriginShape(assitShape);
  tensorDescW.SetOriginFormat(ge::FORMAT_ND);
  ge::GeTensorDesc tensorDescH(GeShape(), ge::FORMAT_ND, ge::DT_FLOAT16);
  tensorDescH.SetShape(assitShape);
  tensorDescH.SetOriginShape(assitShape);
  tensorDescH.SetOriginFormat(ge::FORMAT_ND);

  FUSION_PASS_MAKE_SHARED(
      (assitPtrW = std::make_shared<ge::GeTensor>(tensorDescW, reinterpret_cast<uint8_t*>(inputAssitW.get()),
                                                  assitDimInfo[0] * assitDimInfo[1] * sizeof(uint16_t))),
      assitPtrW = nullptr;
      return PARAM_INVALID);
  FUSION_PASS_MAKE_SHARED(
      (assitPtrH = std::make_shared<ge::GeTensor>(tensorDescH, reinterpret_cast<uint8_t*>(inputAssitH.get()),
                                                  assitDimInfo[0] * assitDimInfo[1] * sizeof(uint16_t))),
      assitPtrH = nullptr;
      return PARAM_INVALID);

  vector<ge::GeTensorPtr> weights = {assitPtrW, assitPtrH};
  ge::OpDescUtils::SetWeights(thisNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(thisNode);
  ge::NodePtr constInput0 = constInputNodes[0];
  constInput0->GetOpDesc()->SetType("Const");
  ge::NodePtr constInput1 = constInputNodes[1];
  constInput1->GetOpDesc()->SetType("Const");

  OP_LOGI(TBE_FUSED_OP_TYPE.c_str(), "SpatialTransformerD StnPre Add Const layer success");
  return SUCCESS;
}

Status SpatialTransformerDPass::MakeStnPreLayer(ge::OpDescPtr& thisOpDesc, const ge::OpDescPtr& formerOpDesc,
                                                bool hasInput1) {
  OP_LOGI(TBE_FUSED_OP_TYPE.c_str(), "Enter SpatialTransformerD make StnPre layer");

  ge::GeTensorDesc formerInputDesc0 = formerOpDesc->GetInputDesc(0);

  vector<int64_t> shapeDims = formerInputDesc0.GetOriginShape().GetDims();
  FUSION_PASS_CHECK(shapeDims.empty(), OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), 
                                               "SpatialTransformerD input shape is NULL."), return FAILED);
  FUSION_PASS_CHECK(shapeDims.size() < 4, OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), 
                                                  "SpatialTransformerD input 0 format must be NC0HWC1"), return FAILED);

  vector<int64_t> output_size;
  vector<float> default_theta;
  vector<bool> use_default_theta;
  bool align_corners = false;

  if (!ge::AttrUtils::GetListInt(formerOpDesc, "output_size", output_size)) {
    output_size.push_back(shapeDims[2]);
    output_size.push_back(shapeDims[3]);
  } else {
    output_size[0] = (output_size[0] == -1) ? shapeDims[2] : output_size[0];
    output_size[1] = (output_size[1] == -1) ? shapeDims[3] : output_size[1];
  }
  vector<int64_t> size;
  size.push_back(shapeDims[2]);
  size.push_back(shapeDims[3]);
  size.push_back(output_size[0]);
  size.push_back(output_size[1]);
  ge::AttrUtils::SetListInt(thisOpDesc, "size", size);

  if (ge::AttrUtils::GetListFloat(formerOpDesc, "default_theta", default_theta)) {
    ge::AttrUtils::SetListFloat(thisOpDesc, "default_theta", default_theta);
  }

  if (!ge::AttrUtils::GetListBool(formerOpDesc, "use_default_theta", use_default_theta)) {
    use_default_theta.push_back(false);
    use_default_theta.push_back(false);
    use_default_theta.push_back(false);
    use_default_theta.push_back(false);
    use_default_theta.push_back(false);
    use_default_theta.push_back(false);
  }
  ge::AttrUtils::SetListBool(thisOpDesc, "use_default_theta", use_default_theta);

  if (!ge::AttrUtils::GetBool(formerOpDesc, "align_corners", align_corners)) {
    align_corners = false;
  }
  ge::AttrUtils::SetBool(thisOpDesc, "align_corners", align_corners);

  if (hasInput1) {
    ge::GeTensorDesc formerInputDesc1 = formerOpDesc->GetInputDesc(1);
    ge::GeTensorDesc opInputDesc;
    opInputDesc.SetShape(formerInputDesc1.GetShape());
    opInputDesc.SetFormat(formerInputDesc1.GetFormat());
    opInputDesc.SetDataType(formerInputDesc1.GetDataType());
    opInputDesc.SetOriginShape(formerInputDesc1.GetOriginShape());
    opInputDesc.SetOriginFormat(formerInputDesc1.GetOriginFormat());
    opInputDesc.SetOriginDataType(formerInputDesc1.GetOriginDataType());
    thisOpDesc->AddInputDesc("theta", opInputDesc);
  }

  vector<int64_t> outputShapeDims;
  outputShapeDims.push_back(shapeDims[0]);
  outputShapeDims.push_back((shapeDims[1] + 15) / 16);
  outputShapeDims.push_back(size[2] * size[3] * 4);

  ge::GeTensorDesc outputDesc0;
  outputDesc0.SetShape(ge::GeShape(outputShapeDims));
  outputDesc0.SetFormat(ge::FORMAT_ND);
  outputDesc0.SetDataType(formerInputDesc0.GetDataType());
  outputDesc0.SetOriginShape(ge::GeShape(outputShapeDims));
  outputDesc0.SetOriginFormat(ge::FORMAT_ND);
  outputDesc0.SetOriginDataType(formerInputDesc0.GetOriginDataType());
  thisOpDesc->AddOutputDesc("pos_coef", outputDesc0);

  ge::GeTensorDesc outputDesc1;
  outputDesc1.SetShape(ge::GeShape(outputShapeDims));
  outputDesc1.SetFormat(ge::FORMAT_ND);
  outputDesc1.SetDataType(ge::DT_INT32);
  outputDesc1.SetOriginShape(ge::GeShape(outputShapeDims));
  outputDesc1.SetOriginFormat(ge::FORMAT_ND);
  outputDesc1.SetOriginDataType(ge::DT_INT32);
  thisOpDesc->AddOutputDesc("pos_offset", outputDesc1);

  OP_LOGI(TBE_FUSED_OP_TYPE.c_str(), "SpatialTransformerD make StnPre layer success");
  return SUCCESS;
}

Status SpatialTransformerDPass::MakeStnComputeLayer(ge::OpDescPtr& thisOpDesc, const ge::OpDescPtr& bottomOpDesc,
                                                    const ge::OpDescPtr& formerOpDesc) {
  OP_LOGI(TBE_FUSED_OP_TYPE.c_str(), "Enter SpatialTransformerD make StnCompute layer");

  vector<int64_t> size;
  bool align_corners = false;

  ge::AttrUtils::GetListInt(bottomOpDesc, "size", size);
  FUSION_PASS_CHECK(size.empty(), OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "StnPre size shape is NULL."), return FAILED);
  ge::AttrUtils::SetListInt(thisOpDesc, "size", size);

  if (!ge::AttrUtils::GetBool(bottomOpDesc, "align_corners", align_corners)) {
    align_corners = false;
  }
  ge::AttrUtils::SetBool(thisOpDesc, "align_corners", align_corners);

  ge::GeTensorDesc formerInOpDesc0 = formerOpDesc->GetInputDesc(0);
  ge::GeTensorDesc opInputDesc0;
  opInputDesc0.SetShape(formerInOpDesc0.GetShape());
  opInputDesc0.SetFormat(formerInOpDesc0.GetFormat());
  opInputDesc0.SetDataType(formerInOpDesc0.GetDataType());
  opInputDesc0.SetOriginShape(formerInOpDesc0.GetOriginShape());
  opInputDesc0.SetOriginFormat(formerInOpDesc0.GetOriginFormat());
  opInputDesc0.SetOriginDataType(formerInOpDesc0.GetOriginDataType());
  thisOpDesc->AddInputDesc("x", opInputDesc0);

  ge::GeTensorDesc bottomOutputDesc0 = bottomOpDesc->GetOutputDesc(0);
  ge::GeTensorDesc opInputDesc1;
  opInputDesc1.SetShape(bottomOutputDesc0.GetShape());
  opInputDesc1.SetFormat(bottomOutputDesc0.GetFormat());
  opInputDesc1.SetDataType(bottomOutputDesc0.GetDataType());
  opInputDesc1.SetOriginShape(bottomOutputDesc0.GetOriginShape());
  opInputDesc1.SetOriginFormat(bottomOutputDesc0.GetOriginFormat());
  opInputDesc1.SetOriginDataType(bottomOutputDesc0.GetOriginDataType());
  thisOpDesc->AddInputDesc("pos_coef", opInputDesc1);

  ge::GeTensorDesc bottomOutputDesc1 = bottomOpDesc->GetOutputDesc(1);
  ge::GeTensorDesc opInputDesc2;
  opInputDesc2.SetShape(bottomOutputDesc1.GetShape());
  opInputDesc2.SetFormat(bottomOutputDesc1.GetFormat());
  opInputDesc2.SetDataType(bottomOutputDesc1.GetDataType());
  opInputDesc2.SetOriginShape(bottomOutputDesc1.GetOriginShape());
  opInputDesc2.SetOriginFormat(bottomOutputDesc1.GetOriginFormat());
  opInputDesc2.SetOriginDataType(bottomOutputDesc1.GetOriginDataType());
  thisOpDesc->AddInputDesc("pos_offset", opInputDesc2);

  ge::GeTensorDesc formerOutOpDesc0 = formerOpDesc->GetOutputDesc(0);
  ge::GeTensorDesc outputDesc;
  outputDesc.SetShape(formerOutOpDesc0.GetShape());
  outputDesc.SetFormat(formerOutOpDesc0.GetFormat());
  outputDesc.SetDataType(formerOutOpDesc0.GetDataType());
  outputDesc.SetOriginShape(formerOutOpDesc0.GetOriginShape());
  outputDesc.SetOriginFormat(formerOutOpDesc0.GetOriginFormat());
  outputDesc.SetOriginDataType(formerOutOpDesc0.GetOriginDataType());
  thisOpDesc->AddOutputDesc("y", outputDesc);

  OP_LOGI(TBE_FUSED_OP_TYPE.c_str(), "SpatialTransformerD make StnCompute layer success");
  return SUCCESS;
}

vector<FusionPattern*> SpatialTransformerDPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  // define Fusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("SpatialTransformerDPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  // define origin graph
  pattern->AddOpDesc(PATTERN_SPATIAL_TRANSFORMER_D, {SPATIAL_TRANSFORMER_D}).SetOutput(PATTERN_SPATIAL_TRANSFORMER_D);

  patterns.push_back(pattern);

  return patterns;
}

Status SpatialTransformerDPass::TbeFusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGD("Enter SpatialTransformerDPass::TbeFusion.");

  // diag node
  ge::NodePtr spatialTransformerDNode = GetNodeFromMapping(PATTERN_SPATIAL_TRANSFORMER_D, mapping);
  FUSION_PASS_CHECK(spatialTransformerDNode == nullptr,
                    OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "spatialTransformerDsNode is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OpDescPtr spatialTransformerDOpDesc = spatialTransformerDNode->GetOpDesc();
  FUSION_PASS_CHECK(spatialTransformerDOpDesc == nullptr,
                  OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "spatialTransformerDOpDesc is null, fusion failed."),
                  return PARAM_INVALID);

  // Get Input Node
  ge::InDataAnchorPtr oriInAnchorPtr0 = spatialTransformerDNode->GetInDataAnchor(0);
  ge::OutDataAnchorPtr oriBottomPeerAnchorPtr0 = oriInAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr inputNode0 = oriBottomPeerAnchorPtr0->GetOwnerNode();

  ge::InDataAnchorPtr oriInAnchorPtr1 = spatialTransformerDNode->GetInDataAnchor(1);
  ge::OutDataAnchorPtr oriBottomPeerAnchorPtr1 = nullptr;
  ge::NodePtr inputNode1 = nullptr;
  if (oriInAnchorPtr1 != nullptr) {
    oriBottomPeerAnchorPtr1 = oriInAnchorPtr1->GetPeerOutAnchor();
    inputNode1 = oriBottomPeerAnchorPtr1->GetOwnerNode();
  }

  // Get Output Node
  ge::OutDataAnchorPtr oriOutAnchorPtr0 = spatialTransformerDNode->GetOutDataAnchor(0);
  auto oriTopPeerAnchors = oriOutAnchorPtr0->GetPeerInDataAnchors();

  for (auto inAnchor : spatialTransformerDNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }

  if (spatialTransformerDNode->GetInControlAnchor() != nullptr) {
    spatialTransformerDNode->GetInControlAnchor()->UnlinkAll();
  }

  for (auto outAnchor : spatialTransformerDNode->GetAllOutDataAnchors()) {
    if (outAnchor != nullptr) {
      outAnchor->UnlinkAll();
    }
  }
  if (spatialTransformerDNode->GetOutControlAnchor() != nullptr) {
    spatialTransformerDNode->GetOutControlAnchor()->UnlinkAll();
  }

  ge::OpDescPtr stnPreOp;
  ge::NodePtr stnPreNode;
  FUSION_PASS_MAKE_SHARED((stnPreOp = 
    std::make_shared<ge::OpDesc>(spatialTransformerDNode->GetName() + "_stn_pre", "StnPre")), return INTERNAL_ERROR);

  FUSION_PASS_CHECK(SUCCESS != MakeStnPreLayer(stnPreOp, spatialTransformerDOpDesc, oriInAnchorPtr1 != nullptr),
                    OP_LOGW(TBE_FUSED_OP_TYPE.c_str(), "make stn_pre layer failed."), return NOT_CHANGED);

  stnPreNode = graph.AddNode(stnPreOp);

  FUSION_PASS_CHECK(SUCCESS != StnPreAddConst(stnPreNode, stnPreOp),
                    OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "make stn_compute layer failed."), return FAILED);

  if (oriInAnchorPtr1 != nullptr) {
    OP_LOGI(TBE_FUSED_OP_TYPE.c_str(), "Start to  Add StnPre Theta Edge");
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(oriBottomPeerAnchorPtr1, stnPreNode->GetInDataAnchor(0)),
                      OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              inputNode1->GetName().c_str(), stnPreNode->GetName().c_str()),
                      return FAILED);
    OP_LOGI(TBE_FUSED_OP_TYPE.c_str(), "StnPre Add theta Edge success");
  }

  ge::OpDescPtr stnComputeOp;
  ge::NodePtr stnComputeNode;
  FUSION_PASS_MAKE_SHARED((stnComputeOp = 
    std::make_shared<ge::OpDesc>(spatialTransformerDNode->GetName() + "_stn_compute", "StnCompute")),
    return INTERNAL_ERROR);

  FUSION_PASS_CHECK(SUCCESS != MakeStnComputeLayer(stnComputeOp, stnPreOp, spatialTransformerDOpDesc),
                    OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "make stn_compute layer failed."), return FAILED);

  stnComputeNode = graph.AddNode(stnComputeOp);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(oriBottomPeerAnchorPtr0, stnComputeNode->GetInDataAnchor(0)),
                    OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "add edge from src node[%s]'s out to dst node[%s]'s in1 failed.",
                            inputNode0->GetName().c_str(), stnComputeNode->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(stnPreNode->GetOutDataAnchor(0), stnComputeNode->GetInDataAnchor(1)),
      OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "add edge from src node[%s]'s out0 to dst node[%s]'s in1 failed.",
              stnPreNode->GetName().c_str(), stnComputeNode->GetName().c_str()),
      return FAILED);

  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(stnPreNode->GetOutDataAnchor(1), stnComputeNode->GetInDataAnchor(2)),
      OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "add edge from src node[%s]'s out1 to dst node[%s]'s in2 failed.",
              stnPreNode->GetName().c_str(), stnComputeNode->GetName().c_str()),
      return FAILED);

  for (uint64_t i = 0; i < oriTopPeerAnchors.size(); i++) {
    ge::InDataAnchorPtr oriTopPeerAnchorPtri = oriTopPeerAnchors.at(i);
    FUSION_PASS_CHECK(oriTopPeerAnchorPtri == nullptr,
                  OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "oriTopPeerAnchorPtri is null, fusion failed."),
                  return PARAM_INVALID);
    ge::NodePtr outputNode = oriTopPeerAnchorPtri->GetOwnerNode();
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(stnComputeNode->GetOutDataAnchor(0), oriTopPeerAnchorPtri),
                      OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              stnComputeNode->GetName().c_str(), outputNode->GetName().c_str()),
                      return FAILED);
  }

  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(spatialTransformerDNode),
                    OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "remove SpatialTransformerD node failed"), return FAILED);

  OP_LOGD("Exit SpatialTransformerDPass::TbeFusion success.");
  return SUCCESS;
}

Status SpatialTransformerDPass::AicpuFusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGD("Enter SpatialTransformerPass AicpuFusion.");

  ge::NodePtr spatialTransformerDNode = GetNodeFromMapping(PATTERN_SPATIAL_TRANSFORMER_D, mapping);
  FUSION_PASS_CHECK(spatialTransformerDNode == nullptr,
                    OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "spatialTransformerDsNode is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OpDescPtr spatialTransformerDOpDesc = spatialTransformerDNode->GetOpDesc();
  FUSION_PASS_CHECK(spatialTransformerDOpDesc == nullptr,
                  OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "spatialTransformerDOpDesc is null, fusion failed."),
                  return PARAM_INVALID);

  // get input and output info
  ge::InDataAnchorPtr oriInAnchorPtr0 = spatialTransformerDNode->GetInDataAnchor(0);
  ge::OutDataAnchorPtr oriBottomPeerAnchorPtr0 = nullptr;
  ge::NodePtr inputNode0 = nullptr;
  if (oriInAnchorPtr0 != nullptr) {
    oriBottomPeerAnchorPtr0 = oriInAnchorPtr0->GetPeerOutAnchor();
    inputNode0 = oriBottomPeerAnchorPtr0->GetOwnerNode();
  } else {
    OP_LOGD(AICPU_FUSED_OP_TYPE.c_str(), "oriInAnchorPtr0 is null, fusion failed.");
    return FAILED;
  }

  ge::InDataAnchorPtr oriInAnchorPtr1 = spatialTransformerDNode->GetInDataAnchor(1);
  ge::OutDataAnchorPtr oriBottomPeerAnchorPtr1 = nullptr;
  ge::NodePtr inputNode1 = nullptr;
  if (oriInAnchorPtr1 != nullptr) {
    oriBottomPeerAnchorPtr1 = oriInAnchorPtr1->GetPeerOutAnchor();
    inputNode1 = oriBottomPeerAnchorPtr1->GetOwnerNode();
  } else {
    OP_LOGD(AICPU_FUSED_OP_TYPE.c_str(), "oriInAnchorPtr1 is null, fusion failed.");
    return FAILED;
  }

  ge::OutDataAnchorPtr oriOutAnchorPtr0 = spatialTransformerDNode->GetOutDataAnchor(0);
  auto oriTopPeerAnchors = oriOutAnchorPtr0->GetPeerInDataAnchors();

  // unlink all edge
  for (auto inAnchor : spatialTransformerDNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }

  if (spatialTransformerDNode->GetInControlAnchor() != nullptr) {
    spatialTransformerDNode->GetInControlAnchor()->UnlinkAll();
  }

  for (auto outAnchor : spatialTransformerDNode->GetAllOutDataAnchors()) {
    if (outAnchor != nullptr) {
      outAnchor->UnlinkAll();
    }
  }
  if (spatialTransformerDNode->GetOutControlAnchor() != nullptr) {
    spatialTransformerDNode->GetOutControlAnchor()->UnlinkAll();
  }

  // create new aicpu op
  ge::OpDescPtr stnDesc;
  ge::NodePtr stnNode;
  FUSION_PASS_MAKE_SHARED((stnDesc = std::make_shared<ge::OpDesc>(spatialTransformerDNode->GetName() + "_aicpu",
                          "SpatialTransformer")),
                          return INTERNAL_ERROR);

  vector<int32_t> output_size;
  vector<float> default_theta;
  vector<bool> use_default_theta_tmp;
  vector<int32_t> use_default_theta;
  bool align_corners = false;
  int32_t stn_ori_channel = 0;

  ge::AttrUtils::GetListInt(spatialTransformerDOpDesc, "output_size", output_size);
  ge::AttrUtils::SetListInt(stnDesc, "output_size", output_size);

  ge::AttrUtils::GetListFloat(spatialTransformerDOpDesc, "default_theta", default_theta);
  ge::AttrUtils::SetListFloat(stnDesc, "default_theta", default_theta);

  ge::AttrUtils::GetListBool(spatialTransformerDOpDesc, "use_default_theta", use_default_theta_tmp);
  for (uint32_t i = 0; i < use_default_theta_tmp.size(); i++) {
    if (use_default_theta_tmp[i]) {
      use_default_theta.push_back(1);
    } else {
      use_default_theta.push_back(0);
    }
  }
  ge::AttrUtils::SetListInt(stnDesc, "use_default_theta", use_default_theta);

  ge::AttrUtils::GetBool(spatialTransformerDOpDesc, "align_corners", align_corners);
  ge::AttrUtils::SetBool(stnDesc, "align_corners", align_corners);

  ge::AttrUtils::GetInt(spatialTransformerDOpDesc, "stn_ori_channel", stn_ori_channel);
  ge::AttrUtils::SetInt(stnDesc, "stn_ori_channel", stn_ori_channel);

  ge::GeTensorDesc formerInputDesc0 = spatialTransformerDOpDesc->GetInputDesc(0);
  ge::GeTensorDesc opInputDesc0;
  opInputDesc0.SetShape(formerInputDesc0.GetShape());
  opInputDesc0.SetFormat(formerInputDesc0.GetFormat());
  opInputDesc0.SetDataType(ge::DT_FLOAT16);
  opInputDesc0.SetOriginShape(formerInputDesc0.GetOriginShape());
  opInputDesc0.SetOriginFormat(formerInputDesc0.GetOriginFormat());
  opInputDesc0.SetOriginDataType(formerInputDesc0.GetOriginDataType());
  stnDesc->AddInputDesc("x", opInputDesc0);

  ge::GeTensorDesc formerInputDesc1 = spatialTransformerDOpDesc->GetInputDesc(1);
  ge::GeTensorDesc opInputDesc1;
  opInputDesc1.SetShape(formerInputDesc1.GetShape());
  opInputDesc1.SetFormat(formerInputDesc1.GetFormat());
  opInputDesc1.SetDataType(ge::DT_FLOAT16);
  opInputDesc1.SetOriginShape(formerInputDesc1.GetOriginShape());
  opInputDesc1.SetOriginFormat(formerInputDesc1.GetOriginFormat());
  opInputDesc1.SetOriginDataType(formerInputDesc1.GetOriginDataType());
  stnDesc->AddInputDesc("theta", opInputDesc1);

  ge::GeTensorDesc formerOuputDesc0 = spatialTransformerDOpDesc->GetOutputDesc(0);
  ge::GeTensorDesc opOutputDesc0;
  opOutputDesc0.SetShape(formerOuputDesc0.GetShape());
  opOutputDesc0.SetFormat(formerOuputDesc0.GetFormat());
  opOutputDesc0.SetDataType(ge::DT_FLOAT16);
  opOutputDesc0.SetOriginShape(formerOuputDesc0.GetOriginShape());
  opOutputDesc0.SetOriginFormat(formerOuputDesc0.GetOriginFormat());
  opOutputDesc0.SetOriginDataType(formerOuputDesc0.GetOriginDataType());
  stnDesc->AddOutputDesc("y", opOutputDesc0);

  stnNode = graph.AddNode(stnDesc);

  // add edge
  OP_LOGI(AICPU_FUSED_OP_TYPE.c_str(), "Start to add edge between stn and x.");
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(oriBottomPeerAnchorPtr0, stnNode->GetInDataAnchor(0)),
                    OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                    inputNode0->GetName().c_str(), stnNode->GetName().c_str()),
                    return FAILED);

  OP_LOGI(AICPU_FUSED_OP_TYPE.c_str(), "Start to add edge between stn and theta.");
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(oriBottomPeerAnchorPtr1, stnNode->GetInDataAnchor(1)),
                    OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                    inputNode1->GetName().c_str(), stnNode->GetName().c_str()),
                    return FAILED);

  OP_LOGI(AICPU_FUSED_OP_TYPE.c_str(), "Start to add edge between stn and y.");
  for (uint32_t i = 0; i < oriTopPeerAnchors.size(); i++) {
    ge::InDataAnchorPtr oriTopPeerAnchorPtri = oriTopPeerAnchors.at(i);
    if (oriTopPeerAnchorPtri == nullptr) {
      OP_LOGD(TBE_FUSED_OP_TYPE.c_str(), "oriTopPeerAnchorPtri is null.");
      continue;
    }
    ge::NodePtr outputNode = oriTopPeerAnchorPtri->GetOwnerNode();
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(stnNode->GetOutDataAnchor(0), oriTopPeerAnchorPtri),
                      OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                      stnNode->GetName().c_str(), outputNode->GetName().c_str()),
                      return FAILED);
  }

  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(spatialTransformerDNode),
    OP_LOGE(TBE_FUSED_OP_TYPE.c_str(), "remove SpatialTransformerD node failed"), return FAILED);

  OP_LOGD("Exit SpatialTransformerPass AicpuFusion success.");
  return SUCCESS;
}

Status SpatialTransformerDPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(TBE_FUSED_OP_TYPE.c_str(), "enter into SpatialTransformerDPass");

  // tbe op have performance issue on lhisi, use aicpu op
  PlatformInfo platform_info;
  OptionalInfo opti_compilation_info;
  if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(
    platform_info, opti_compilation_info) != SUCCESS) {
    OP_LOGW("Fail to get platform info.");
  }
  OP_LOGD("SpatialTransformerPass", "Get opti_compilation_info.soc_version[%s].", 
          opti_compilation_info.soc_version.c_str());

  if (opti_compilation_info.soc_version == "SD3403" || opti_compilation_info.soc_version == "Hi3796CV300CS") {
    Status ret = AicpuFusion(graph, mapping, newNodes);
    if (ret != SUCCESS) {
      return FAILED;
    }
  } else {
    Status ret = TbeFusion(graph, mapping, newNodes);
    if (ret != SUCCESS) {
      return FAILED;
    }
  }

  OP_LOGI("Exit SpatialTransformerPass success.");

  return SUCCESS;
}

REGISTER_PASS("SpatialTransformerDPass", BUILT_IN_GRAPH_PASS, SpatialTransformerDPass);
}  // namespace fe
