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
 * \file softmax_fusion_pass.cpp
 * \brief
 */
#include "softmax_fusion_pass.h"
#include <vector>
#include <string>

#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "common/util/platform_info.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "SoftmaxV2";
static const std::string PATTERN_FUSEDNODE = "Softmax";
static const vector<vector<int>> SHAPE = {{8732, 21}};
static const vector<vector<int>> SHAPE1 = {{224, 224, 160, 4}};

bool SoftmaxFusionPass::CheckISUsePattern(vector<int64_t>& dimInfo) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter in CheckISUsePattern.");
  if (dimInfo.size() == 3) {
    PlatformInfo platform_info;
    OptionalInfo optional_info;
    FUSION_PASS_CHECK(PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info,
                                                                                       optional_info) != fe::SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get platformInfo failed."), return false);
    uint32_t core_num = platform_info.soc_info.ai_core_cnt;
    for (int64_t i = 0; i < (int64_t)SHAPE.size(); i++) {
      if (SHAPE[i][0] == dimInfo[1] && SHAPE[i][1] == dimInfo[2]) {
        if (i == 1 && (dimInfo[0] < 8 || core_num <= 2 || core_num  > 10)) {
          return false;
        }
        return true;
      }
    }
    return false;
  }
  if (dimInfo.size() == 5) {
    for (int64_t i = 0; i < (int64_t)SHAPE1.size(); i++) {
      if (SHAPE1[i][0] == dimInfo[1] && SHAPE1[i][1] == dimInfo[2] &&
          SHAPE1[i][2] == dimInfo[3] && SHAPE1[i][3] == dimInfo[4]) {
        return true;
      }
    }
    return false;
  }
  return false;
}

vector<FusionPattern*> SoftmaxFusionPass::DefinePatterns() {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define SoftmaxFusionPass pattern begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("SoftmaxFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define SoftmaxFusionPass pattern end");
}

Status SoftmaxFusionPass::UpdateFormat(ge::NodePtr& inNodePtr) {
  ge::OpDescPtr inOpDescPtr = inNodePtr->GetOpDesc();
  ge::GeTensorDesc xInputDesc = inOpDescPtr->GetInputDesc(0);
  ge::GeTensorDesc yOutputDesc = inOpDescPtr->GetOutputDesc(0);
  ge::Format inputOriginFormat = xInputDesc.GetOriginFormat();
  vector<int64_t> inputOriginShap = xInputDesc.GetOriginShape().GetDims();
  uint32_t oriShapelens = inputOriginShap.size();
  OP_LOGD(FUSED_OP_TYPE.c_str(), "softmax updateFormat input format = %d.", inputOriginFormat);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "softmax updateFormat input size = %d.", oriShapelens);
  if(inputOriginFormat != ge::FORMAT_NCHW && inputOriginFormat != ge::FORMAT_NHWC) {
    if (oriShapelens <= 4) {
      xInputDesc.SetFormat(ge::FORMAT_NHWC);
      xInputDesc.SetOriginFormat(ge::FORMAT_NHWC);
      yOutputDesc.SetFormat(ge::FORMAT_NHWC);
      yOutputDesc.SetOriginFormat(ge::FORMAT_NHWC);
    }else if(oriShapelens == 5) {
      xInputDesc.SetFormat(ge::FORMAT_NDHWC);
      xInputDesc.SetOriginFormat(ge::FORMAT_NDHWC);
      yOutputDesc.SetFormat(ge::FORMAT_NDHWC);
      yOutputDesc.SetOriginFormat(ge::FORMAT_NDHWC);
    }
    auto ret = inOpDescPtr->UpdateInputDesc(0, xInputDesc);
    auto ret1 = inOpDescPtr->UpdateOutputDesc(0, yOutputDesc);

    if (ret != ge::GRAPH_SUCCESS || ret1 != ge::GRAPH_SUCCESS) {
      return FAILED;
    }
  }
  return SUCCESS;
}

Status SoftmaxFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter SoftmaxFusionPass.");
  ge::NodePtr softmaxNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);

  FUSION_PASS_CHECK(softmaxNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "softmax node is null."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(UpdateFormat(softmaxNode) != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "update format fail."),
                    return NOT_CHANGED);
  ge::OpDescPtr softmaxOpDesc = softmaxNode->GetOpDesc();
  FUSION_PASS_CHECK(softmaxOpDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "softmax is null."),
                    return PARAM_INVALID);

  ge::GeTensorDesc softmaxInputOpDesc = softmaxOpDesc->GetInputDesc(0);
  ge::GeTensorDesc softmaxOutputOpDesc = softmaxOpDesc->GetOutputDesc(0);

  ge::GeShape softmaxInputShape = softmaxInputOpDesc.GetShape();
  vector<int64_t> dimInfo = softmaxInputShape.GetDims();
  OP_LOGD(FUSED_OP_TYPE.c_str(), "softmax updateFormat input size = %d.", dimInfo.size());
  bool isUsePattern = false;
  if (dimInfo.size() == 3 || dimInfo.size() == 5) {
    isUsePattern = CheckISUsePattern(dimInfo);
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "SoftmaxFusionPass is not support.");
    return NOT_CHANGED;
  }

  vector<int64_t> axes;
  ge::AttrUtils::GetListInt(softmaxOpDesc, "axes", axes);
  FUSION_PASS_CHECK(axes.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "axes is null, please check!"),
                    return FAILED);
  if (axes[0] < 0) {
    axes[0] = axes[0] + dimInfo.size();
  }

  if (dimInfo.size() == 3 && axes[0] == 2 && isUsePattern) {
    int64_t inputH = dimInfo[0];
    int64_t inputW = dimInfo[1];
    int64_t inputC = dimInfo[2];

    vector<int64_t> inputDimInfo = {inputH, inputC, inputW};
    ge::GeShape assitShape(inputDimInfo);
    ge::GeShape assitShapeOrigin(inputDimInfo);
    softmaxInputOpDesc.SetShape(assitShape);
    softmaxInputOpDesc.SetOriginShape(assitShapeOrigin);
    softmaxOutputOpDesc.SetShape(assitShape);
    softmaxOutputOpDesc.SetOriginShape(assitShapeOrigin);
    Status ret = softmaxOpDesc->UpdateInputDesc(0, softmaxInputOpDesc);
    ret = softmaxOpDesc->UpdateOutputDesc(0, softmaxOutputOpDesc);
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "UpdateInputDesc failed."), return FAILED);

    vector<int64_t> axes = {1};
    FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(softmaxNode->GetOpDesc(), "axes", axes),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "Set axes attr failed."), return FAILED);

    ge::InDataAnchorPtr softmaxAnchorPtr0 = softmaxNode->GetInDataAnchor(0);
    ge::OutDataAnchorPtr preAnchorPtr0 = softmaxAnchorPtr0->GetPeerOutAnchor();
    ge::NodePtr preNode = preAnchorPtr0->GetOwnerNode();

    // creat a transposeD node
    std::shared_ptr<ge::OpDesc> transposeDDesc = nullptr;
    transposeDDesc = std::make_shared<ge::OpDesc>(softmaxNode->GetName() + "_transposeD_layer", "TransposeD");
    FUSION_PASS_CHECK(transposeDDesc == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "TransposeD failed."),
                      return PARAM_INVALID);

    // add input
    ge::GeTensorDesc input_desc = preNode->GetOpDesc()->GetOutputDesc(0);
    FUSION_PASS_CHECK(transposeDDesc->AddInputDesc(input_desc) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transposeDDesc input failed."),
                      return FAILED);

    // add output
    ge::GeTensorDesc output_desc = softmaxNode->GetOpDesc()->GetInputDesc(0);
    FUSION_PASS_CHECK(transposeDDesc->AddOutputDesc(output_desc) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transposeDDesc output failed."),
                      return FAILED);

    // add node
    ge::NodePtr transposeDNode = graph.AddNode(transposeDDesc);

    ge::AttrUtils::SetListInt(transposeDDesc, "perm", {0, 2, 1});

    // remove edge between quant and pooling
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(preAnchorPtr0, softmaxAnchorPtr0) != SUCCESS,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "remove edge between quant and pooling failed!"), return FAILED);

    // add edge between quant and antiquant
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(preAnchorPtr0, transposeDNode->GetInDataAnchor(0)) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s and node %s failed.",
                              preNode->GetName().c_str(), transposeDNode->GetName().c_str()),
                      return FAILED);

    // add edge between antiquant and pooling
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(transposeDNode->GetOutDataAnchor(0), softmaxAnchorPtr0) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s and node %s failed.",
                              softmaxNode->GetName().c_str(), transposeDNode->GetName().c_str()),
                      return FAILED);

    // creat a transposeD node
    ge::OutDataAnchorPtr softmaxAnchorPtr1 = softmaxNode->GetOutDataAnchor(0);
    ge::NodePtr postNode = nullptr;
    auto transposeDpPtr1 = softmaxAnchorPtr1->GetPeerInDataAnchors().at(0);

    std::shared_ptr<ge::OpDesc> transposeDDesc1 = nullptr;
    transposeDDesc1 = std::make_shared<ge::OpDesc>(softmaxNode->GetName() + "_transposeD1_layer", "TransposeD");
    FUSION_PASS_CHECK(transposeDDesc1 == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transposeD is null, TransposeD failed."),
                      return PARAM_INVALID);

    // add input
    ge::GeTensorDesc input_desc1 = softmaxNode->GetOpDesc()->GetOutputDesc(0);
    FUSION_PASS_CHECK(transposeDDesc1->AddInputDesc(input_desc1) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transposeDDesc1 input failed."),
                      return FAILED);

    // add output
    ge::GeTensorDesc output_desc1 = transposeDpPtr1->GetOwnerNode()->GetOpDesc()->GetInputDesc(0);
    FUSION_PASS_CHECK(transposeDDesc1->AddOutputDesc(output_desc1) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transposeDDesc1 output failed."),
                      return FAILED);

    ge::GeTensorDesc transposeD1OutputOpDesc = transposeDDesc1->GetOutputDesc(0);
    vector<int64_t> inputDimInfo1 = {inputH, inputW, inputC};
    ge::GeShape assitShape1(inputDimInfo1);
    ge::GeShape assitShapeOrigin1(inputDimInfo1);
    transposeD1OutputOpDesc.SetShape(assitShape1);
    transposeD1OutputOpDesc.SetOriginShape(assitShapeOrigin1);
    ret = transposeDDesc1->UpdateOutputDesc(0, transposeD1OutputOpDesc);
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "UpdateInputDesc failed."), return FAILED);

    // add node
    ge::NodePtr transposeDNode1 = graph.AddNode(transposeDDesc1);

    ge::AttrUtils::SetListInt(transposeDDesc1, "perm", {0, 2, 1});

    // add edge between softmax and transdateD
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(softmaxAnchorPtr1, transposeDNode1->GetInDataAnchor(0)) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s and node %s failed.",
                              softmaxNode->GetName().c_str(), transposeDNode1->GetName().c_str()),
                      return FAILED);

    for (auto postAnchorPtr0 : softmaxAnchorPtr1->GetPeerInDataAnchors()) {
      if (postAnchorPtr0->GetOwnerNode()->GetName() != softmaxNode->GetName() + "_transposeD1_layer") {
        postNode = postAnchorPtr0->GetOwnerNode();
        // remove edge between pooling and next node
        FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(postAnchorPtr0, softmaxAnchorPtr1) != SUCCESS,
                          OP_LOGI(FUSED_OP_TYPE.c_str(), "remove edge between softmax and next node failed!"),
                          return FAILED);

        // add edge between transdateD and post
        FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(transposeDNode1->GetOutDataAnchor(0), postAnchorPtr0) != SUCCESS,
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s and node %s failed.",
                                  transposeDDesc1->GetName().c_str(), softmaxNode->GetName().c_str()),
                          return FAILED);
      }
    }
    return SUCCESS;
  } else if (dimInfo.size() == 5 && axes[0] == 4 && isUsePattern) {
    vector<int64_t> inputDimInfo = {dimInfo[0], dimInfo[1], dimInfo[4], dimInfo[2], dimInfo[3]};
    ge::GeShape assitShape(inputDimInfo);
    ge::GeShape assitShapeOrigin(inputDimInfo);
    softmaxInputOpDesc.SetShape(assitShape);
    softmaxInputOpDesc.SetOriginShape(assitShapeOrigin);
    softmaxOutputOpDesc.SetShape(assitShape);
    softmaxOutputOpDesc.SetOriginShape(assitShapeOrigin);
    Status ret = softmaxOpDesc->UpdateInputDesc(0, softmaxInputOpDesc);
    ret = softmaxOpDesc->UpdateOutputDesc(0, softmaxOutputOpDesc);
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "UpdateInputDesc failed."),
                      return FAILED);

    vector<int64_t> axes = {2};
    FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(softmaxNode->GetOpDesc(), "axes", axes),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "Set axes attr failed."), return FAILED);

    ge::InDataAnchorPtr softmaxAnchorPtr0 = softmaxNode->GetInDataAnchor(0);
    ge::OutDataAnchorPtr preAnchorPtr0 = softmaxAnchorPtr0->GetPeerOutAnchor();
    ge::NodePtr preNode = preAnchorPtr0->GetOwnerNode();

    // creat a transposeD node
    std::shared_ptr<ge::OpDesc> transposeDDesc = nullptr;
    transposeDDesc = std::make_shared<ge::OpDesc>(softmaxNode->GetName() + "_transposeD_layer", "TransposeD");
    FUSION_PASS_CHECK(transposeDDesc == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transposeDDesc is null, TransposeD failed."),
                      return PARAM_INVALID);

    // add input
    ge::GeTensorDesc input_desc = preNode->GetOpDesc()->GetOutputDesc(0);
    FUSION_PASS_CHECK(transposeDDesc->AddInputDesc(input_desc) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transposeDDesc input failed."),
                      return FAILED);

    // add output
    ge::GeTensorDesc output_desc = softmaxNode->GetOpDesc()->GetInputDesc(0);
    FUSION_PASS_CHECK(transposeDDesc->AddOutputDesc(output_desc) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transposeDDesc output failed."),
                      return FAILED);

    // add node
    ge::NodePtr transposeDNode = graph.AddNode(transposeDDesc);

    ge::AttrUtils::SetListInt(transposeDDesc, "perm", {0, 1, 4, 2, 3});

    // remove edge between quant and pooling
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(preAnchorPtr0, softmaxAnchorPtr0) != SUCCESS,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "remove edge between quant and pooling failed!"),
                      return FAILED);

    // add edge between quant and antiquant
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(preAnchorPtr0, transposeDNode->GetInDataAnchor(0)) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s and node %s failed.",
                              preNode->GetName().c_str(), transposeDNode->GetName().c_str()),
                      return FAILED);

    // add edge between antiquant and pooling
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(transposeDNode->GetOutDataAnchor(0), softmaxAnchorPtr0) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s and node %s failed.",
                              softmaxNode->GetName().c_str(), transposeDNode->GetName().c_str()),
                      return FAILED);

    // creat a transposeD node
    ge::OutDataAnchorPtr softmaxAnchorPtr1 = softmaxNode->GetOutDataAnchor(0);
    ge::NodePtr postNode = nullptr;
    auto transposeDpPtr1 = softmaxAnchorPtr1->GetPeerInDataAnchors().at(0);

    std::shared_ptr<ge::OpDesc> transposeDDesc1 = nullptr;
    transposeDDesc1 = std::make_shared<ge::OpDesc>(softmaxNode->GetName() + "_transposeD1_layer", "TransposeD");
    FUSION_PASS_CHECK(transposeDDesc1 == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transposeD is null, TransposeD failed."),
                      return PARAM_INVALID);

    // add input
    ge::GeTensorDesc input_desc1 = softmaxNode->GetOpDesc()->GetOutputDesc(0);
    FUSION_PASS_CHECK(transposeDDesc1->AddInputDesc(input_desc1) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transposeDDesc1 input failed."),
                      return FAILED);

    // add output
    ge::GeTensorDesc output_desc1 = transposeDpPtr1->GetOwnerNode()->GetOpDesc()->GetInputDesc(0);
    FUSION_PASS_CHECK(transposeDDesc1->AddOutputDesc(output_desc1) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transposeDDesc1 output failed."),
                      return FAILED);

    ge::GeTensorDesc transposeD1OutputOpDesc = transposeDDesc1->GetOutputDesc(0);
    vector<int64_t> inputDimInfo1 = {dimInfo[0], dimInfo[1], dimInfo[2], dimInfo[3], dimInfo[4]};
    ge::GeShape assitShape1(inputDimInfo1);
    ge::GeShape assitShapeOrigin1(inputDimInfo1);
    transposeD1OutputOpDesc.SetShape(assitShape1);
    transposeD1OutputOpDesc.SetOriginShape(assitShapeOrigin1);
    ret = transposeDDesc1->UpdateOutputDesc(0, transposeD1OutputOpDesc);
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "UpdateInputDesc failed."),
                      return FAILED);

    // add node
    ge::NodePtr transposeDNode1 = graph.AddNode(transposeDDesc1);

    ge::AttrUtils::SetListInt(transposeDDesc1, "perm", {0, 1, 3, 4, 2});

    // add edge between softmax and transdateD
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(softmaxAnchorPtr1, transposeDNode1->GetInDataAnchor(0)) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s and node %s failed.",
                              softmaxNode->GetName().c_str(), transposeDNode1->GetName().c_str()),
                      return FAILED);

    for (auto postAnchorPtr0 : softmaxAnchorPtr1->GetPeerInDataAnchors()) {
      if (postAnchorPtr0->GetOwnerNode()->GetName() != softmaxNode->GetName() + "_transposeD1_layer") {
        postNode = postAnchorPtr0->GetOwnerNode();
        // remove edge between pooling and next node
        FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(postAnchorPtr0, softmaxAnchorPtr1) != SUCCESS,
                          OP_LOGI(FUSED_OP_TYPE.c_str(), "remove edge between softmax and next node failed!"),
                          return FAILED);

        // add edge between transdateD and post
        FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(transposeDNode1->GetOutDataAnchor(0), postAnchorPtr0) != SUCCESS,
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s and node %s failed.",
                                  transposeDDesc1->GetName().c_str(), softmaxNode->GetName().c_str()),
                          return FAILED);
      }
    }
    return SUCCESS;
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "SoftmaxFusionPass is not match.");
    return NOT_CHANGED;
  }
}

REGISTER_PASS("SoftmaxFusionPass", BUILT_IN_GRAPH_PASS, SoftmaxFusionPass);
}  // namespace fe
