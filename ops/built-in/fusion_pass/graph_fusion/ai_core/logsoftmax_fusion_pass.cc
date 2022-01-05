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
 * \file logsoftmax_fusion_pass.cpp
 * \brief
 */
#include "logsoftmax_fusion_pass.h"
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
static const char* FUSED_NODE = "LogSoftmaxV2";
static const std::string PATTERN_FUSEDNODE = "LogSoftmax";
static const vector<vector<int>> SHAPE = {{2000, 29}};

bool LogSoftmaxFusionPass::CheckISUsePattern(int64_t inputW, int64_t inputC) const {
  PlatformInfo platform_info;
  OptionalInfo optional_info;
  FUSION_PASS_CHECK(PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info,
                                                                                     optional_info) != fe::SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get platform_info failed."), return false);
  uint32_t core_num = platform_info.soc_info.ai_core_cnt;
  for (int64_t i = 0; i < (int64_t)SHAPE.size(); i++) {
    if (SHAPE[i][0] == inputW && SHAPE[i][1] == inputC && core_num == 2) {
      return true;
    }
  }
  return false;
}

std::vector<FusionPattern*> LogSoftmaxFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("LogSoftmaxFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status LogSoftmaxFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, std::vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter LogSoftmaxFusionPass.");
  ge::NodePtr logsoftmaxNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);

  FUSION_PASS_CHECK(logsoftmaxNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "logsoftmax node is null."),
                    return PARAM_INVALID);
  ge::OpDescPtr logsoftmaxOpDesc = logsoftmaxNode->GetOpDesc();
  FUSION_PASS_CHECK(logsoftmaxOpDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "logsoftmax is null."),
                    return PARAM_INVALID);

  ge::GeTensorDesc softmaxInputOpDesc = logsoftmaxOpDesc->GetInputDesc(0);
  ge::GeTensorDesc softmaxOutputOpDesc = logsoftmaxOpDesc->GetOutputDesc(0);

  ge::GeShape softmaxInputShape = softmaxInputOpDesc.GetShape();
  vector<int64_t> dimInfo = softmaxInputShape.GetDims();
  vector<int64_t> axes_val;
  int64_t inputC = 0;
  int64_t inputH = 0;
  int64_t inputW = 0;
  bool isUsePattern = false;
  ge::AttrUtils::GetListInt(logsoftmaxOpDesc, "axes", axes_val);
  FUSION_PASS_CHECK(axes_val.empty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "axes is null, please check!"),
                    return FAILED);
  if (axes_val[0] < 0) {
    axes_val[0] = axes_val[0] + dimInfo.size();
  }
  if (dimInfo.size() == 3) {
    inputH = dimInfo[0];
    inputW = dimInfo[1];
    inputC = dimInfo[2];
  } else {
    return NOT_CHANGED;
  }
  isUsePattern = CheckISUsePattern(inputW, inputC);
  const size_t axes_0_val_use_pattern = 2;
  if (axes_val[0] == axes_0_val_use_pattern && isUsePattern) {
    vector<int64_t> inputDimInfo = {inputH, inputC, inputW};
    ge::GeShape assitShape(inputDimInfo);
    ge::GeShape assitShapeOrigin(inputDimInfo);
    softmaxInputOpDesc.SetShape(assitShape);
    softmaxInputOpDesc.SetOriginShape(assitShapeOrigin);
    softmaxOutputOpDesc.SetShape(assitShape);
    softmaxOutputOpDesc.SetOriginShape(assitShapeOrigin);
    Status ret = logsoftmaxOpDesc->UpdateInputDesc(0, softmaxInputOpDesc);
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "UpdateInputDesc failed."), return FAILED);
    ret = logsoftmaxOpDesc->UpdateOutputDesc(0, softmaxOutputOpDesc);
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "UpdateOutputDesc failed."), return FAILED);

    vector<int64_t> axes_set = {1};
    FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(logsoftmaxNode->GetOpDesc(), "axes", axes_set),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "Set axes attr failed."), return FAILED);

    ge::InDataAnchorPtr softmaxAnchorPtr0 = logsoftmaxNode->GetInDataAnchor(0);
    FUSION_PASS_CHECK(softmaxAnchorPtr0 == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "softmaxAnchorPtr0 is null, get node input failed."),
                      return PARAM_INVALID);
    ge::OutDataAnchorPtr preAnchorPtr0 = softmaxAnchorPtr0->GetPeerOutAnchor();
    FUSION_PASS_CHECK(preAnchorPtr0 == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "preAnchorPtr0 is null, get prenode output failed."),
                      return PARAM_INVALID);
    ge::NodePtr preNode = preAnchorPtr0->GetOwnerNode();
    FUSION_PASS_CHECK(preNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "preNode is null."),
                      return PARAM_INVALID);

    // creat a transposeD node
    std::shared_ptr<ge::OpDesc> transposeDDesc = nullptr;
    transposeDDesc = std::make_shared<ge::OpDesc>(logsoftmaxNode->GetName() + "_transposeD_layer", "TransposeD");
    FUSION_PASS_CHECK(transposeDDesc == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transposeDDesc is null, TransposeD failed."),
                      return PARAM_INVALID);

    // add input
    ge::GeTensorDesc input_desc = preNode->GetOpDesc()->GetOutputDesc(0);
    FUSION_PASS_CHECK(transposeDDesc->AddInputDesc(input_desc) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transposeDDesc input failed."), return FAILED);

    // add output
    ge::GeTensorDesc output_desc = logsoftmaxNode->GetOpDesc()->GetInputDesc(0);
    FUSION_PASS_CHECK(transposeDDesc->AddOutputDesc(output_desc) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transposeDDesc output failed."), return FAILED);

    // add node
    ge::NodePtr transposeDNode = graph.AddNode(transposeDDesc);

    ge::AttrUtils::SetListInt(transposeDDesc, "perm", {0, 2, 1});

    // remove edge between quant and pooling
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(preAnchorPtr0, softmaxAnchorPtr0) != SUCCESS,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "remove edge between quant and pooling failed!"), return FAILED);

    // add edge between quant and antiquant
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(preAnchorPtr0, transposeDNode->GetInDataAnchor(0)) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                              preNode->GetName().c_str(), transposeDNode->GetName().c_str()),
                      return FAILED);

    // add edge between antiquant and pooling
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(transposeDNode->GetOutDataAnchor(0), softmaxAnchorPtr0) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                              logsoftmaxNode->GetName().c_str(), transposeDNode->GetName().c_str()),
                      return FAILED);

    // creat a transposeD node
    ge::OutDataAnchorPtr softmaxAnchorPtr1 = logsoftmaxNode->GetOutDataAnchor(0);
    ge::NodePtr postNode = nullptr;
    auto transposeDpPtr1 = softmaxAnchorPtr1->GetPeerInDataAnchors().at(0);

    std::shared_ptr<ge::OpDesc> transposeDDesc1 = nullptr;
    transposeDDesc1 = std::make_shared<ge::OpDesc>(logsoftmaxNode->GetName() + "_transposeD1_layer", "TransposeD");
    FUSION_PASS_CHECK(transposeDDesc1 == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transposeD is null, TransposeD failed."), return PARAM_INVALID);

    // add input
    ge::GeTensorDesc input_desc1 = logsoftmaxNode->GetOpDesc()->GetOutputDesc(0);
    FUSION_PASS_CHECK(transposeDDesc1->AddInputDesc(input_desc1) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transposeDDesc1 input failed."), return FAILED);

    // add output
    ge::GeTensorDesc output_desc1 = transposeDpPtr1->GetOwnerNode()->GetOpDesc()->GetInputDesc(0);
    FUSION_PASS_CHECK(transposeDDesc1->AddOutputDesc(output_desc1) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transposeDDesc1 output failed."), return FAILED);

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
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                              logsoftmaxNode->GetName().c_str(), transposeDNode1->GetName().c_str()),
                      return FAILED);

    for (auto postAnchorPtr0 : softmaxAnchorPtr1->GetPeerInDataAnchors()) {
      if (postAnchorPtr0->GetOwnerNode()->GetName() != logsoftmaxNode->GetName() + "_transposeD1_layer") {
        postNode = postAnchorPtr0->GetOwnerNode();
        // remove edge between pooling and next node
        FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(postAnchorPtr0, softmaxAnchorPtr1) != SUCCESS,
                          OP_LOGI(FUSED_OP_TYPE.c_str(), "remove edge between softmax and next node failed!"),
                          return FAILED);

        // add edge between transdateD and post
        FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(transposeDNode1->GetOutDataAnchor(0), postAnchorPtr0) != SUCCESS,
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                                  transposeDDesc1->GetName().c_str(), logsoftmaxNode->GetName().c_str()),
                          return FAILED);
      }
    }
    return SUCCESS;
  } else {
    return NOT_CHANGED;
  }
}

REGISTER_PASS("LogSoftmaxFusionPass", BUILT_IN_GRAPH_PASS, LogSoftmaxFusionPass);
}  // namespace fe
