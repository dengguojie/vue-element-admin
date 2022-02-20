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
 * \file prod_force_se_a_fusion_pass.cc
 * \brief
 */
#include "prod_force_se_a_fusion_pass.h"
#include <vector>
#include <string>

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
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "pattern_fusion_util.h"
#include "common/debug/log.h"
#include "fp16_t.hpp"

namespace fe {
static const char* FUSED_NODE = "ProdForceSeA";
static const std::string PATTERN_FUSEDNODE = "ProdForceSeA";
static const int64_t DYNAMIC_SHAPE = -1;
static const int64_t DIM_IDX_0 = 0;
static const int64_t DIM_IDX_1 = 1;
static const int64_t DIM_IDX_2 = 2;
static const int64_t DIM_SIZE = 3;

bool ProdForceSeAFusionPass::CheckUsePattern(vector<int64_t>& dimInfo) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter in CheckUsePattern.");
  return true;
}

vector<FusionPattern*> ProdForceSeAFusionPass::DefinePatterns() {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define ProdForceSeAFusionPass pattern begin.");
  vector<FusionPattern*> patterns;
  // define DiagFusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ProdForceSeAFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  // define origin graph
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define ProdForceSeAFusionPass pattern end.");
  return patterns;
}

Status ProdForceSeAFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter into ProdForceSeAFusionPass");
  // prod_force_se_a node
  ge::NodePtr prodForceSeANode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(prodForceSeANode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "prodForceSeANode is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr prodForceSeADesc = prodForceSeANode->GetOpDesc();
  FUSION_PASS_CHECK(
      prodForceSeADesc == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "prodForceSeANode's OpDesc is null, fusion failed."),
      return PARAM_INVALID);
  ge::GeTensorDesc prodForceSeAOutputOpDesc = prodForceSeADesc->GetOutputDesc(0);
  ge::GeShape prodForceSeAOutputShape = prodForceSeAOutputOpDesc.GetShape();
  vector<int64_t> dimInfo = prodForceSeAOutputShape.GetDims();
  if (dimInfo.size() != DIM_SIZE) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "ProdForceSeA output dim is %ld.", dimInfo.size());
    return PARAM_INVALID;
  }
  std::vector<std::pair<int64_t, int64_t>> forceOutRange;
  FUSION_PASS_CHECK(prodForceSeAOutputOpDesc.GetShapeRange(forceOutRange) != GRAPH_SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "ProdForceSeA get shape range failed."), return NOT_CHANGED);
  if (forceOutRange.size() != DIM_SIZE) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "ProdForceSeA output range dim is %ld.", forceOutRange.size());
    return PARAM_INVALID;
  }
  int64_t outputNFrames = dimInfo[DIM_IDX_0];
  int64_t outputNall = dimInfo[DIM_IDX_1];
  int64_t outputLocal = dimInfo[DIM_IDX_2];
  std::pair<int64_t, int64_t> outputNFramesRange = forceOutRange[DIM_IDX_0];
  std::pair<int64_t, int64_t> outputNallRange = forceOutRange[DIM_IDX_1];
  std::pair<int64_t, int64_t> outputLocalRange = forceOutRange[DIM_IDX_2];
  bool secondInfer = (outputNall == DYNAMIC_SHAPE);
  ge::AttrUtils::SetBool(prodForceSeADesc, "second_infer", secondInfer);

  // create transpose node
  ge::OutDataAnchorPtr prodForceSeAOutAnchorPtr = prodForceSeANode->GetOutDataAnchor(0);
  ge::NodePtr postNode = nullptr;
  ge::OpDescPtr transposeDDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (transposeDDesc = std::make_shared<ge::OpDesc>(prodForceSeANode->GetName() + "_transposD_layer", "TransposeD")),
      return FAILED);

  // add input
  ge::GeTensorDesc transposeInputOpDesc = prodForceSeANode->GetOpDesc()->GetOutputDesc(0);
  vector<int64_t> inputDimInfo = {outputNFrames, outputLocal, outputNall};
  std::vector<std::pair<int64_t, int64_t>> newOutRange = {outputNFramesRange, outputLocalRange, outputNallRange};
  ge::GeShape assistShape(inputDimInfo);
  transposeInputOpDesc.SetShape(assistShape);
  transposeInputOpDesc.SetShapeRange(newOutRange);
  transposeInputOpDesc.SetOriginShape(assistShape);
  transposeInputOpDesc.SetOriginShapeRange(newOutRange);
  FUSION_PASS_CHECK(transposeDDesc->AddInputDesc(transposeInputOpDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transpose input fail."), return FAILED);

  // add output
  ge::GeTensorDesc transposeOutputOpDesc = prodForceSeANode->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(transposeDDesc->AddOutputDesc(transposeOutputOpDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "add transpose output fail."), return FAILED);

  // add node
  ge::NodePtr transposeDNode = graph.AddNode(transposeDDesc);
  ge::AttrUtils::SetListInt(transposeDDesc, "perm", {0, 2, 1});

  // update output of ProdForceSeA
  prodForceSeADesc->UpdateOutputDesc(0, transposeInputOpDesc);

  // add edges
  for (auto postAnchorPtr0 : prodForceSeAOutAnchorPtr->GetPeerInDataAnchors()) {
    postNode = postAnchorPtr0->GetOwnerNode();
    // remove edge between pooling and next node
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(prodForceSeAOutAnchorPtr, postAnchorPtr0) != SUCCESS,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "remove edge between prodForceSeA and next node failed!"),
                      return FAILED);

    // add edge between transdateD and post
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(transposeDNode->GetOutDataAnchor(0), postAnchorPtr0) != SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                                       transposeDDesc->GetName().c_str(), prodForceSeANode->GetName().c_str()),
        return FAILED);
  }
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(prodForceSeAOutAnchorPtr, transposeDNode->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transpose input0 edge fail."),
                    return FAILED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "ProdForceSeAFusionPass pass handle success.");
  return SUCCESS;
}
REGISTER_PASS("ProdForceSeAFusionPass", BUILT_IN_GRAPH_PASS, ProdForceSeAFusionPass);
}  // namespace fe
