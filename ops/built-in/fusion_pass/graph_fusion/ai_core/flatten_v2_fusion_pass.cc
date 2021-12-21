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
 * \file flatten_v2_fusion_pass.cpp
 * \brief diag flatten_v2 pass
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
#include "flatten_v2_fusion_pass.h"

using namespace std;
using namespace ge;

namespace fe {
static const string PATTERN_FLATTEN_V2 = "FlattenV2";
static const char* FLATTEN_V2 = "FlattenV2";
static const char* FLATTEN_V2_ATTR_AXIS = "axis";
static const char* FLATTEN_V2_ATTR_END_AXIS = "end_axis";

vector<FusionPattern*> FlattenV2Pass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  // define Fusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("FlattenV2Pass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  // define origin graph
  pattern->AddOpDesc(PATTERN_FLATTEN_V2, {FLATTEN_V2}).SetOutput(PATTERN_FLATTEN_V2);

  patterns.push_back(pattern);

  return patterns;
}

Status FlattenV2Pass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "enter into FlattenV2Pass");
  // diag node
  ge::NodePtr flattenV2Node = GetNodeFromMapping(PATTERN_FLATTEN_V2, mapping);
  FUSION_PASS_CHECK(flattenV2Node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "flattenV2Node is null, fusion failed."),
                    return PARAM_INVALID);

  ge::InDataAnchorPtr oriInAnchorPtr0 = flattenV2Node->GetInDataAnchor(0);
  ge::OutDataAnchorPtr oriBottomPeerAnchorPtr0 = oriInAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr inputNode = oriBottomPeerAnchorPtr0->GetOwnerNode();

  ge::OutDataAnchorPtr oriOutAnchorPtr0 = flattenV2Node->GetOutDataAnchor(0);
  auto oriTopPeerAnchors = oriOutAnchorPtr0->GetPeerInDataAnchors();

  ge::OpDescPtr flattenV2Desc = flattenV2Node->GetOpDesc();

  ge::GeTensorDesc flattenV2InputDesc = flattenV2Desc->GetInputDesc(0);

  vector<int64_t> inputShape = flattenV2InputDesc.GetShape().GetDims();
  int64_t dimCnt = inputShape.size();
  for (int64_t i = 0; i < dimCnt; i++) {
    if (PatternFusionUtil::IsUnknownShape(inputShape[i])) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "FlattenV2Pass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
  }
  int64_t flattenV2Axis = 1;
  int64_t flattenV2EndAxis = -1;
  ge::AttrUtils::GetInt(flattenV2Desc, FLATTEN_V2_ATTR_AXIS, flattenV2Axis);
  ge::AttrUtils::GetInt(flattenV2Desc, FLATTEN_V2_ATTR_END_AXIS, flattenV2EndAxis);
  if (flattenV2Axis < 0) {
    flattenV2Axis += dimCnt;
  }
  if (flattenV2EndAxis < 0) {
    flattenV2EndAxis += dimCnt;
  }

  vector<int64_t> outputShape;
  for (int64_t i = 0; i < flattenV2Axis; i++) {
    outputShape.push_back(inputShape[i]);
  }
  int64_t dimVal = 1;
  FUSION_PASS_CHECK(static_cast<int64_t>(inputShape.size()) < flattenV2EndAxis + 1,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "invalid inputShape."), return FAILED);
  for (int64_t i = flattenV2Axis; i < flattenV2EndAxis + 1; i++) {
    if (PatternFusionUtil::IsUnknownShape(inputShape[i])) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "FlattenV2Pass cannot be applied for unknown shape.");
      return FAILED;
    }
    dimVal = dimVal * inputShape[i];
  }
  outputShape.push_back(dimVal);

  for (int64_t i = flattenV2EndAxis + 1; i < dimCnt; i++) {
    outputShape.push_back(inputShape[i]);
  }

  for (auto inAnchor : flattenV2Node->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }

  if (flattenV2Node->GetInControlAnchor() != nullptr) {
    flattenV2Node->GetInControlAnchor()->UnlinkAll();
  }

  for (auto outAnchor : flattenV2Node->GetAllOutDataAnchors()) {
    if (outAnchor != nullptr) {
      outAnchor->UnlinkAll();
    }
  }
  if (flattenV2Node->GetOutControlAnchor() != nullptr) {
    flattenV2Node->GetOutControlAnchor()->UnlinkAll();
  }

  for (uint64_t i = 0; i < oriTopPeerAnchors.size(); i++) {
    ge::InDataAnchorPtr oriTopPeerAnchorPtri = oriTopPeerAnchors.at(i);
    ge::NodePtr outputNode = oriTopPeerAnchorPtri->GetOwnerNode();
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(oriBottomPeerAnchorPtr0, oriTopPeerAnchorPtri),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              inputNode->GetName().c_str(), outputNode->GetName().c_str()),
                      return FAILED);
  }

  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(flattenV2Node),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove flattenv2 node failed"), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "FlattenV2Pass success!!!!");

  return SUCCESS;
}
REGISTER_PASS("FlattenV2Pass", BUILT_IN_GRAPH_PASS, FlattenV2Pass);
}  // namespace fe
