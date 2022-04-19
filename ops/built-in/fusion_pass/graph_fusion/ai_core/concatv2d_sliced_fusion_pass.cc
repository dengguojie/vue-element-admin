/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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
 * \file concatv2d_sliced_fusion_pass.cc
 * \brief concatv2d sliced fusion pass
 */
#include "concatv2d_sliced_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "external/graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe {
static const std::string PATTERN_CONCATV2D = "ConcatV2D";
static const std::string PATTERN_SLICED_0 = "SliceD0";
static const std::string OP_TYPE_CONCATV2D = "ConcatV2D";
static const std::string OP_TYPE_SLICED = "SliceD";
static const char* ATTR_SLICED_OFFSETS = "offsets";

/*!
 * @brief Define pattern.
 * The graph struct need to adapt and target is shown as follows:
 *
 *        preNodes                preNodes
 *            |                    |    |
 *        ConcatV2D                |    |
 *         /     \        ==>      |    |
 *     SliceD   SliceD             |    |
 *         \     /                 |    |
 *        postNodes               postNodes
*
 * @return vector<FusionPattern*> All valid patterns.
 */
vector<FusionPattern*> Concatv2dSlicedFusionPass::DefinePatterns() {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into define Concatv2dSlicedFusionPass pattern");

  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("Concatv2dSlicedFusion");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to new a pattern object"),
                    return patterns);

  pattern->AddOpDesc(PATTERN_CONCATV2D, {OP_TYPE_CONCATV2D})
      .AddOpDesc(PATTERN_SLICED_0, {OP_TYPE_SLICED})
      .SetInputs(PATTERN_SLICED_0, {PATTERN_CONCATV2D})
      .SetOutput(PATTERN_SLICED_0);
  patterns.push_back(pattern);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define Concatv2dSlicedFusionPass pattern");
  return patterns;
}

bool Concatv2dSlicedFusionPass::IsVectorZeros(const vector<int64_t>& vec) const {
  for (size_t i = 0; i < vec.size(); i++) {
    if (vec[i] != 0) {
      return false;
    }
  }
  return true;
}

bool Concatv2dSlicedFusionPass::IsVectorEquals(const vector<int64_t>& v0, const vector<int64_t>& v1) const {
  if (v0.size() != v1.size()) {
    return false;
  }
  for (size_t i = 0; i < v0.size(); i++) {
    if (v0[i] != v1[i]) {
      return false;
    }
  }
  return true;
}

Status Concatv2dSlicedFusionPass::GetSortedFusedNodes(const ge::ComputeGraph& graph, const Mapping& mapping,
                                                      vector<ge::NodePtr>& fusedNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "GetSortedFusedNodes begin");

  ge::NodePtr concatV2DNode = GetNodeFromMapping(PATTERN_CONCATV2D, mapping);
  FUSION_PASS_CHECK(concatV2DNode == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to get ConcatV2D node"),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(concatV2DNode->GetOutDataNodesSize() != OUTPUTSIZE_LIMIT_TWO,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Number of ConcatV2D output should be 2"), return NOT_CHANGED);
  ge::OutDataAnchorPtr concatOutAnchor0 = concatV2DNode->GetOutDataAnchor(0);

  NodePtr sliceNode0 = concatOutAnchor0->GetPeerInDataAnchors().at(0)->GetOwnerNode();
  FUSION_PASS_CHECK(sliceNode0 == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to get SliceD(0) node"),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(sliceNode0->GetType() != OP_TYPE_SLICED,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Output data node of ConcatV2D should be SliceD"),
                    return NOT_CHANGED);

  NodePtr sliceNode1 = concatOutAnchor0->GetPeerInDataAnchors().at(1)->GetOwnerNode();
  FUSION_PASS_CHECK(sliceNode1 == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to get SliceD(1) node"),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(sliceNode1->GetType() != OP_TYPE_SLICED,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Output data node of ConcatV2D should be SliceD"),
                    return NOT_CHANGED);

  fusedNodes.push_back(concatV2DNode);

  // Identify which is SliceD(0) based on offsets
  ge::Operator sliceOp0 = ge::OpDescUtils::CreateOperatorFromNode(sliceNode0);
  std::vector<int64_t> offsets0;
  FUSION_PASS_CHECK(GRAPH_SUCCESS != sliceOp0.GetAttr(ATTR_SLICED_OFFSETS, offsets0),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Failed to get SliceD(0) attr offsets"), return NOT_CHANGED);
  if (IsVectorZeros(offsets0)) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "SliceD ( %s ) attr offsets are all 0", sliceNode0->GetName().c_str());
    fusedNodes.push_back(sliceNode0);
    fusedNodes.push_back(sliceNode1);
  } else {
    std::vector<int64_t> offsets1;
    ge::Operator sliceOp1 = ge::OpDescUtils::CreateOperatorFromNode(sliceNode1);
    FUSION_PASS_CHECK(GRAPH_SUCCESS != sliceOp1.GetAttr(ATTR_SLICED_OFFSETS, offsets1),
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "Failed to get SliceD(1) attr offsets"), return NOT_CHANGED);
    FUSION_PASS_CHECK(!IsVectorZeros(offsets1), OP_LOGD(FUSED_OP_TYPE.c_str(), "One of SliceD offsets should be all 0"),
                      return NOT_CHANGED);

    OP_LOGD(FUSED_OP_TYPE.c_str(), "SliceD ( %s ) attr offsets are all 0", sliceNode1->GetName().c_str());
    fusedNodes.push_back(sliceNode1);
    fusedNodes.push_back(sliceNode0);
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "GetSortedFusedNodes end");
  return SUCCESS;
}

Status Concatv2dSlicedFusionPass::CheckSliceInfo(const ge::NodePtr& concatV2DNode, const ge::NodePtr& sliceNode0,
                                                 const ge::NodePtr& sliceNode1) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckSliceInfo begin");

  ge::OpDescPtr concatV2DOpDesc = concatV2DNode->GetOpDesc();
  FUSION_PASS_CHECK(concatV2DOpDesc == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to get ConcatV2D op desc"),
                    return NOT_CHANGED);
  vector<int64_t> inputDims0 = concatV2DOpDesc->GetInputDesc(0).GetShape().GetDims();
  vector<int64_t> inputDims1 = concatV2DOpDesc->GetInputDesc(1).GetShape().GetDims();
  vector<int64_t> concatOutputDims = concatV2DOpDesc->GetOutputDesc(0).GetShape().GetDims();

  ge::OpDescPtr sliceOpDesc0 = sliceNode0->GetOpDesc();
  FUSION_PASS_CHECK(sliceOpDesc0 == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to get SliceD(0) op desc"),
                    return NOT_CHANGED);
  vector<int64_t> sliceOutputDims0 = sliceOpDesc0->GetOutputDesc(0).GetShape().GetDims();

  ge::Operator sliceOp1 = ge::OpDescUtils::CreateOperatorFromNode(sliceNode1);
  std::vector<int64_t> offsets1;
  FUSION_PASS_CHECK(GRAPH_SUCCESS != sliceOp1.GetAttr(ATTR_SLICED_OFFSETS, offsets1),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Failed to get SliceD(1) attr offsets"), return NOT_CHANGED);
  ge::OpDescPtr sliceOpDesc1 = sliceNode1->GetOpDesc();
  FUSION_PASS_CHECK(sliceOpDesc1 == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to get SliceD(1) op desc"),
                    return NOT_CHANGED);
  vector<int64_t> sliceOutputDims1 = sliceOpDesc1->GetOutputDesc(0).GetShape().GetDims();

  FUSION_PASS_CHECK(!IsVectorEquals(inputDims0, sliceOutputDims0),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "ConcatV2D input(0) shape and Slice(0) output shape should be same"),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(!IsVectorEquals(inputDims1, sliceOutputDims1),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "ConcatV2D input(1) shape and Slice(1) output shape should be same"),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(
      offsets1.size() != sliceOutputDims1.size(),
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Slice(1) attr offsets and Slice(1) output shape should be same length"),
      return NOT_CHANGED);
  std::vector<int64_t> tmpDims;
  for (size_t i = 0; i < offsets1.size(); i++) {
    tmpDims.push_back(offsets1[i] + sliceOutputDims1[i]);
  }
  FUSION_PASS_CHECK(!IsVectorEquals(tmpDims, concatOutputDims),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Slice(1) should be corresponds to ConcatV2D input(1)"),
                    return NOT_CHANGED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckSliceInfo end");
  return SUCCESS;
}

Status Concatv2dSlicedFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into Concatv2dSlicedFusionPass");

  vector<ge::NodePtr> fusedNodes;
  FUSION_PASS_CHECK(GetSortedFusedNodes(graph, mapping, fusedNodes) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Does not fit the fusion scene"), return NOT_CHANGED);
  ge::NodePtr concatV2DNode = fusedNodes[0];
  ge::NodePtr sliceNode0 = fusedNodes[1];
  ge::NodePtr sliceNode1 = fusedNodes[2];

  FUSION_PASS_CHECK(CheckSliceInfo(concatV2DNode, sliceNode0, sliceNode1) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Failed to check slice info"), return NOT_CHANGED);

  // concatV2DNode InDataAnchor(0) corresponds to  sliceNode0 OutDataAnchor
  for (auto inAnchor : sliceNode0->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    inAnchor->UnlinkAll();
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(concatV2DNode->GetInDataAnchor(0)->GetPeerOutAnchor(), inAnchor) != ge::GRAPH_SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from input(0) to output(0)"),
        return FAILED);
  }
  // concatV2DNode InDataAnchor(1) corresponds to  sliceNode1 OutDataAnchor
  for (auto inAnchor : sliceNode1->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    inAnchor->UnlinkAll();
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(concatV2DNode->GetInDataAnchor(1)->GetPeerOutAnchor(), inAnchor) != ge::GRAPH_SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from input(1) to output(1)"),
        return FAILED);
  }

  FUSION_PASS_CHECK(graph.RemoveNode(concatV2DNode) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove ConcatV2D node"),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(sliceNode0) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove SliceD(0) node"),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(sliceNode1) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove SliceD(1) node"),
                    return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to Concatv2dSlicedFusionPass");
  return SUCCESS;
}

REGISTER_PASS("ZSliceDConcatV2DFusionPass", BUILT_IN_GRAPH_PASS, Concatv2dSlicedFusionPass);
}  // namespace fe
