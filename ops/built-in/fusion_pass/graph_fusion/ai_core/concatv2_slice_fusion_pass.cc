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
 * \file concatv2_slice_fusion_pass.cc
 * \brief concatv2 slice fusion pass
 */
#include "concatv2_slice_fusion_pass.h"

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
#include "tbe_fusion_pass_util.h"
#include "op_common_util.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe {
static const std::string PATTERN_CONCATV2 = "ConcatV2";
static const std::string PATTERN_SLICE_0 = "Slice0";
static const std::string PATTERN_SLICE_1 = "Slice1";
static const std::string OP_TYPE_CONCATV2 = "ConcatV2";
static const std::string OP_TYPE_SLICE = "Slice";

/*!
 * @brief Define pattern.
 * The graph struct need to adapt and target is shown as follows:
 *
 *        preNodes                preNodes
 *            |                    |    |
 *        ConcatV2                 |    |
 *         /     \        ==>      |    |
 *     Slice   Slice               |    |
 *         \     /                 |    |
 *        postNodes               postNodes
*
 * @return vector<FusionPattern*> All valid patterns.
 */
vector<FusionPattern*> Concatv2SliceFusionPass::DefinePatterns() {
  OP_LOGD(FUSED_OP_TYPE, "Enter into define Concatv2SliceFusionPass pattern");

  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("Concatv2SliceFusion");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to new a pattern object"),
                    return patterns);

  pattern->AddOpDesc(PATTERN_CONCATV2, {OP_TYPE_CONCATV2})
          .AddOpDesc(PATTERN_SLICE_0, {OP_TYPE_SLICE})
          .AddOpDesc(PATTERN_SLICE_1, {OP_TYPE_SLICE})
          .SetInputs(PATTERN_SLICE_0, {PATTERN_CONCATV2})
          .SetInputs(PATTERN_SLICE_1, {PATTERN_CONCATV2})
          .SetOutput(PATTERN_SLICE_0);
  patterns.push_back(pattern);

  OP_LOGD(FUSED_OP_TYPE, "End to define Concatv2SliceFusionPass pattern");
  return patterns;
}

bool Concatv2SliceFusionPass::IsVectorZeros(const vector<int64_t>& vec) const {
  for (size_t i = 0; i < vec.size(); i++) {
    if (vec[i] != 0) {
      return false;
    }
  }
  return true;
}

Status Concatv2SliceFusionPass::GetConstValue(const ge::NodePtr &node_name,
                                              const string &attr_name,
                                              vector<int64_t> &const_data) {
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(node_name);
  Tensor constTensor;
  op.GetInputConstData(attr_name, constTensor);
  if (!TbeFusionPassUtil::GetConstIntData(constTensor, op.GetInputDescByName(attr_name.c_str()).GetDataType(),
                                          const_data)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get [%s] value failed", attr_name.c_str());
    return NOT_CHANGED;
  }
  return SUCCESS;
}

Status Concatv2SliceFusionPass::GetSortedFusedNodes(const ge::ComputeGraph& graph, const Mapping& mapping,
                                                    vector<ge::NodePtr>& fusedNodes) {
  OP_LOGD(FUSED_OP_TYPE, "GetSortedFusedNodes begin");

  ge::NodePtr concatV2Node = GetNodeFromMapping(PATTERN_CONCATV2, mapping);
  FUSION_PASS_CHECK(concatV2Node == nullptr, OP_LOGW(FUSED_OP_TYPE, "Failed to get ConcatV2 node"),
                    return NOT_CHANGED);

  ge::Tensor concat_dim;
  ge::Operator concatV2NodeOp = ge::OpDescUtils::CreateOperatorFromNode(concatV2Node);
  if (GRAPH_SUCCESS != concatV2NodeOp.GetInputConstData("concat_dim", concat_dim)) {
    OP_LOGI(FUSED_OP_TYPE, "The concat_dim of ConcatV2 is not a constant.");
    return NOT_CHANGED;
  }

  FUSION_PASS_CHECK(concatV2Node->GetOutDataNodesSize() != OUTPUTSIZE_LIMIT_TWO,
                    OP_LOGD(FUSED_OP_TYPE, "Number of ConcatV2 output should be 2"), return NOT_CHANGED);
  ge::OutDataAnchorPtr concatOutAnchor0 = concatV2Node->GetOutDataAnchor(0);

  NodePtr sliceNode0 = concatOutAnchor0->GetPeerInDataAnchors().at(0)->GetOwnerNode();
  FUSION_PASS_CHECK(sliceNode0 == nullptr, OP_LOGW(FUSED_OP_TYPE, "Failed to get Slice(0) node"),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(sliceNode0->GetType() != OP_TYPE_SLICE,
                    OP_LOGW(FUSED_OP_TYPE, "Output data node of ConcatV2 should be Slice"),
                    return NOT_CHANGED);

  std::vector<int64_t> offsets0;
  FUSION_PASS_CHECK(GetConstValue(sliceNode0, "offsets", offsets0) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE, "Slice0 get offsets value failed."), return NOT_CHANGED);
  OP_LOGD(FUSED_OP_TYPE, "offsets0: %s", ops::to_string(offsets0).c_str());
  std::vector<int64_t> size0;
  FUSION_PASS_CHECK(GetConstValue(sliceNode0, "size", size0) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE, "Slice0 get size value failed."), return NOT_CHANGED);
  OP_LOGD(FUSED_OP_TYPE, "size0: %s", ops::to_string(size0).c_str());

  NodePtr sliceNode1 = concatOutAnchor0->GetPeerInDataAnchors().at(1)->GetOwnerNode();
  FUSION_PASS_CHECK(sliceNode1 == nullptr, OP_LOGW(FUSED_OP_TYPE, "Failed to get Slice(1) node"),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(sliceNode1->GetType() != OP_TYPE_SLICE,
                    OP_LOGW(FUSED_OP_TYPE, "Output data node of ConcatV2 should be Slice"),
                    return NOT_CHANGED);

  std::vector<int64_t> offsets1;
  FUSION_PASS_CHECK(GetConstValue(sliceNode1, "offsets", offsets1) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE, "Slice1 get offsets value failed."), return NOT_CHANGED);
  OP_LOGD(FUSED_OP_TYPE, "offsets1: %s", ops::to_string(offsets1).c_str());

  std::vector<int64_t> size1;
  FUSION_PASS_CHECK(GetConstValue(sliceNode1, "size", size1) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE, "Slice1 get size value failed."), return NOT_CHANGED);
  OP_LOGD(FUSED_OP_TYPE, "size1: %s", ops::to_string(size1).c_str());

  fusedNodes.push_back(concatV2Node);

  if (IsVectorZeros(offsets0)) {
    OP_LOGD(FUSED_OP_TYPE, "Slice ( %s ) attr offsets are all 0", sliceNode0->GetName().c_str());
    fusedNodes.push_back(sliceNode0);
    fusedNodes.push_back(sliceNode1);
  } else {
    FUSION_PASS_CHECK(!IsVectorZeros(offsets1), OP_LOGD(FUSED_OP_TYPE, "One of Slice offsets should be all 0"),
                      return NOT_CHANGED);

    OP_LOGD(FUSED_OP_TYPE, "Slice ( %s ) attr offsets are all 0", sliceNode1->GetName().c_str());
    fusedNodes.push_back(sliceNode1);
    fusedNodes.push_back(sliceNode0);
  }

  OP_LOGD(FUSED_OP_TYPE, "GetSortedFusedNodes end");
  return SUCCESS;
}

Status Concatv2SliceFusionPass::CheckSliceInfo(const ge::NodePtr& concatV2Node, const ge::NodePtr& sliceNode0,
                                               const ge::NodePtr& sliceNode1) {
  OP_LOGD(FUSED_OP_TYPE, "CheckSliceInfo begin");

  ge::OpDescPtr concatV2OpDesc = concatV2Node->GetOpDesc();
  vector<int64_t> inputDims0 = concatV2OpDesc->GetInputDesc(0).GetShape().GetDims();
  vector<int64_t> inputDims1 = concatV2OpDesc->GetInputDesc(1).GetShape().GetDims();
  vector<int64_t> concatOutputDims = concatV2OpDesc->GetOutputDesc(0).GetShape().GetDims();
  OP_LOGD(FUSED_OP_TYPE, "inputDims0 shape: %s", ops::to_string(inputDims0).c_str());
  OP_LOGD(FUSED_OP_TYPE, "inputDims1 shape: %s", ops::to_string(inputDims1).c_str());
  OP_LOGD(FUSED_OP_TYPE, "concatOutputDims shape: %s", ops::to_string(concatOutputDims).c_str());

  ge::OpDescPtr sliceOpDesc0 = sliceNode0->GetOpDesc();
  FUSION_PASS_CHECK(sliceOpDesc0 == nullptr, OP_LOGW(FUSED_OP_TYPE, "Failed to get Slice(0) op desc"),
                    return NOT_CHANGED);
  vector<int64_t> sliceOutputDims0 = sliceOpDesc0->GetOutputDesc(0).GetShape().GetDims();
  OP_LOGD(FUSED_OP_TYPE, "sliceOutputDims0 shape: %s", ops::to_string(sliceOutputDims0).c_str());

  std::vector<int64_t> offsets1;
  FUSION_PASS_CHECK(GetConstValue(sliceNode1, "offsets", offsets1) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE, "Slice1 get offsets value failed."), return NOT_CHANGED);

  ge::OpDescPtr sliceOpDesc1 = sliceNode1->GetOpDesc();
  FUSION_PASS_CHECK(sliceOpDesc1 == nullptr, OP_LOGW(FUSED_OP_TYPE, "Failed to get Slice(1) op desc"),
                    return NOT_CHANGED);
  vector<int64_t> sliceOutputDims1 = sliceOpDesc1->GetOutputDesc(0).GetShape().GetDims();
  OP_LOGD(FUSED_OP_TYPE, "sliceOutputDims1 shape: %s", ops::to_string(sliceOutputDims1).c_str());

  FUSION_PASS_CHECK(inputDims0 != sliceOutputDims0,
                    OP_LOGD(FUSED_OP_TYPE, "ConcatV2 input(0) shape and Slice(0) output shape should be same"),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(inputDims1 != sliceOutputDims1,
                    OP_LOGD(FUSED_OP_TYPE, "ConcatV2 input(1) shape and Slice(1) output shape should be same"),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(
      offsets1.size() != sliceOutputDims1.size(),
      OP_LOGD(FUSED_OP_TYPE, "Slice(1) attr offsets and Slice(1) output shape should be same length"),
      return NOT_CHANGED);
  std::vector<int64_t> tmpDims;
  for (size_t i = 0; i < offsets1.size(); i++) {
    tmpDims.push_back(offsets1[i] + sliceOutputDims1[i]);
  }
  FUSION_PASS_CHECK(tmpDims != concatOutputDims,
                    OP_LOGD(FUSED_OP_TYPE, "Slice(1) should be corresponds to ConcatV2 input(1)"),
                    return NOT_CHANGED);

  OP_LOGD(FUSED_OP_TYPE, "CheckSliceInfo end");
  return SUCCESS;
}

Status Concatv2SliceFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE, "Enter into Concatv2SliceFusionPass");

  vector<ge::NodePtr> fusedNodes;
  FUSION_PASS_CHECK(GetSortedFusedNodes(graph, mapping, fusedNodes) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE, "Does not fit the fusion scene"), return NOT_CHANGED);
  ge::NodePtr concatV2Node = fusedNodes[0];
  ge::NodePtr sliceNode0 = fusedNodes[1];
  ge::NodePtr sliceNode1 = fusedNodes[2];

  FUSION_PASS_CHECK(CheckSliceInfo(concatV2Node, sliceNode0, sliceNode1) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE, "Failed to check slice info"), return NOT_CHANGED);

  // concatV2Node InDataAnchor(0) corresponds to  sliceNode0 OutDataAnchor
  for (auto inAnchor : sliceNode0->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    inAnchor->UnlinkAll();
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(concatV2Node->GetInDataAnchor(0)->GetPeerOutAnchor(), inAnchor) != ge::GRAPH_SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from input(0) to output(0)"),
        return FAILED);
  }
  // concatV2Node InDataAnchor(1) corresponds to  sliceNode1 OutDataAnchor
  for (auto inAnchor : sliceNode1->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    inAnchor->UnlinkAll();
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(concatV2Node->GetInDataAnchor(1)->GetPeerOutAnchor(), inAnchor) != ge::GRAPH_SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from input(1) to output(1)"),
        return FAILED);
  }

  FUSION_PASS_CHECK(graph.RemoveNode(concatV2Node) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove ConcatV2 node"),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(sliceNode0) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove Slice(0) node"),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(sliceNode1) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove Slice(1) node"),
                    return FAILED);

  OP_LOGD(FUSED_OP_TYPE, "End to Concatv2SliceFusionPass");
  return SUCCESS;
}

REGISTER_PASS("ASliceConcatV2FusionPass", BUILT_IN_GRAPH_PASS, Concatv2SliceFusionPass);
}  // namespace fe

