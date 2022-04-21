/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#include "padd_remove_fusion_pass.h"

#include <math.h>

#include <algorithm>
#include <iostream>
#include <map>

#include "error_util.h"
#include "external/graph/operator_factory.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "securec.h"
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe {
static const std::string PATTERN_INPUT = "Input0";
static const std::string PATTERN_TRANSDATA1 = "TransData1";
static const std::string PATTERN_PADD = "PadD";
static const std::string PATTERN_TRANSDATA2 = "TransData2";
static const std::string PATTERN_CONCATV2D = "ConcatV2D";

static const char* TRANSDATA = "TransData";
static const char* PADD = "PadD";
static const char* CONCATV2D = "ConcatV2D";
static const char* CONCATV2 = "ConcatV2";

static const int64_t ALIGN_UNIT_16 = 16;
static const int32_t ALLOWED_DIM_NUM = 2;

static const int32_t TRANSDATA1_IDX = 0;
static const int32_t PADD_IDX = 1;
static const int32_t TRANSDATA2_IDX = 2;
static const int32_t CONCATV2D_IDX = 3;

/*!
 * @brief Define pattern.
 * The graph struct need to adapt and target is shown as follows:
 *    preNode
 *        |
 *    TransData          PreNode
 *        |                \    /
 *      PadD       ==>    ConcatV2D
 *        |                   |
 *    TransData            PostNode
 *        \     /
 *       ConcatV2D
 *           |
 *        PostNode
 *
 * @return vector<FusionPattern*> All valid patterns.
 */
vector<FusionPattern*> PadDRemoveFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("PadDRemoveFusion");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create a pattern object."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_INPUT)
      .AddOpDesc(PATTERN_TRANSDATA1, {TRANSDATA})
      .AddOpDesc(PATTERN_PADD, {PADD})
      .AddOpDesc(PATTERN_TRANSDATA2, {TRANSDATA})
      .AddOpDesc(PATTERN_CONCATV2D, {CONCATV2D, CONCATV2})
      .SetInputs(PATTERN_PADD, {PATTERN_TRANSDATA1})
      .SetInputs(PATTERN_TRANSDATA2, {PATTERN_PADD})
      .SetInputs(PATTERN_CONCATV2D, {PATTERN_TRANSDATA2, PATTERN_INPUT})
      .SetOutput(PATTERN_CONCATV2D);
  patterns.push_back(pattern);
  return patterns;
}

Status PadDRemoveFusionPass::CheckFusedNodes(vector<ge::NodePtr>& fusedNodes) const {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckFusedNodes begin.");

  FUSION_PASS_CHECK(fusedNodes.size() <= CONCATV2D_IDX,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "FusedNodes size should be greater than %d, cur is %d. not changed.",
                            CONCATV2D_IDX, fusedNodes.size()),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(HasUnKnowShape(fusedNodes[TRANSDATA1_IDX]) || HasUnKnowShape(fusedNodes[PADD_IDX]) ||
                        HasUnKnowShape(fusedNodes[TRANSDATA2_IDX]) || HasUnKnowShape(fusedNodes[CONCATV2D_IDX]),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "PadDRemoveFusion do not support dynamic shape. not changed."),
                    return NOT_CHANGED);

  OpDescPtr transDataOpDesc1 = fusedNodes.at(TRANSDATA1_IDX)->GetOpDesc();
  FUSION_PASS_CHECK(transDataOpDesc1 == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to get op desc. not changed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(fusedNodes.at(TRANSDATA1_IDX)->GetOutDataNodes().size() > 1,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Output node num of transdata is more than 1. not changed."),
                    return NOT_CHANGED);

  OpDescPtr padDOpDesc = fusedNodes.at(PADD_IDX)->GetOpDesc();
  FUSION_PASS_CHECK(padDOpDesc == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to get op desc. not changed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(fusedNodes.at(PADD_IDX)->GetOutDataNodes().size() > 1,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "The output node num of padD is more than 1. not changed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK((padDOpDesc->GetInputDesc(0).GetShape().GetDimNum() != ALLOWED_DIM_NUM),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Dim num of x should be %d. not changed.", ALLOWED_DIM_NUM),
                    return NOT_CHANGED);

  vector<vector<int64_t>> paddingsValue;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListListInt(padDOpDesc, "paddings", paddingsValue),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to get op attr."), return FAILED);
  FUSION_PASS_CHECK((paddingsValue.size() != ALLOWED_DIM_NUM),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "The size of paddingsValue should be %d, cur is %d. not changed.",
                            ALLOWED_DIM_NUM, paddingsValue.size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK((paddingsValue.at(0).size() != ALLOWED_DIM_NUM),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "The size of paddingsValue[0] should be %d, cur is %d. not changed.",
                            ALLOWED_DIM_NUM, paddingsValue.at(0).size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK((paddingsValue.at(1).size() != ALLOWED_DIM_NUM),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "The size of paddingsValue[1] should be %d, cur is %d. not changed.",
                            ALLOWED_DIM_NUM, paddingsValue.at(1).size()),
                    return NOT_CHANGED);

  OpDescPtr transDataOpDesc2 = fusedNodes.at(TRANSDATA2_IDX)->GetOpDesc();
  FUSION_PASS_CHECK(transDataOpDesc2 == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to get op desc. not changed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(fusedNodes.at(TRANSDATA2_IDX)->GetOutDataNodes().size() > 1,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Output node num of transdata is more than 1. not changed."),
                    return NOT_CHANGED);

  vector<int64_t> transData1InputDims = transDataOpDesc1->GetInputDesc(0).GetShape().GetDims();
  vector<int64_t> transData2OutputDims = transDataOpDesc2->GetOutputDesc(0).GetShape().GetDims();

  bool condition1 = false;
  if (transDataOpDesc1->GetInputDesc(0).GetFormat() == ge::FORMAT_FRACTAL_NZ) {
    condition1 = true;
  }

  bool condition2 = false;
  if (transDataOpDesc2->GetOutputDesc(0).GetFormat() == ge::FORMAT_FRACTAL_NZ) {
    condition2 = true;
  }

  bool condition3 = false;
  if (transData1InputDims == transData2OutputDims) {
    condition3 = true;
  }

  bool condition4 = false;
  if (paddingsValue[0][0] == 0 && paddingsValue[1][0] == 0) {
    condition4 = true;
  }

  if (!condition1 || !condition2 || !condition3 || !condition4) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Not all fusion conditions are met. conditions:[%d,%d,%d,%d]. not changed.",
            condition1, condition2, condition3, condition4);
    return NOT_CHANGED;
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckFusedNodes successful.");

  return SUCCESS;
}

Status PadDRemoveFusionPass::RemoveNodes(ge::ComputeGraph& graph, vector<ge::NodePtr>& fusedNodes) const {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "RemoveNodes start.");

  for (auto nodePtr : fusedNodes) {
    if ((nodePtr->GetType() == CONCATV2D) || (nodePtr->GetType() == CONCATV2)) {
      continue;
    }
    FUSION_PASS_CHECK(
        graph.RemoveNode(nodePtr) != ge::GRAPH_SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove node:%s.", nodePtr->GetName().c_str()),
        return FAILED);
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "RemoveNodes successful.");

  return SUCCESS;
}

Status PadDRemoveFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "PadDRemoveFusionPass start.");

  NodePtr transDataNode1 = GetNodeFromMapping(PATTERN_TRANSDATA1, mapping);
  FUSION_PASS_CHECK(transDataNode1 == nullptr, OP_LOGI("TransData1 node is null."), return NOT_CHANGED);
  NodePtr padDNode = GetNodeFromMapping(PATTERN_PADD, mapping);
  FUSION_PASS_CHECK(padDNode == nullptr, OP_LOGI("PadD node is null."), return NOT_CHANGED);
  NodePtr transDataNode2 = GetNodeFromMapping(PATTERN_TRANSDATA2, mapping);
  FUSION_PASS_CHECK(transDataNode2 == nullptr, OP_LOGI("TransData2 node is null."), return NOT_CHANGED);
  NodePtr concatV2DNode = GetNodeFromMapping(PATTERN_CONCATV2D, mapping);
  FUSION_PASS_CHECK(concatV2DNode == nullptr, OP_LOGI("ConcatV2D node is null."), return NOT_CHANGED);

  vector<ge::NodePtr> fusedNodes = {transDataNode1, padDNode, transDataNode2, concatV2DNode};

  FUSION_PASS_CHECK(CheckFusedNodes(fusedNodes) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Does not fit the fusion scene."), return NOT_CHANGED);
  FUSION_PASS_CHECK(RemoveNodes(graph, fusedNodes) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove node."), return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "PadDRemoveFusionPass successful.");

  return SUCCESS;
}

REGISTER_PASS("APadDRemoveFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, PadDRemoveFusionPass);
}  // namespace fe
