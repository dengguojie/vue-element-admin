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

#include "pad_addn_fusion_pass.h"

#include <math.h>
#include <iostream>
#include <map>
#include <algorithm>

#include "external/graph/operator_factory.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe {

static const std::string PATTERN_PADD_1 = "PadD_1";
static const std::string PATTERN_PADD_2 = "PadD_2";
static const std::string PATTERN_ADDN = "AddN";
static const char* PADD = "PadD";
static const char* ADDN = "AddN";

vector<FusionPattern*> PadAddnFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("PadAddnFusion");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_PADD_1, {PADD})
      .AddOpDesc(PATTERN_PADD_2, {PADD})
      .AddOpDesc(PATTERN_ADDN, {ADDN})
      .SetInputs(PATTERN_ADDN, {PATTERN_PADD_1, PATTERN_PADD_2})
      .SetOutput(PATTERN_ADDN);
  patterns.push_back(pattern);
  return patterns;
}

Status PadAddnFusionPass::CheckPadDInfo(const ge::NodePtr& padDNode1, const ge::NodePtr& padDNode2,
                                        const ge::NodePtr& addNNode, std::map<uint32_t, ge::NodePtr>& mapPadDNode) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckPadDInfo begin");

  FUSION_PASS_CHECK(padDNode1->GetOutDataNodes().size() > 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "padDNode1:[%s]: padDNode1 outdata size is invalid.",
                            padDNode1->GetName().c_str()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(padDNode2->GetOutDataNodes().size() > 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "padDNode2:[%s]: padDNode2 outdata size is invalid.",
                            padDNode2->GetName().c_str()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(
      addNNode->GetInDataNodes().size() != SIZE_LIMIT_TWO,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "addNNode:[%s]: PadAddN graph fusion only support 2 inputs for addNNode.",
              addNNode->GetName().c_str()),
      return NOT_CHANGED);

  ge::OpDescPtr padD1Desc = padDNode1->GetOpDesc();
  FUSION_PASS_CHECK(padD1Desc == nullptr, OP_LOGD(FUSED_OP_TYPE.c_str(), "padDNode1's OpDesc is null, fusion failed."),
                    return NOT_CHANGED);
  ge::OpDescPtr padD2Desc = padDNode2->GetOpDesc();
  FUSION_PASS_CHECK(padD2Desc == nullptr, OP_LOGD(FUSED_OP_TYPE.c_str(), "padDNode2's OpDesc is null, fusion failed."),
                    return NOT_CHANGED);

  vector<int64_t> padInputShape1 = padD1Desc->GetInputDesc(0).GetShape().GetDims();
  FUSION_PASS_CHECK(padInputShape1.size() != SIZE_LIMIT_TWO,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "padInputShape1 only supports two dimensions"), return NOT_CHANGED);
  vector<int64_t> padInputShape2 = padD2Desc->GetInputDesc(0).GetShape().GetDims();
  FUSION_PASS_CHECK(padInputShape2.size() != SIZE_LIMIT_TWO,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "padInputShape2 only supports two dimensions"), return NOT_CHANGED);
  FUSION_PASS_CHECK(padInputShape1[0] != padInputShape2[0],
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "padInputShape1 hava different dim[0] with padInputShape2"),
                    return NOT_CHANGED);

  Operator opPadD1 = ge::OpDescUtils::CreateOperatorFromNode(padDNode1);
  Operator opPadD2 = ge::OpDescUtils::CreateOperatorFromNode(padDNode2);
  vector<vector<int64_t>> paddingsValue1;
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != opPadD1.GetAttr("paddings", paddingsValue1),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "padDNode1 get paddings failed!"), return NOT_CHANGED);

  vector<vector<int64_t>> paddingsValue2;
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != opPadD2.GetAttr("paddings", paddingsValue2),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "padDNode2 get paddings failed!"), return NOT_CHANGED);

  bool nonZeroCheck = (paddingsValue1[0][0] != 0 || paddingsValue1[0][1] != 0 || paddingsValue2[0][0] != 0 ||
                       paddingsValue2[0][1] != 0);
  FUSION_PASS_CHECK(nonZeroCheck,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "The value of the first dimension of the paddings is not 0"),
                    return NOT_CHANGED);

  bool rightConcatLeft = (paddingsValue1[1][0] == 0 && paddingsValue2[1][1] == 0 &&
                          paddingsValue1[1][1] == padInputShape2[1] && paddingsValue2[1][0] == padInputShape1[1]);
  bool leftConcatRight = (paddingsValue1[1][1] == 0 && paddingsValue2[1][0] == 0 &&
                          paddingsValue1[1][0] == padInputShape2[1] && paddingsValue2[1][1] == padInputShape1[1]);
  FUSION_PASS_CHECK(!(rightConcatLeft || leftConcatRight),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "paddings dim value check failed"), return NOT_CHANGED);

  mapPadDNode[0] = padDNode1;
  mapPadDNode[1] = padDNode2;
  if (leftConcatRight) {
    mapPadDNode[0] = padDNode2;
    mapPadDNode[1] = padDNode1;
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckPadDInfo end");
  return SUCCESS;
}

Status PadAddnFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into PadAddnFusionPass");

  ge::NodePtr padDNode1 = GetNodeFromMapping(PATTERN_PADD_1, mapping);
  FUSION_PASS_CHECK(padDNode1 == nullptr, OP_LOGI("padd1 Node is null."), return NOT_CHANGED);
  ge::NodePtr padDNode2 = GetNodeFromMapping(PATTERN_PADD_2, mapping);
  FUSION_PASS_CHECK(padDNode2 == nullptr, OP_LOGI("padd2 Node is null."), return NOT_CHANGED);
  ge::NodePtr addNNode = GetNodeFromMapping(PATTERN_ADDN, mapping);
  FUSION_PASS_CHECK(addNNode == nullptr, OP_LOGI("addN Node is null."), return NOT_CHANGED);

  std::map<uint32_t, ge::NodePtr> mapPadDNode;
  FUSION_PASS_CHECK(CheckPadDInfo(padDNode1, padDNode2, addNNode, mapPadDNode) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Failed to check padd info"), return NOT_CHANGED);

  ge::OpDescPtr concatV2DOp = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (concatV2DOp = std::make_shared<ge::OpDesc>(addNNode->GetName() + '_' + "ConcatV2D", "ConcatV2D")),
      return INTERNAL_ERROR);
  concatV2DOp->AddDynamicInputDesc("x", 2);
  concatV2DOp->UpdateInputDesc(0, mapPadDNode[0]->GetOpDesc()->GetInputDesc(0));
  concatV2DOp->UpdateInputDesc(1, mapPadDNode[1]->GetOpDesc()->GetInputDesc(0));
  ge::GeTensorDesc outputDesc = addNNode->GetOpDesc()->GetOutputDesc(0);
  ge::AttrUtils::SetInt(concatV2DOp, "concat_dim", 1);
  concatV2DOp->AddOutputDesc("y", outputDesc);
  ge::NodePtr concatV2DNode = graph.AddNode(concatV2DOp);
  FUSION_PASS_CHECK(concatV2DNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "concatV2DNode is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OutDataAnchorPtr padDNode1InAnchor = mapPadDNode[0]->GetInDataAnchor(0)->GetPeerOutAnchor();
  ge::OutDataAnchorPtr padDNode2InAnchor = mapPadDNode[1]->GetInDataAnchor(0)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(padDNode1InAnchor, concatV2DNode->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add input data0 edge failed."),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(padDNode2InAnchor, concatV2DNode->GetInDataAnchor(1)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add input data1 edge failed."),
                    return FAILED);

  // connect output edge
  for (auto inDataAnchor : addNNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(addNNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."),
                      return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(concatV2DNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."),
                      return FAILED);
  }

  FUSION_PASS_CHECK(graph.RemoveNode(mapPadDNode[0]) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove padDNode1 node"),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(mapPadDNode[1]) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove padDNode2 node"),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(addNNode) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove addNNode node"),
                    return FAILED);
  fusionNodes.push_back(concatV2DNode);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to PadAddnFusionPass");
  return SUCCESS;
}
REGISTER_PASS("PadZeroAddnFusionPass", BUILT_IN_GRAPH_PASS, PadAddnFusionPass);
}  // namespace fe