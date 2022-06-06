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
 * \file fullyconnection_reshape_fusion_pass.cpp
 * \brief
 */
#include "fullyconnection_reshape_fusion_pass.h"

#include <iostream>
#include <vector>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"

#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"

using namespace std;
using namespace ge;

namespace fe {
static const string PATTERN_RESHAPE = "Reshape";
static const string PATTERN_FULLYCONNECTION = "FullyConnection";
static const int kDimNumZero = 0;
static const int kDimNumOne = 1;
static const int kDimNumTwo = 2;
static const int kDimNumThree = 3;

/*
 Reshape ---> FullyConnection
*/

vector<FusionPattern*> FullyConnectionReshapePass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern1 = new (std::nothrow) FusionPattern("FullyConnectionReshapePass");
  FUSION_PASS_CHECK(pattern1 == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "new an object failed"), return patterns);

  pattern1->AddOpDesc(PATTERN_RESHAPE, {"Reshape", "FlattenV2"})
      .AddOpDesc(PATTERN_FULLYCONNECTION, {"FullyConnection"})
      .SetInputs(PATTERN_FULLYCONNECTION, {PATTERN_RESHAPE})
      .SetOutput(PATTERN_FULLYCONNECTION);
  patterns.push_back(pattern1);

  return patterns;
}

Status FullyConnectionReshapePass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                          vector<ge::NodePtr>& /* fusionNodes */) {
  ge::NodePtr reshapeNode = GetNodeFromMapping(PATTERN_RESHAPE, mapping);
  ge::NodePtr fullyConnectionNode = GetNodeFromMapping(PATTERN_FULLYCONNECTION, mapping);

  FUSION_PASS_CHECK(reshapeNode == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "reshapeNode is null"),
                    return fe::NOT_CHANGED);
  FUSION_PASS_CHECK(fullyConnectionNode == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "fullyConnectionNode is null"),
                    return fe::NOT_CHANGED);

  FUSION_PASS_CHECK(reshapeNode->GetOutDataNodes().size() > 1,
                    OP_LOGW(reshapeNode, "output data nodes can not greater than 1"), return fe::NOT_CHANGED);

  ge::OpDescPtr fullyConnectionDesc = fullyConnectionNode->GetOpDesc();
  FUSION_PASS_CHECK(fullyConnectionDesc == nullptr,
                    OP_LOGW(fullyConnectionNode, "fullyConnectionNode's OpDesc is null."), return fe::NOT_CHANGED);
  int64_t axis = 0;
  FUSION_PASS_CHECK(
      !ge::AttrUtils::GetInt(fullyConnectionNode->GetOpDesc(), "axis", axis),
      OP_LOGI(fullyConnectionNode, "Get node[%s]'s axis attr not success.", fullyConnectionNode->GetName().c_str()),
      return fe::NOT_CHANGED);
  FUSION_PASS_CHECK(
      axis != 1,
      OP_LOGI(fullyConnectionNode,
              "node[FullyConnection]'s axis is not 1, not support fusion, FullyConnectionReshapePass fusion end"),
      return fe::NOT_CHANGED);
  ge::OpDescPtr reshapeDesc = reshapeNode->GetOpDesc();
  FUSION_PASS_CHECK(reshapeDesc == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "ReshapeNode's OpDesc is null."),
                    return NOT_CHANGED);

  // reshape input shape dim0 need same with reshape output shape dim0 or dim0 is zero
  ge::GeTensorDesc reshapeInput0Desc = reshapeDesc->GetInputDesc(0);
  ge::GeTensorDesc reshapeOutput0Desc = reshapeDesc->GetOutputDesc(0);
  ge::GeShape reshapeInputShape = reshapeInput0Desc.GetShape();
  ge::GeShape reshapeOutputShape = reshapeOutput0Desc.GetShape();
  int64_t reshapeInputDim0Value = reshapeInputShape.GetDim(0);
  int64_t reshapeOutputDim0Value = reshapeOutputShape.GetDim(0);
  // get and check reshapeInputDim0Value, reshapeOutputDim0Value
  FUSION_PASS_CHECK(PatternFusionUtil::IsUnknownShape(reshapeInputDim0Value) ||
                        PatternFusionUtil::IsUnknownShape(reshapeOutputDim0Value),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "FullyConnectionReshapePass cannot be applied for unknown shape."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(reshapeInputDim0Value != reshapeOutputDim0Value && reshapeInputDim0Value != 0,
                    OP_LOGD(FUSED_OP_TYPE.c_str(),
                            "ReshapeInputDim0Value is different from reshapeOutputDim0Value, "
                            "and fullyConnectionReshapePass fusion end."),
                    return NOT_CHANGED);
  // get and check transposeAttr
  bool transposeAttr = false;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetBool(fullyConnectionNode->GetOpDesc(), "transpose", transposeAttr),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Failed to get node[%s]'s transpose attr.",
                            fullyConnectionNode->GetName().c_str()),
                    return NOT_CHANGED);
  // get and check xShape
  auto xShape = reshapeInputShape.GetDims();
  FUSION_PASS_CHECK(xShape.size() <= 1,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Size of input shape must > 1, while it is [%zu].", xShape.size()),
                    return NOT_CHANGED);
  // get and check wShape
  ge::GeTensorDesc fullyConnectionWeightDesc = fullyConnectionDesc->GetInputDesc(1);
  auto wShape = fullyConnectionWeightDesc.GetShape().GetDims();
  FUSION_PASS_CHECK(
      wShape.size() < kDimNumTwo,
      OP_LOGW(FUSED_OP_TYPE.c_str(),
              "Node[FullyConnection]'s weight size is less than 2, and fullyConnectionReshapePass fusion end"),
      return NOT_CHANGED);

  // begin to change the graph
  // refesh the fullyConnection input desc
  FUSION_PASS_CHECK(
      fullyConnectionDesc->UpdateInputDesc(0, reshapeInput0Desc) != ge::GRAPH_SUCCESS,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to refesh the inputdesc of fullyConnection's input node."),
      return FAILED);
  // adjust weights shape
  if (xShape.size() == kDimNumTwo) {
    xShape.push_back(1);
    xShape.push_back(1);
  } else if (xShape.size() == kDimNumThree) {
    xShape.push_back(1);
  }
  vector<int64_t> changedWeightShape;
  if (!transposeAttr) {
    changedWeightShape.push_back(wShape[kDimNumZero]);
    changedWeightShape.push_back(xShape[kDimNumOne]);
    changedWeightShape.push_back(xShape[kDimNumTwo]);
    changedWeightShape.push_back(xShape[kDimNumThree]);
  } else {
    changedWeightShape.push_back(xShape[kDimNumOne]);
    changedWeightShape.push_back(xShape[kDimNumTwo]);
    changedWeightShape.push_back(xShape[kDimNumThree]);
    changedWeightShape.push_back(wShape[kDimNumOne]);
  }
  fullyConnectionWeightDesc.SetShape(ge::GeShape(changedWeightShape));
  fullyConnectionWeightDesc.SetOriginShape(ge::GeShape(changedWeightShape));
  // refesh the fullyConnection weight desc
  FUSION_PASS_CHECK(
      fullyConnectionDesc->UpdateInputDesc(1, fullyConnectionWeightDesc) != ge::GRAPH_SUCCESS,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to refesh the inputDesc of fullyConnection's weight node."),
      return FAILED);
  // delete reshape node, it will add edge automaticlly
  FUSION_PASS_CHECK(graph.RemoveNode(reshapeNode) != ge::GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove the reshape node."), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Do FullyConnectionReshapePass success.");
  return SUCCESS;
}

REGISTER_PASS("AFullyConnectionReshapePass", BUILT_IN_GRAPH_PASS, FullyConnectionReshapePass);
} // namespace fe
