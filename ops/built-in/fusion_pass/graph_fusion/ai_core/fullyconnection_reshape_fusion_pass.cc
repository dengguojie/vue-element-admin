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
  FUSION_PASS_CHECK(reshapeDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reshapeNode's OpDesc is null."),
                    return PARAM_INVALID);

  // reshape input shape dim0 need same with reshape output shape dim0 or dim0 is zero
  ge::GeTensorDesc reshapeInput0Desc = reshapeDesc->GetInputDesc(0);
  ge::GeTensorDesc reshapeOutput0Desc = reshapeDesc->GetOutputDesc(0);
  ge::GeShape reshapeInputShape = reshapeInput0Desc.GetShape();
  ge::GeShape reshapeOutputShape = reshapeOutput0Desc.GetShape();
  int64_t reshapeInputDim0Value = reshapeInputShape.GetDim(0);
  int64_t reshapeOutputDim0Value = reshapeOutputShape.GetDim(0);
  if (PatternFusionUtil::IsUnknownShape(reshapeInputDim0Value) ||
      PatternFusionUtil::IsUnknownShape(reshapeOutputDim0Value)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                   "FullyConnectionReshapePass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }
  if ((reshapeInputDim0Value != reshapeOutputDim0Value) && (reshapeInputDim0Value != 0)) {
    OP_LOGI(
        FUSED_OP_TYPE.c_str(),
        "reshapeInputDim0Value is not same with reshapeOutputDim0Value fusion, FullyConnectionReshapePass fusion end");
    return SUCCESS;
  }

  // refesh the fullyConnection input desc
  FUSION_PASS_CHECK(fullyConnectionDesc->UpdateInputDesc(0, reshapeInput0Desc) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "refesh the fullyConnection input failed"),
                    return FAILED);
  // adjust weights shape
  bool transposeAttr = false;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetBool(fullyConnectionNode->GetOpDesc(), "transpose", transposeAttr),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Get node[%s]'s transpose attr not success.",
                            fullyConnectionNode->GetName().c_str()),
                    return PARAM_INVALID);
  auto xShape = reshapeInputShape.GetDims();
  FUSION_PASS_CHECK(xShape.size() <= 1,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "size of [%ld] input shape of reshape must > 1.", xShape.size()),
                    return PARAM_INVALID);
  if (xShape.size() == kDimNumTwo) {
    xShape.push_back(1);
    xShape.push_back(1);
  } else if (xShape.size() == kDimNumThree) {
    xShape.push_back(1);
  }

  ge::GeTensorDesc fullyConnectionWeightDesc = fullyConnectionDesc->GetInputDesc(1);
  auto wShape = fullyConnectionWeightDesc.GetShape().GetDims();
  if (wShape.size() < kDimNumTwo) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "node[FullyConnection]'s weigth sizeis less 2, FullyConnectionReshapePass fusion end");
    return PARAM_INVALID;
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
  FUSION_PASS_CHECK(fullyConnectionDesc->UpdateInputDesc(1, fullyConnectionWeightDesc) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "refesh the fullyConnection weight failed"),
                    return FAILED);
  // delete reshape node, it will add edge automaticlly
  FUSION_PASS_CHECK(graph.RemoveNode(reshapeNode) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove node failed"), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "FullyConnectionReshapePass fusion success");
  return SUCCESS;
}

REGISTER_PASS("AFullyConnectionReshapePass", BUILT_IN_GRAPH_PASS, FullyConnectionReshapePass);
} // namespace fe
