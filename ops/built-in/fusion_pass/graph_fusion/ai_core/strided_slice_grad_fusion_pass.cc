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
 * \file strided_slice_grad_fusion_pass.cpp
 * \brief split fusion pass(strided_slice_grad --> strided_slice_grad_d)
 */
#include "strided_slice_grad_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_fusion_pass_util.h"
#include "../../../op_proto/util/op_common_util.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "StridedSliceGrad";

static const std::string PATTERN_FUSEDNODE = "FusedNodeStridedSliceGrad";

vector<FusionPattern*> ConstToAttrStridedSliceGradPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("ConstToAttrStridedSliceGradFusion");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

Status ConstToAttrStridedSliceGradPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                               vector<ge::NodePtr>& fusionNodes) {
  std::string fusionOpType = "StridedSliceGradD";
  std::vector<PassAttrInfo> strided_slice_gradAttrInfo;
  PassAttrInfo shape = {0, "shape", "SetListInt"};
  PassAttrInfo begin = {1, "begin", "SetListInt"};
  PassAttrInfo end = {2, "end", "SetListInt"};
  PassAttrInfo strides = {3, "strides", "SetListInt"};
  strided_slice_gradAttrInfo.push_back(shape);
  strided_slice_gradAttrInfo.push_back(begin);
  strided_slice_gradAttrInfo.push_back(end);
  strided_slice_gradAttrInfo.push_back(strides);

  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GetNodeFromMapping failed, fusion failed."),
                    return PARAM_INVALID);
  // Check StridedSliceGrad support dynamic
  bool unknownShape = false;
  if (ge::NodeUtils::GetNodeUnknownShapeStatus(*(fusedNode.get()), unknownShape) == ge::GRAPH_SUCCESS && unknownShape) {
    OP_LOGI("StridedSliceGrad", "unknownShape is True.");
    ge::OpDescPtr org_desc = fusedNode->GetOpDesc();
    FUSION_PASS_CHECK(org_desc == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "Fusion OP Desc is nullptr."),
                      return NOT_CHANGED);
    FUSION_PASS_CHECK(CheckOpSupported(org_desc), OP_LOGI("StridedSliceGrad", "Op StridedSliceGrad Supported Dynamic."),
                      return NOT_CHANGED);
  }
  OP_LOGI("StridedSliceGrad", "unknownShape is False.");
  ge::OpDescPtr fusionDescPtr = PatternFusionUtil::GetFusionOpDesc(fusedNode, fusionOpType, strided_slice_gradAttrInfo);
  FUSION_PASS_CHECK(fusionDescPtr == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "Fusion OP Desc is nullptr."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(!CheckOpSupported(fusionDescPtr),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Op StridedSliceGradD Not Supported."), return NOT_CHANGED);
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fusedNode);

  // check mask value
  int64_t ellipsis_mask = 0;
  int64_t new_axis_mask = 0;
  int64_t shrink_axis_mask = 0;

  if ((ge::GRAPH_SUCCESS != op.GetAttr("ellipsis_mask", ellipsis_mask)) ||
      (ge::GRAPH_SUCCESS != op.GetAttr("new_axis_mask", new_axis_mask)) ||
      (ge::GRAPH_SUCCESS != op.GetAttr("shrink_axis_mask", shrink_axis_mask))) {
    OP_LOGW(FUSED_OP_TYPE.c_str(),
            "op StridedSliceGrad get attribute ellipsis_mask or "
            "new axis mask or shrink axis mask failed");
    return NOT_CHANGED;
  }

  Tensor const_tensor1;
  op.GetInputConstData("strides", const_tensor1);

  std::vector<int64_t> strides_value;
  if (!TbeFusionPassUtil::GetConstIntData(const_tensor1, op.GetInputDescByName("strides").GetDataType(),
                                          strides_value)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "StridedSliceGrad get strides value failed.");
    return FAILED;
  }

  if (std::any_of(strides_value.begin(), strides_value.end(), [](int value) { return value != 1; })) {
    OP_LOGI(FUSED_OP_TYPE, "StridedSliceGrad has no 1 value, graph not changed.");
    return NOT_CHANGED;
  }

  Tensor const_tensor0;
  op.GetInputConstData("shape", const_tensor0);

  std::vector<int64_t> shape_value;
  if (!TbeFusionPassUtil::GetConstIntData(const_tensor0, op.GetInputDescByName("shape").GetDataType(), shape_value)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "StridedSliceGrad get shape value failed.");
    return FAILED;
  }

  const int64_t TWO = 2;
  if (shrink_axis_mask == TWO && (ellipsis_mask != 1 || strides_value.size() != TWO || shape_value.size() <= TWO)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "StridedSliceGradD can't support such param, graph not changed.");
    return NOT_CHANGED;
  }

  FUSION_PASS_CHECK(fusedNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);
  ge::NodePtr fusionNode = nullptr;
  Status ret =
      PatternFusionUtil::ConstToAttrWithNode(graph, fusedNode, fusionOpType, strided_slice_gradAttrInfo, fusionNode);
  if (ret != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "StridedSliceGrad has input which is not a CONST, graph not changed.");
    return NOT_CHANGED;
  }
  fusionNodes.push_back(fusionNode);
  return SUCCESS;
}

REGISTER_PASS("StridedSliceGradFusionPass", BUILT_IN_GRAPH_PASS, ConstToAttrStridedSliceGradPass);
}  // namespace fe
