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
 * \file decode_bbox_v2_insert_transpose_pass.cpp
 * \brief decode_bbox_v2 fusion pass
 */
#include "decode_bbox_v2_insert_transpose_pass.h"
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>
#include <algorithm>
#include "op_log.h"
#include "error_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_fusion_pass_util.h"

namespace fe {
static const string PATTERN_FUSEDNODE = "FusedNodeDecodeBboxV2";
static const string FUSED_NODE = "DecodeBboxV2";
static const int32_t INT_NUM_TWO = 2;

vector<FusionPattern*> DecodeBboxV2InsertTransposePass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("DecodeBboxV2InsertTransposePass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status DecodeBboxV2InsertTransposePass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                               vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define DecodeBboxV2InsertTransposePass fusion begin.");
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr fuseDesc = fusedNode->GetOpDesc();
  bool reversed_box = false;
  FUSION_PASS_CHECK(
      !ge::AttrUtils::GetBool(fuseDesc, "reversed_box", reversed_box),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get DecodeBboxV2's attr reversed_box failed."),
      return FAILED);
  if (reversed_box == false) {
    // do infer for fused node again, and update fused node output shape
    ge::GeTensorDesc outputDesc = fusedNode->GetOpDesc()->GetOutputDesc(0);
    vector<int64_t> oriOutputShape = outputDesc.GetShape().GetDims();

    if (oriOutputShape.empty()) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "can not get output shape. shape is empty!");
      return NOT_CHANGED;
    }

    vector<int64_t> permBoxesList = {1, 0};
    AddTransposeBeforeNode(fusedNode, 0, permBoxesList, graph);
    AddTransposeBeforeNode(fusedNode, 1, permBoxesList, graph);

    vector<int64_t> outputShapeVec;
    if (oriOutputShape.size() < INT_NUM_TWO) {
      return NOT_CHANGED;
    }
    outputShapeVec.push_back(oriOutputShape[1]);
    outputShapeVec.push_back(oriOutputShape[0]);
    ge::GeShape outputShape(outputShapeVec);
    outputDesc.SetShape(outputShape);
    outputDesc.SetOriginShape(outputShape);
    // update fused node output info
    auto opOutputDesc = fusedNode->GetOpDesc();
    opOutputDesc->UpdateOutputDesc(0, outputDesc);
    AddTransposeAfterNode(fusedNode, 0, permBoxesList, graph);
    FUSION_PASS_CHECK(
        !ge::AttrUtils::SetBool(fuseDesc, "reversed_box", true),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set DecodeBboxV2's attr reversed_box True failed."),
        return FAILED);
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "DecodeBboxV2 reversed_box is True don't need insert Transpose");
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define DecodeBboxV2InsertTransposePass fusion end");

  return SUCCESS;
}

REGISTER_PASS("DecodeBboxV2InsertTransposePass", BUILT_IN_GRAPH_PASS, DecodeBboxV2InsertTransposePass);
}  // namespace fe
