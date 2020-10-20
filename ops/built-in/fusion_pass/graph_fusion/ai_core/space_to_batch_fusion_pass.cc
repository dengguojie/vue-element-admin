/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file space_to_batch_fusion_pass.cpp
 * \brief
 */
#include "space_to_batch_fusion_pass.h"

#include <iostream>
#include <vector>
#include <map>
#include <string>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "op_log.h"

#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace std;
using namespace ge;

namespace fe {
static const string PATTERN_SPACE = "SpaceToBatch";

vector<FusionPattern*> SpaceToBatchFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("SpaceToBatchFusion");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_SPACE, {"SpaceToBatch"}).SetOutput(PATTERN_SPACE);

  patterns.push_back(pattern);

  return patterns;
}

Status SpaceToBatchFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr spaceNode = GetNodeFromMapping(PATTERN_SPACE, mapping);
  FUSION_PASS_CHECK(spaceNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "spaceNode is null, fusion failed."),
                    return PARAM_INVALID);

  std::vector<PassAttrInfo> attr_infos = {{1, "paddings", "SetListInt"}};
  const std::string fusion_op_type = "SpaceToBatchD";
  ge::OpDescPtr fusionDescPtr = PatternFusionUtil::GetFusionOpDesc(spaceNode, fusion_op_type, attr_infos);
  FUSION_PASS_CHECK(fusionDescPtr == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Fusion OP Desc is nullptr."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(!CheckOpSupported(fusionDescPtr), OP_LOGI(FUSED_OP_TYPE.c_str(), "Op Not Supported."),
                    return NOT_CHANGED);

  ge::OpDescPtr spaceDesc = spaceNode->GetOpDesc();
  FUSION_PASS_CHECK(spaceDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "spaceNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  vector<int64_t> dims = spaceDesc->GetOutputDesc("y").GetShape().GetDims();
  for (int64_t ele : dims) {
    if (ele == UNKNOWN_DIM) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "SpaceToBatchFusionPass got unknown shape, not changed");
      return NOT_CHANGED;
    }
  }

  ge::InDataAnchorPtr spaceVAnchorPtr2 = spaceNode->GetInDataAnchor(1);
  ge::OutDataAnchorPtr constAnchorPtr2 = spaceVAnchorPtr2->GetPeerOutAnchor();
  ge::NodePtr constNode2 = constAnchorPtr2->GetOwnerNode();

  ge::ConstGeTensorPtr constTensor2 = nullptr;
  ge::AttrUtils::GetTensor(constNode2->GetOpDesc(), "value", constTensor2);
  size_t constSize2 = constTensor2->GetData().GetSize();
  const uint8_t* constData2 = constTensor2->GetData().GetData();
  ge::DataType constType2 = constTensor2->GetTensorDesc().GetDataType();

  size_t numsize2;
  if (constType2 == ge::DT_INT32) {
    numsize2 = constSize2 / sizeof(int32_t);
    vector<int32_t> paddings;
    for (size_t i = 0; i < numsize2; i++) {
      paddings.push_back(*((int32_t*)constData2 + i));
    }
    ge::AttrUtils::SetListInt(spaceDesc, "paddings", paddings);
  } else {
    numsize2 = constSize2 / sizeof(int64_t);
    vector<int64_t> paddings;
    for (size_t i = 0; i < numsize2; i++) {
      paddings.push_back(*((int64_t*)constData2 + i));
    }
    ge::AttrUtils::SetListInt(spaceDesc, "paddings", paddings);
  }

  ge::GraphUtils::RemoveEdge(constAnchorPtr2, spaceVAnchorPtr2);
  ge::NodeUtils::ClearInDataAnchor(spaceNode, spaceVAnchorPtr2);
  ge::OpDescUtils::ClearInputDesc(spaceDesc, 1);
  if (PatternFusionUtil::GetOutEdgeSize(constNode2) == 0) {
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(constNode2),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove Node[%s] failed", constNode2->GetName().c_str()),
                      return FAILED);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Remove const Node:[%s].", constNode2->GetName().c_str());
  }
  vector<bool> is_input_const = {false};
  spaceDesc->SetIsInputConst(is_input_const);

  spaceDesc->SetType(fusion_op_type);
  fusionNodes.push_back(spaceNode);
  return SUCCESS;
}

REGISTER_PASS("SpaceToBatch", BUILT_IN_GRAPH_PASS, SpaceToBatchFusionPass);
}  // namespace fe
