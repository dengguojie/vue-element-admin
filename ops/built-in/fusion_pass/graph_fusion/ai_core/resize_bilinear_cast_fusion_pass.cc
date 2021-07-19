/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file resize_bilinear_cast_fusion_pass.cpp
 * \brief ResizeBilinearV2 + Cast--->ResizeBilinearV2 fusion pass
 *   (ResizeBilinearV2 + Cast--->ResizeBilinearV2)
 */
#include "resize_bilinear_cast_fusion_pass.h"

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
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;
using namespace std;

namespace fe {

// node type
static const string RESIZE_NODE = "ResizeBilinearV2";
static const string CAST_NODE = "Cast";

// node name id
static const string PATTERN_RESIZE = "ResizeBilinearV2";
static const string PATTERN_CAST = "Cast";

/*
before:
             ResizeBilinearV2
                |
                |
              Cast
                |
               xxx

 after:
                |
                |
              ResizeBilinearV2
                |
               xxx
*/

vector<FusionPattern*> ResizeBilinearV2CastFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("ResizeBilinearV2CastFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_RESIZE, {RESIZE_NODE})
      .AddOpDesc(PATTERN_CAST, {CAST_NODE})
      .SetInputs(PATTERN_CAST, {PATTERN_RESIZE})
      .SetOutput(PATTERN_CAST);

  patterns.push_back(pattern);

  return patterns;
}

Status ResizeBilinearV2CastFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                              vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter graph fusion ResizeBilinearV2CastFusionPass!");

  ge::NodePtr resizeNode = GetNodeFromMapping(RESIZE_NODE, mapping);
  ge::NodePtr castNode = GetNodeFromMapping(CAST_NODE, mapping);

  FUSION_PASS_CHECK(resizeNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "ResizeBilinearV2 Node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(castNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Cast Node is null, fusion failed."),
                    return PARAM_INVALID);

  // check Cast node
  ge::OpDescPtr resizeDesc = resizeNode->GetOpDesc();
  FUSION_PASS_CHECK(resizeDesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "resizeNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr castDesc = castNode->GetOpDesc();
  FUSION_PASS_CHECK(castDesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "castNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  ge::GeTensorDesc outputResizeDesc = resizeDesc->GetOutputDesc(0);
  ge::GeTensorDesc inputCastDesc = castDesc->GetInputDesc(0);
  ge::GeTensorDesc outputCastDesc = castDesc->GetOutputDesc(0);
  Format inputCastFormat = inputCastDesc.GetFormat();
  DataType inputCastDataType = inputCastDesc.GetDataType();
  DataType outputCastDataType = outputCastDesc.GetDataType();

  if (inputCastFormat != ge::FORMAT_NC1HWC0 || inputCastDataType != ge::DT_FLOAT ||
      outputCastDataType != ge::DT_FLOAT16) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "castNode's DataType is not float32 to float16, fusion failed.");
    return NOT_CHANGED;
  }

  // set ResizeBilinearV2's output dtype to float16
  outputResizeDesc.SetDataType(outputCastDataType);
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != resizeDesc->UpdateOutputDesc("y", outputResizeDesc),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "UpdateOutputDesc node %s failed", resizeDesc->GetName().c_str()),
                    return FAILED);

  // delete Cast node
  for (auto &inDataAnchor : castNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(castNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(resizeNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
  }

  FUSION_PASS_CHECK(graph.RemoveNode(castNode) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove castNode failed."),
                    return FAILED);

  if (castNode->GetOutControlAnchor()) {
    for (auto in_control_anchor : castNode->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(
          ge::GraphUtils::AddEdge(resizeNode->GetOutControlAnchor(), in_control_anchor) != SUCCESS,
          OP_LOGE(FUSED_OP_TYPE.c_str(), "Add ResizeBilinearV2 node out control edge failed."), return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(castNode->GetOutControlAnchor(), in_control_anchor) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove Cast node out control edge failed."), return FAILED);
    }
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Leave graph fusion ResizeBilinearV2CastFusionPass!");

  return SUCCESS;
}

REGISTER_PASS("ResizeBilinearV2CastFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, ResizeBilinearV2CastFusionPass);

}  // namespace fe
