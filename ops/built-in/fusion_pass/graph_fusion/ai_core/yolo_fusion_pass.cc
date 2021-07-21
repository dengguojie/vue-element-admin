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
 * \file yolo_fusion_pass.cpp
 * \brief
 */
#include "yolo_fusion_pass.h"
#include <iostream>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "fp16_t.hpp"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "pattern_fusion_util.h"

namespace fe {
static const char PATTERN_YOLO[] = "Yolo";
static const char YOLO[] = "Yolo";
static const char YOLO_ATTR_BOXES[] = "boxes";
static const char YOLO_ATTR_COORDS[] = "coords";
static const char YOLO_ATTR_CLASSES[] = "classes";

vector<FusionPattern*> YoloPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  // define Fusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("YoloPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  // define origin graph
  pattern->AddOpDesc(PATTERN_YOLO, {YOLO}).SetOutput(PATTERN_YOLO);

  patterns.push_back(pattern);

  return patterns;
}

Status YoloPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "enter into YoloPass");
  // diag node
  ge::NodePtr yoloNode = GetNodeFromMapping(PATTERN_YOLO, mapping);
  FUSION_PASS_CHECK(yoloNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "yoloNode is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OpDescPtr yoloDesc = yoloNode->GetOpDesc();
  ge::GeTensorDesc yoloInputDesc = yoloDesc->GetInputDesc(0);

  vector<int64_t> inputShape = yoloInputDesc.GetShape().GetDims();
  FUSION_PASS_CHECK(inputShape.size() < 4,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node[%s] input shape less then 4.", yoloNode->GetName().c_str()),
                    return FAILED);

  for (size_t i = 2; i <= 3; i++) {
    auto dim = inputShape[i];
    if (PatternFusionUtil::IsUnknownShape(dim)) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "YoloPass cannot be applied for unknown shape.");
      return FAILED;
    }
  }

  int64_t boxesNum = 3;
  int64_t coordsNum = 4;
  int64_t classesNum = 80;

  if (!ge::AttrUtils::GetInt(yoloDesc, YOLO_ATTR_BOXES, boxesNum)) {
    boxesNum = 3;
  }

  if (!ge::AttrUtils::GetInt(yoloDesc, YOLO_ATTR_COORDS, coordsNum)) {
    coordsNum = 4;
  }

  if (!ge::AttrUtils::GetInt(yoloDesc, YOLO_ATTR_CLASSES, classesNum)) {
    classesNum = 80;
  }

  int64_t hwSize = inputShape[2] * inputShape[3];
  std::vector<std::vector<int64_t>> oriShape;
  std::vector<int64_t> coordsOriShape;
  coordsOriShape.push_back(inputShape[0]);
  coordsOriShape.push_back(boxesNum * coordsNum);
  coordsOriShape.push_back(hwSize);
  std::vector<int64_t> objOriShape;
  objOriShape.push_back(inputShape[0]);
  objOriShape.push_back(boxesNum * hwSize);
  std::vector<int64_t> classesOriShape;
  classesOriShape.push_back(inputShape[0]);
  classesOriShape.push_back(classesNum);
  classesOriShape.push_back(boxesNum * hwSize);
  oriShape.push_back(coordsOriShape);
  oriShape.push_back(objOriShape);
  oriShape.push_back(classesOriShape);

  for (uint64_t i = 0; i < 3; i++) {
    ge::GeTensorDesc yoloOutputDesc = yoloDesc->GetOutputDesc(i);
    yoloOutputDesc.SetOriginShape(ge::GeShape(oriShape[i]));
    yoloDesc->UpdateOutputDesc(i, yoloOutputDesc);
    ge::OutDataAnchorPtr oriOutAnchorPtr = yoloNode->GetOutDataAnchor(i);
    auto oriTopPeerAnchors = oriOutAnchorPtr->GetPeerInDataAnchors();
    for (uint64_t j = 0; j < oriTopPeerAnchors.size(); j++) {
      ge::InDataAnchorPtr oriTopPeerAnchorPtr = oriTopPeerAnchors.at(j);
      ge::NodePtr outputNode = oriTopPeerAnchorPtr->GetOwnerNode();
      int idx = oriTopPeerAnchorPtr->GetIdx();
      ge::GeTensorDesc nextInputDesc = outputNode->GetOpDesc()->GetInputDesc(idx);
      nextInputDesc.SetOriginShape(ge::GeShape(oriShape[i]));
      outputNode->GetOpDesc()->UpdateInputDesc(idx, nextInputDesc);
    }
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "YoloPass success!!!!");

  return SUCCESS;
}
REGISTER_PASS("YoloPass", BUILT_IN_GRAPH_PASS, YoloPass);
}  // namespace fe
