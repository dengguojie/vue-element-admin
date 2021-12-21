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
 * \file yolo_v3_detection_output_fusion_pass.cpp
 * \brief
 */
#include "yolo_v3_detection_output_fusion_pass.h"

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
#include "error_util.h"
#include "fp16_t.hpp"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "pattern_fusion_util.h"

using namespace std;
using namespace ge;

namespace fe {
static const string PATTERN_YOLOV3 = "YoloV3DetectionOutput";
static const char* YOLOV3 = "YoloV3DetectionOutput";

Status GenerateWIndexFP16(const int32_t h, const int32_t w, uint16_t* output1) {
  for (int32_t i = 0; i < h; ++i) {
    for (int32_t j = 0; j < w; ++j) {
      fp16_t t;
      t.val = 0;
      t = j;
      output1[i * w + j] = t.val;
    }
  }
  return SUCCESS;
}

Status GenerateHIndexFP16(const int32_t h, const int32_t w, uint16_t* output1) {
  for (int32_t i = 0; i < h; ++i) {
    for (int32_t j = 0; j < w; ++j) {
      fp16_t t;
      t.val = 0;
      t = i;
      output1[i * w + j] = t.val;
    }
  }
  return SUCCESS;
}

vector<FusionPattern*> YoloV3DetectionOutputPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  // yolo_v3_detection_output->yolo_v3_detection_output_d
  // define Fusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("YoloV3DetectionOutputPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  // define origin graph
  pattern->AddOpDesc(PATTERN_YOLOV3, {YOLOV3}).SetOutput(PATTERN_YOLOV3);

  patterns.push_back(pattern);

  return patterns;
}

Status YoloV3DetectionOutputPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "enter into YoloV3DetectionOutputPass");
  // diag node
  ge::NodePtr yolov3VNode = GetNodeFromMapping(PATTERN_YOLOV3, mapping);
  FUSION_PASS_CHECK(yolov3VNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "yolov3VNode is null, fusion failed."),
                    return PARAM_INVALID);

  // input of diag
  ge::OpDescPtr yolov3Desc = yolov3VNode->GetOpDesc();
  FUSION_PASS_CHECK(yolov3Desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "yolov3VNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  // find the parent node of yolov3
  ge::InDataAnchorPtr yolov3AnchorPtr0 = yolov3VNode->GetInDataAnchor(0);
  ge::OutDataAnchorPtr constAnchorPtr0 = yolov3AnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr region1Node = constAnchorPtr0->GetOwnerNode();

  ge::InDataAnchorPtr yolov3AnchorPtr1 = yolov3VNode->GetInDataAnchor(1);
  ge::OutDataAnchorPtr constAnchorPtr1 = yolov3AnchorPtr1->GetPeerOutAnchor();
  ge::NodePtr region2Node = constAnchorPtr1->GetOwnerNode();

  ge::InDataAnchorPtr yolov3AnchorPtr2 = yolov3VNode->GetInDataAnchor(2);
  ge::OutDataAnchorPtr constAnchorPtr2 = yolov3AnchorPtr2->GetPeerOutAnchor();
  ge::NodePtr region3Node = constAnchorPtr2->GetOwnerNode();

  // get the input desc of entrance node to differentiate const and var
  ge::GeTensorDesc region1AnchorPtr0 = region1Node->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc region2AnchorPtr0 = region2Node->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc region3AnchorPtr0 = region3Node->GetOpDesc()->GetInputDesc(0);

  // get the shape info
  ge::GeShape diagInputShape1 = region1AnchorPtr0.GetShape();
  ge::GeShape diagInputShape2 = region2AnchorPtr0.GetShape();
  ge::GeShape diagInputShape3 = region3AnchorPtr0.GetShape();

  // GESHAPE->vector
  vector<int64_t> dimInfo1 = diagInputShape1.GetDims();
  vector<int64_t> dimInfo2 = diagInputShape2.GetDims();
  vector<int64_t> dimInfo3 = diagInputShape3.GetDims();
  if (dimInfo1.size() < 4 || dimInfo2.size() < 4 || dimInfo3.size() < 4) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "yolov3VNode's dimInfo is empty, fusion failed.");
    return PARAM_INVALID;
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "YoloV3DetectionOutputPass dimInfo1:%d,%d,%d,%d", dimInfo1[0], dimInfo1[1],
          dimInfo1[2], dimInfo1[3]);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "YoloV3DetectionOutputPass dimInfo2:%d,%d,%d,%d", dimInfo2[0], dimInfo2[1],
          dimInfo2[2], dimInfo2[3]);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "YoloV3DetectionOutputPass dimInfo3:%d,%d,%d,%d", dimInfo3[0], dimInfo3[1],
          dimInfo3[2], dimInfo3[3]);

  if (PatternFusionUtil::IsUnknownShape(dimInfo1[2]) || PatternFusionUtil::IsUnknownShape(dimInfo1[3]) ||
      PatternFusionUtil::IsUnknownShape(dimInfo2[2]) || PatternFusionUtil::IsUnknownShape(dimInfo2[3]) ||
      PatternFusionUtil::IsUnknownShape(dimInfo3[2]) || PatternFusionUtil::IsUnknownShape(dimInfo3[3])) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "YoloV3DetectionOutputPass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }

  ge::GeTensorPtr assitPtrW1 = nullptr;
  ge::GeTensorPtr assitPtrH1 = nullptr;
  ge::GeTensorPtr assitPtrW2 = nullptr;
  ge::GeTensorPtr assitPtrH2 = nullptr;
  ge::GeTensorPtr assitPtrW3 = nullptr;
  ge::GeTensorPtr assitPtrH3 = nullptr;

  unique_ptr<uint16_t[]> inputAssitW1(new (std::nothrow) uint16_t[dimInfo1[2] * dimInfo1[3]]());
  FUSION_PASS_CHECK(inputAssitW1.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssitW1 is NULL"),
                    return PARAM_INVALID);
  unique_ptr<uint16_t[]> inputAssitH1(new (std::nothrow) uint16_t[dimInfo1[2] * dimInfo1[3]]());
  FUSION_PASS_CHECK(inputAssitH1.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssitH1 is NULL"),
                    return PARAM_INVALID);

  unique_ptr<uint16_t[]> inputAssitW2(new (std::nothrow) uint16_t[dimInfo2[2] * dimInfo2[3]]());
  FUSION_PASS_CHECK(inputAssitW2.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssitW2 is NULL"),
                    return PARAM_INVALID);
  unique_ptr<uint16_t[]> inputAssitH2(new (std::nothrow) uint16_t[dimInfo2[2] * dimInfo2[3]]());
  FUSION_PASS_CHECK(inputAssitH2.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssitH2 is NULL"),
                    return PARAM_INVALID);

  unique_ptr<uint16_t[]> inputAssitW3(new (std::nothrow) uint16_t[dimInfo3[2] * dimInfo3[3]]());
  FUSION_PASS_CHECK(inputAssitW3.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssitW3 is NULL"),
                    return PARAM_INVALID);
  unique_ptr<uint16_t[]> inputAssitH3(new (std::nothrow) uint16_t[dimInfo3[2] * dimInfo3[3]]());
  FUSION_PASS_CHECK(inputAssitH3.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssitH3 is NULL"),
                    return PARAM_INVALID);

  Status ret = GenerateWIndexFP16(dimInfo1[2], dimInfo1[3], inputAssitW1.get());
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "GenerateWIndex1 failed."), return NOT_CHANGED);
  ret = GenerateHIndexFP16(dimInfo1[2], dimInfo1[3], inputAssitH1.get());
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "GenerateHIndex1 failed."), return NOT_CHANGED);

  ret = GenerateWIndexFP16(dimInfo2[2], dimInfo2[3], inputAssitW2.get());
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "GenerateWIndex2 failed."), return NOT_CHANGED);
  ret = GenerateHIndexFP16(dimInfo2[2], dimInfo2[3], inputAssitH2.get());
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "GenerateHIndex2 failed."), return NOT_CHANGED);

  ret = GenerateWIndexFP16(dimInfo3[2], dimInfo3[3], inputAssitW3.get());
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "GenerateWIndex3 failed."), return NOT_CHANGED);
  ret = GenerateHIndexFP16(dimInfo3[2], dimInfo3[3], inputAssitH3.get());
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "GenerateHIndex3 failed."), return NOT_CHANGED);

  // define the shape of auxiliary matrix
  vector<int64_t> assitDimInfo1;
  assitDimInfo1.push_back(dimInfo1[2]);
  assitDimInfo1.push_back(dimInfo1[3]);
  assitDimInfo1.push_back(1);
  assitDimInfo1.push_back(1);

  vector<int64_t> assitDimInfo2;
  assitDimInfo2.push_back(dimInfo2[2]);
  assitDimInfo2.push_back(dimInfo2[3]);
  assitDimInfo2.push_back(1);
  assitDimInfo2.push_back(1);

  vector<int64_t> assitDimInfo3;
  assitDimInfo3.push_back(dimInfo3[2]);
  assitDimInfo3.push_back(dimInfo3[3]);
  assitDimInfo3.push_back(1);
  assitDimInfo3.push_back(1);

  ge::GeShape assitShape1(assitDimInfo1);
  ge::GeTensorDesc tensorDescW1(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensorDescW1.SetShape(assitShape1);
  tensorDescW1.SetOriginShape(assitShape1);
  tensorDescW1.SetOriginFormat(ge::FORMAT_NCHW);
  ge::GeTensorDesc tensorDescH1(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensorDescH1.SetShape(assitShape1);
  tensorDescH1.SetOriginShape(assitShape1);
  tensorDescH1.SetOriginFormat(ge::FORMAT_NCHW);

  ge::GeShape assitShape2(assitDimInfo2);
  ge::GeTensorDesc tensorDescW2(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensorDescW2.SetShape(assitShape2);
  tensorDescW2.SetOriginShape(assitShape2);
  tensorDescW2.SetOriginFormat(ge::FORMAT_NCHW);
  ge::GeTensorDesc tensorDescH2(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensorDescH2.SetShape(assitShape2);
  tensorDescH2.SetOriginShape(assitShape2);
  tensorDescH2.SetOriginFormat(ge::FORMAT_NCHW);

  ge::GeShape assitShape3(assitDimInfo3);
  ge::GeTensorDesc tensorDescW3(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensorDescW3.SetShape(assitShape3);
  tensorDescW3.SetOriginShape(assitShape3);
  tensorDescW3.SetOriginFormat(ge::FORMAT_NCHW);
  ge::GeTensorDesc tensorDescH3(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensorDescH3.SetShape(assitShape3);
  tensorDescH3.SetOriginShape(assitShape3);
  tensorDescH3.SetOriginFormat(ge::FORMAT_NCHW);

  FUSION_PASS_MAKE_SHARED(
      (assitPtrW1 = std::make_shared<ge::GeTensor>(tensorDescW1, reinterpret_cast<uint8_t*>(inputAssitW1.get()),
                                                   dimInfo1[2] * dimInfo1[3] * sizeof(uint16_t))),
      assitPtrW1 = nullptr;
      return PARAM_INVALID);
  FUSION_PASS_MAKE_SHARED(
      (assitPtrH1 = std::make_shared<ge::GeTensor>(tensorDescH1, reinterpret_cast<uint8_t*>(inputAssitH1.get()),
                                                   dimInfo1[2] * dimInfo1[3] * sizeof(uint16_t))),
      assitPtrH1 = nullptr;
      return PARAM_INVALID);

  FUSION_PASS_MAKE_SHARED(
      (assitPtrW2 = std::make_shared<ge::GeTensor>(tensorDescW2, reinterpret_cast<uint8_t*>(inputAssitW2.get()),
                                                   dimInfo2[2] * dimInfo2[3] * sizeof(uint16_t))),
      assitPtrW2 = nullptr;
      return PARAM_INVALID);
  FUSION_PASS_MAKE_SHARED(
      (assitPtrH2 = std::make_shared<ge::GeTensor>(tensorDescH2, reinterpret_cast<uint8_t*>(inputAssitH2.get()),
                                                   dimInfo2[2] * dimInfo2[3] * sizeof(uint16_t))),
      assitPtrH2 = nullptr;
      return PARAM_INVALID);

  FUSION_PASS_MAKE_SHARED(
      (assitPtrW3 = std::make_shared<ge::GeTensor>(tensorDescW3, reinterpret_cast<uint8_t*>(inputAssitW3.get()),
                                                   dimInfo3[2] * dimInfo3[3] * sizeof(uint16_t))),
      assitPtrW3 = nullptr;
      return PARAM_INVALID);
  FUSION_PASS_MAKE_SHARED(
      (assitPtrH3 = std::make_shared<ge::GeTensor>(tensorDescH3, reinterpret_cast<uint8_t*>(inputAssitH3.get()),
                                                   dimInfo3[2] * dimInfo3[3] * sizeof(uint16_t))),
      assitPtrH3 = nullptr;
      return PARAM_INVALID);

  vector<ge::GeTensorPtr> weights = {assitPtrW1, assitPtrW2, assitPtrW3, assitPtrH1, assitPtrH2, assitPtrH3};
  ge::OpDescUtils::SetWeights(yolov3VNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(yolov3VNode);
  if (constInputNodes.size() < 6) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "yolov3VNode's ConstInputNodes is empty, fusion failed.");
    return PARAM_INVALID;
  }
  NodePtr constInput0 = constInputNodes[0];
  constInput0->GetOpDesc()->SetType("Const");
  NodePtr constInput1 = constInputNodes[1];
  constInput1->GetOpDesc()->SetType("Const");
  NodePtr constInput2 = constInputNodes[2];
  constInput2->GetOpDesc()->SetType("Const");
  NodePtr constInput3 = constInputNodes[3];
  constInput3->GetOpDesc()->SetType("Const");
  NodePtr constInput4 = constInputNodes[4];
  constInput4->GetOpDesc()->SetType("Const");
  NodePtr constInput5 = constInputNodes[5];
  constInput5->GetOpDesc()->SetType("Const");
  yolov3Desc->SetType("YoloV3DetectionOutputD");
  OP_LOGI(FUSED_OP_TYPE.c_str(), "YoloV3DetectionOutputPass pass handle success!!!!");

  return SUCCESS;
}
REGISTER_PASS("YoloV3DetectionOutputPass", BUILT_IN_GRAPH_PASS, YoloV3DetectionOutputPass);
}  // namespace fe
