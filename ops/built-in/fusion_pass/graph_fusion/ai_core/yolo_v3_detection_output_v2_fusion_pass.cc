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
 * \file yolo_v3_detection_output_v2_fusion_pass.cpp
 * \brief
 */
#include "yolo_v3_detection_output_v2_fusion_pass.h"

#include <iostream>
#include <map>
#include <memory>


#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "fp16_t.hpp"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "pattern_fusion_util.h"

using namespace std;
using namespace ge;

namespace fe {
static const char PATTERN_YOLOV3[] = "YoloV3DetectionOutputV2";
static const char YOLOV3[] = "YoloV3DetectionOutputV2";
static const char YOLOV3_D[] = "YoloV3DetectionOutputV2D";
uint32_t YOLOV3_DEFAULT_N = 10;
uint32_t YOLOV3_PARAM_NUM_PER_YOLO = 3;

Status GenerateWIndexFP16V2(const int32_t h, const int32_t w, uint16_t* output1, const int32_t outLength) {
  if (outLength < h * w) {
    return FAILED;
  }
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

Status GenerateHIndexFP16V2(const int32_t h, const int32_t w, uint16_t* output1, const int32_t outLength) {
  if (outLength < h * w) {
    return FAILED;
  }
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

vector<FusionPattern*> YoloV3DetectionOutputV2Pass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  // yolo_v3_detection_output_v2->yolo_v3_detection_output_v2
  // define Fusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("YoloV3DetectionOutputV2Pass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  // define origin graph
  pattern->AddOpDesc(PATTERN_YOLOV3, {YOLOV3, YOLOV3_D}).SetOutput(PATTERN_YOLOV3);

  patterns.push_back(pattern);

  return patterns;
}

Status YoloV3DetectionOutputV2Pass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                           vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "enter into YoloV3DetectionOutputV2Pass");
  // diag node
  ge::NodePtr yolov3VNode = GetNodeFromMapping(PATTERN_YOLOV3, mapping);
  FUSION_PASS_CHECK(yolov3VNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "yolov3VNode is null, fusion failed."),
                    return PARAM_INVALID);

  // input of diag
  ge::OpDescPtr yolov3Desc = yolov3VNode->GetOpDesc();
  FUSION_PASS_CHECK(yolov3Desc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "yolov3VNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  // find the parent node of yolov3
  uint32_t yolov3_input_num = YOLOV3_DEFAULT_N;
  ge::AttrUtils::GetInt(yolov3Desc, "N", yolov3_input_num);
  uint32_t yolo_num = (yolov3_input_num - 1) / YOLOV3_PARAM_NUM_PER_YOLO;
  vector<ge::GeTensorPtr> weights(yolo_num * 2);
  for (uint32_t i = 0; i < yolo_num; i++) {
    ge::InDataAnchorPtr yolov3AnchorPtr = yolov3VNode->GetInDataAnchor(i);
    FUSION_PASS_CHECK(yolov3AnchorPtr == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "InDataAnchor is null, fusion failed."),
                      return PARAM_INVALID);
    ge::OutDataAnchorPtr constAnchorPtr = yolov3AnchorPtr->GetPeerOutAnchor();
    FUSION_PASS_CHECK(constAnchorPtr == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "PeerOutAnchor is null, fusion failed."),
                      return PARAM_INVALID);
    ge::NodePtr regionNode = constAnchorPtr->GetOwnerNode();
    FUSION_PASS_CHECK(regionNode == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "regionNode is null, fusion failed."),
                      return PARAM_INVALID);

    // get the input desc of entrance node to differentiate const and varj
    ge::GeTensorDesc regionAnchorPtr = regionNode->GetOpDesc()->GetInputDesc(0);

    // get the shape info
    ge::GeShape diagInputShape = regionAnchorPtr.GetShape();

    // GESHAPE->vector
    vector<int64_t> dimInfo = diagInputShape.GetDims();
    FUSION_PASS_CHECK(dimInfo.size() < 4,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "unexpected diagInputShape Dim. Dim(%d) less then 4",
                              dimInfo.size()),
                      return FAILED);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "YoloV3DetectionOutputV2Pass dimInfo%d:%d,%d,%d,%d", i, dimInfo[0], dimInfo[1],
            dimInfo[2], dimInfo[3]);

    if (PatternFusionUtil::IsUnknownShape(dimInfo[2]) ||
        PatternFusionUtil::IsUnknownShape(dimInfo[3])) {
      OP_LOGE(FUSED_OP_TYPE.c_str(), "YoloV3DetectionOutputV2Pass cannot be applied for unknown shape.");
      return FAILED;
    }

    ge::GeTensorPtr assitPtrW = nullptr;
    ge::GeTensorPtr assitPtrH = nullptr;

    unique_ptr<uint16_t[]> inputAssitW(new (std::nothrow) uint16_t[dimInfo[2] * dimInfo[3]]());
    FUSION_PASS_CHECK(inputAssitW.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssitW%d is NULL", i),
                      return PARAM_INVALID);
    unique_ptr<uint16_t[]> inputAssitH(new (std::nothrow) uint16_t[dimInfo[2] * dimInfo[3]]());
    FUSION_PASS_CHECK(inputAssitH.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssitH%d is NULL", i),
                      return PARAM_INVALID);

    int32_t outLength = dimInfo[2] * dimInfo[3];
    Status ret = GenerateWIndexFP16V2(dimInfo[2], dimInfo[3], inputAssitW.get(), outLength);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "GenerateWIndex%d failed.", i), return NOT_CHANGED);
    ret = GenerateHIndexFP16V2(dimInfo[2], dimInfo[3], inputAssitH.get(), outLength);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "GenerateHIndex%d failed.", i), return NOT_CHANGED);

    // define the shape of auxiliary matrix
    vector<int64_t> assitDimInfo;
    assitDimInfo.push_back(dimInfo[2]);
    assitDimInfo.push_back(dimInfo[3]);
    assitDimInfo.push_back(1);
    assitDimInfo.push_back(1);

    ge::GeShape assitShape(assitDimInfo);
    ge::GeTensorDesc tensorDescW(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    tensorDescW.SetShape(assitShape);
    tensorDescW.SetOriginShape(assitShape);
    tensorDescW.SetOriginFormat(ge::FORMAT_NCHW);
    ge::GeTensorDesc tensorDescH(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    tensorDescH.SetShape(assitShape);
    tensorDescH.SetOriginShape(assitShape);
    tensorDescH.SetOriginFormat(ge::FORMAT_NCHW);

    FUSION_PASS_MAKE_SHARED(
        (assitPtrW = std::make_shared<ge::GeTensor>(tensorDescW, reinterpret_cast<uint8_t*>(inputAssitW.get()),
                                                    dimInfo[2] * dimInfo[3] * sizeof(uint16_t))),
        assitPtrW = nullptr;
        return PARAM_INVALID);
    FUSION_PASS_MAKE_SHARED(
        (assitPtrH = std::make_shared<ge::GeTensor>(tensorDescH, reinterpret_cast<uint8_t*>(inputAssitH.get()),
                                                    dimInfo[2] * dimInfo[3] * sizeof(uint16_t))),
        assitPtrH = nullptr;
        return PARAM_INVALID);
    weights[i] = assitPtrW;
    weights[i + yolo_num] = assitPtrH;
  }
  ge::OpDescUtils::SetWeights(yolov3VNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(yolov3VNode);
  FUSION_PASS_CHECK(constInputNodes.size() < yolo_num * 2,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "unexpected const inputs num. num(%d) less then %d",
                            constInputNodes.size(), yolo_num * 2),
                    return FAILED);
  for (uint32_t i = 0; i < yolo_num * 2; i++) {
    NodePtr constInput = constInputNodes[i];
    constInput->GetOpDesc()->SetType("Const");
  }
  yolov3Desc->SetType("YoloV3DetectionOutputV2D");
  OP_LOGI(FUSED_OP_TYPE.c_str(), "YoloV3DetectionOutputV2Pass pass handle success!!!!");

  return SUCCESS;
}
REGISTER_PASS("YoloV3DetectionOutputV2Pass", BUILT_IN_GRAPH_PASS, YoloV3DetectionOutputV2Pass);
}  // namespace fe
