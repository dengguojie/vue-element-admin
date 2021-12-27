/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
 * \file yolo_v5_detection_output_fusion_pass.cpp
 * \brief
 */
#include "yolo_v5_detection_output_fusion_pass.h"

#include <iostream>
#include <map>
#include <memory>


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
static const char PATTERN_YOLOV5[] = "YoloV5DetectionOutput";
static const char YOLOV5[] = "YoloV5DetectionOutput";
static const char YOLOV5_D[] = "YoloV5DetectionOutputD";
uint32_t YOLOV5_DEFAULT_N = 10;
uint32_t YOLOV5_PARAM_NUM_PER_YOLO = 3;
uint32_t TIMES = 2;

Status YoloV5GenerateWIndexFP16V2(const int32_t h, const int32_t w, uint16_t* output1, const int32_t outLength) {
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

Status YoloV5GenerateHIndexFP16V2(const int32_t h, const int32_t w, uint16_t* output1, const int32_t outLength) {
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

vector<FusionPattern*> YoloV5DetectionOutputPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  // yolo_v5_detection_output_v2->yolo_v5_detection_output_v2
  // define Fusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("YoloV5DetectionOutputPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "new a pattern object failed."),
                    return patterns);
  // define origin graph
  pattern->AddOpDesc(PATTERN_YOLOV5, {YOLOV5}).SetOutput(PATTERN_YOLOV5);

  patterns.push_back(pattern);

  return patterns;
}

Status YoloV5DetectionOutputPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                         vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "enter into YoloV5DetectionOutputPass");
  // diag node
  ge::NodePtr yolov5VNode = GetNodeFromMapping(PATTERN_YOLOV5, mapping);
  FUSION_PASS_CHECK(yolov5VNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                   "yolov5VNode is null, fusion failed."),
                    return PARAM_INVALID);

  // input of diag
  ge::OpDescPtr yolov5Desc = yolov5VNode->GetOpDesc();
  FUSION_PASS_CHECK(yolov5Desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "yolov5VNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  // find the parent node of yolov5
  uint32_t yolov5_input_num = YOLOV5_DEFAULT_N;
  ge::AttrUtils::GetInt(yolov5Desc, "N", yolov5_input_num);
  uint32_t yolo_num = (yolov5_input_num - 1) / YOLOV5_PARAM_NUM_PER_YOLO;
  vector<ge::GeTensorPtr> weights(yolo_num * TIMES);
  for (uint32_t i = 0; i < yolo_num; i++) {
    ge::InDataAnchorPtr yolov5AnchorPtr = yolov5VNode->GetInDataAnchor(i);
    FUSION_PASS_CHECK(yolov5AnchorPtr == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "InDataAnchor is null, fusion failed."),
                      return PARAM_INVALID);
    ge::OutDataAnchorPtr constAnchorPtr = yolov5AnchorPtr->GetPeerOutAnchor();
    FUSION_PASS_CHECK(constAnchorPtr == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "PeerOutAnchor is null, fusion failed."),
                      return PARAM_INVALID);
    ge::NodePtr regionNode = constAnchorPtr->GetOwnerNode();
    FUSION_PASS_CHECK(regionNode == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "regionNode is null, fusion failed."),
                      return PARAM_INVALID);

    // get the input desc of entrance node to differentiate const and varj
    ge::GeTensorDesc regionAnchorPtr = regionNode->GetOpDesc()->GetInputDesc(0);

    // get the shape info
    ge::GeShape diagInputShape = regionAnchorPtr.GetShape();

    // GESHAPE->vector
    int32_t h_index = 2;
    int32_t w_index = 3;
    vector<int64_t> dimInfo = diagInputShape.GetDims();
    FUSION_PASS_CHECK(dimInfo.size() < 4,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "unexpected diagInputShape Dim. Dim(%lu) less then 4",
                              dimInfo.size()),
                      return FAILED);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "YoloV5DetectionOutputPass dimInfo%d:%d,%d,%d,%d",
            i, dimInfo[0], dimInfo[1], dimInfo[h_index], dimInfo[w_index]);

    if (PatternFusionUtil::IsUnknownShape(dimInfo[h_index]) ||
        PatternFusionUtil::IsUnknownShape(dimInfo[w_index])) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
      "YoloV5DetectionOutputPass cannot be applied for unknown shape.");
      return FAILED;
    }

    ge::GeTensorPtr assitPtrW = nullptr;
    ge::GeTensorPtr assitPtrH = nullptr;
    unique_ptr<uint16_t[]> inputAssitW(new (std::nothrow) uint16_t[dimInfo[h_index] * dimInfo[w_index]]());
    FUSION_PASS_CHECK(inputAssitW.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "inputAssitW%d is NULL", i),
                      return PARAM_INVALID);
    unique_ptr<uint16_t[]> inputAssitH(new (std::nothrow) uint16_t[dimInfo[h_index] * dimInfo[w_index]]());
    FUSION_PASS_CHECK(inputAssitH.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "inputAssitH%d is NULL", i),
                      return PARAM_INVALID);

    int32_t outLength = dimInfo[h_index] * dimInfo[w_index];
    Status ret = YoloV5GenerateWIndexFP16V2(dimInfo[h_index], dimInfo[w_index], inputAssitW.get(), outLength);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "GenerateWIndex%d failed.", i),
                      return NOT_CHANGED);
    ret = YoloV5GenerateHIndexFP16V2(dimInfo[h_index], dimInfo[w_index], inputAssitH.get(), outLength);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "GenerateHIndex%d failed.", i),
                      return NOT_CHANGED);

    // define the shape of auxiliary matrix
    vector<int64_t> assitDimInfo;
    assitDimInfo.push_back(dimInfo[h_index]);
    assitDimInfo.push_back(dimInfo[w_index]);
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
                                                    dimInfo[h_index] * dimInfo[w_index] * sizeof(uint16_t))),
        assitPtrW = nullptr;
        return PARAM_INVALID);
    FUSION_PASS_MAKE_SHARED(
        (assitPtrH = std::make_shared<ge::GeTensor>(tensorDescH, reinterpret_cast<uint8_t*>(inputAssitH.get()),
                                                    dimInfo[h_index] * dimInfo[w_index] * sizeof(uint16_t))),
        assitPtrH = nullptr;
        return PARAM_INVALID);
    weights[i] = assitPtrW;
    weights[i + yolo_num] = assitPtrH;
  }
  ge::OpDescUtils::SetWeights(yolov5VNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(yolov5VNode);
  FUSION_PASS_CHECK(constInputNodes.size() < yolo_num * TIMES,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "unexpected const inputs num. num(%lu) less then %u",
                            constInputNodes.size(), yolo_num * TIMES),
                    return FAILED);
  for (uint32_t i = 0; i < yolo_num * TIMES; i++) {
    NodePtr constInput = constInputNodes[i];
    constInput->GetOpDesc()->SetType("Const");
  }
  yolov5Desc->SetType("YoloV5DetectionOutputD");
  OP_LOGI(FUSED_OP_TYPE.c_str(), "YoloV5DetectionOutputPass pass handle success!!!!");

  return SUCCESS;
}
REGISTER_PASS("YoloV5DetectionOutputPass", BUILT_IN_GRAPH_PASS, YoloV5DetectionOutputPass);
}  // namespace fe
