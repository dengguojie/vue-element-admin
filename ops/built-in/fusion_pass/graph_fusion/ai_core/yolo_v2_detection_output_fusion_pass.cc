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
 * \file yolo_v2_detection_output_fusion_pass.cpp
 * \brief
 */
#include "yolo_v2_detection_output_fusion_pass.h"

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

namespace fe {
static const char PATTERN_YOLOV2[] = "YoloV2DetectionOutput";
static const char YOLOV2[] = "YoloV2DetectionOutput";

Status GenWIndexFP16(const int32_t h, const int32_t w, uint16_t* output1, const int32_t outLength) {
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

Status GenHIndexFP16(const int32_t h, const int32_t w, uint16_t* output1) {
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

vector<FusionPattern*> YoloV2DetectionOutputPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  // yolo_v3_detection_output->yolo_v3_detection_output_d
  // define Fusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("YoloV2DetectionOutputPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  // define origin graph
  pattern->AddOpDesc(PATTERN_YOLOV2, {YOLOV2}).SetOutput(PATTERN_YOLOV2);

  patterns.push_back(pattern);

  return patterns;
}

Status YoloV2DetectionOutputPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "enter into YoloV2DetectionOutputPass");
  // diag node
  ge::NodePtr yolov2VNode = GetNodeFromMapping(PATTERN_YOLOV2, mapping);
  FUSION_PASS_CHECK(yolov2VNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "yolov2VNode is null, fusion failed."),
                    return PARAM_INVALID);

  // input of diag
  ge::OpDescPtr yolov2Desc = yolov2VNode->GetOpDesc();
  FUSION_PASS_CHECK(yolov2Desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "yolov2VNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  // find the parent node of yolov2
  ge::InDataAnchorPtr yolov2AnchorPtr0 = yolov2VNode->GetInDataAnchor(0);
  FUSION_PASS_CHECK(yolov2AnchorPtr0 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "InDataAnchor is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OutDataAnchorPtr constAnchorPtr0 = yolov2AnchorPtr0->GetPeerOutAnchor();
  FUSION_PASS_CHECK(constAnchorPtr0 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "PeerOutAnchor is null, fusion failed."),
                    return PARAM_INVALID);
  ge::NodePtr regionNode = constAnchorPtr0->GetOwnerNode();
  FUSION_PASS_CHECK(regionNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "regionNode is null, fusion failed."),
                    return PARAM_INVALID);

  // get the input desc of entrance node to differentiate const and varj
  ge::GeTensorDesc region1AnchorPtr0 = regionNode->GetOpDesc()->GetInputDesc(0);

  // get the shape info
  ge::GeShape diagInputShape1 = region1AnchorPtr0.GetShape();

  // GESHAPE->vector
  vector<int64_t> dimInfo1 = diagInputShape1.GetDims();
  FUSION_PASS_CHECK(dimInfo1.size() < 4,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "unexpected diagInputShape Dim. Dim(%lu) less then 4",
                            dimInfo1.size()),
                    return FAILED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "YoloV2DetectionOutputPass dimInfo1:%d,%d,%d,%d", dimInfo1[0], dimInfo1[1],
          dimInfo1[2], dimInfo1[3]);

  if (PatternFusionUtil::IsUnknownShape(dimInfo1[2]) ||
      PatternFusionUtil::IsUnknownShape(dimInfo1[3])) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "YoloV2DetectionOutputPass cannot be applied for unknown shape.");
    return FAILED;
  }

  ge::GeTensorPtr assitPtrW1 = nullptr;
  ge::GeTensorPtr assitPtrH1 = nullptr;

  unique_ptr<uint16_t[]> inputAssitW1(new (std::nothrow) uint16_t[dimInfo1[2] * dimInfo1[3]]());
  FUSION_PASS_CHECK(inputAssitW1.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssitW1 is NULL"),
                    return PARAM_INVALID);
  unique_ptr<uint16_t[]> inputAssitH1(new (std::nothrow) uint16_t[dimInfo1[2] * dimInfo1[3]]());
  FUSION_PASS_CHECK(inputAssitH1.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssitH1 is NULL"),
                    return PARAM_INVALID);

  int32_t outLength = dimInfo1[2] * dimInfo1[3];
  Status ret = GenWIndexFP16(dimInfo1[2], dimInfo1[3], inputAssitW1.get(), outLength);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "GenerateWIndex1 failed."), return NOT_CHANGED);
  ret = GenHIndexFP16(dimInfo1[2], dimInfo1[3], inputAssitH1.get());
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "GenerateHIndex1 failed."), return NOT_CHANGED);

  // define the shape of auxiliary matrix
  vector<int64_t> assitDimInfo1;
  assitDimInfo1.push_back(dimInfo1[2]);
  assitDimInfo1.push_back(dimInfo1[3]);
  assitDimInfo1.push_back(1);
  assitDimInfo1.push_back(1);

  ge::GeShape assitShape1(assitDimInfo1);
  ge::GeTensorDesc tensorDescW1(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensorDescW1.SetShape(assitShape1);
  tensorDescW1.SetOriginShape(assitShape1);
  tensorDescW1.SetOriginFormat(ge::FORMAT_NCHW);
  ge::GeTensorDesc tensorDescH1(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tensorDescH1.SetShape(assitShape1);
  tensorDescH1.SetOriginShape(assitShape1);
  tensorDescH1.SetOriginFormat(ge::FORMAT_NCHW);

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

  vector<ge::GeTensorPtr> weights = {assitPtrW1, assitPtrH1};
  ge::OpDescUtils::SetWeights(yolov2VNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(yolov2VNode);
  FUSION_PASS_CHECK(constInputNodes.size() < 2,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "unexpected const inputs num. num(%lu) less then 2",
                            constInputNodes.size()),
                    return FAILED);
  NodePtr constInput0 = constInputNodes[0];
  constInput0->GetOpDesc()->SetType("Const");
  NodePtr constInput1 = constInputNodes[1];
  constInput1->GetOpDesc()->SetType("Const");

  yolov2Desc->SetType("YoloV2DetectionOutputD");
  OP_LOGI(FUSED_OP_TYPE.c_str(), "YoloV2DetectionOutputPass pass handle success!!!!");

  return SUCCESS;
}

REGISTER_PASS("YoloV2DetectionOutputPass", BUILT_IN_GRAPH_PASS, YoloV2DetectionOutputPass);
}  // namespace fe

