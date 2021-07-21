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
 * \file pass_through_fusion_pass.cpp
 * \brief
 */
#include "pass_through_fusion_pass.h"

#include "fp16_t.hpp"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"

namespace fe {
static const char PASS_THROUGH_NODE[] = "PassThrough";
static const char PATTERN_FUSEDNODE[] = "PassThroughFusion";
static const char CONSTANTOP[] = "Constant";

static const uint16_t UINT_NUM_ZERO = 0;
static const int64_t CIN_THESHOLD = 64;

template <typename Dtype>
Status passThroughAssistHelpFP16(const int32_t n, Dtype& output1, const vector<int64_t>& passThroughDInputDimInfo) {
  OP_LOGI("passThroughAssistHelpFP16", "START TO DO passThroughAssistHelpFP16.");
  Dtype* output = &output1;
  fp16_t t;
  t.val = 1;
  int32_t tmp_x = 1;
  t = tmp_x;
  FUSION_PASS_CHECK(passThroughDInputDimInfo.empty(),
                    VECTOR_FUSION_INNER_ERR_REPORT("passThroughAssistHelpFP16", "AssistHelpFP16 InputDDimInfo is empty, Create Assist exit."),
                    return FAILED);
  int32_t windowSize = passThroughDInputDimInfo[2] * passThroughDInputDimInfo[3];
  OP_LOGI("passThroughAssistHelpFP16", "START TO DO passThroughAssistHelpFP16 windowSize:%d.", windowSize);

  int32_t channelSize = passThroughDInputDimInfo[1];
  OP_LOGI("passThroughAssistHelpFP16", "START TO DO passThroughAssistHelpFP16 channelSize:%d.", channelSize);

  for (int32_t windowIdx = 0; windowIdx < windowSize; windowIdx++) {
    for (int32_t channelIdx = 0; channelIdx < channelSize; channelIdx++) {
      int32_t strideIdx = channelIdx * (channelSize + 1) * windowSize;
      int32_t fillAssistIdx = strideIdx + (windowIdx * (channelSize * channelSize * windowSize + 1));
      output[fillAssistIdx] = t.val;
      OP_LOGI("passThroughhAssitHelpFP16", "=== Idx %d ===", fillAssistIdx);
    }
  }
  return SUCCESS;
}

static Status ParseNumberIdx(ge::GeTensorDesc& tensorDesc, size_t& numberIdx) {
  ge::Format tensorGeFormat = tensorDesc.GetOriginFormat();
  if (tensorGeFormat == FORMAT_NCHW) {
    numberIdx = 0;
    return SUCCESS;
  }
  if (tensorGeFormat == FORMAT_NHWC) {
    numberIdx = 0;
    return SUCCESS;
  }
  if (tensorGeFormat == FORMAT_HWCN) {
    numberIdx = 3;
    return SUCCESS;
  }
  return FAILED;
}

vector<FusionPattern*> PassThroughFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("PassThroughFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {PASS_THROUGH_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status PassThroughFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr passThroughNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(passThroughNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "passThroughNode is null, fusion failed."), return PARAM_INVALID);
  ge::OpDescPtr passThroughDesc = passThroughNode->GetOpDesc();
  FUSION_PASS_CHECK(passThroughDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "passThroughDesc is null, fusion failed."), return PARAM_INVALID);
  std::string passThroughName = passThroughNode->GetName();

  bool reverse = true;
  FUSION_PASS_CHECK(
      !ge::AttrUtils::GetBool(passThroughDesc, "reverse", reverse),
      OP_LOGW(FUSED_OP_TYPE.c_str(), "Node[%s]: Can not get passthrough reverse attr.", passThroughName.c_str()),
      return NOT_CHANGED);
  FUSION_PASS_CHECK(
      reverse,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: passthrough reverse is true, do not fusion.", passThroughName.c_str()),
      return NOT_CHANGED);
  // get attr stride of fused node
  int32_t stride = -1;
  FUSION_PASS_CHECK(
      !ge::AttrUtils::GetInt(passThroughDesc, "stride", stride),
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: Cannot get attr:stride from node.", passThroughName.c_str()),
      return NOT_CHANGED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: The OriginDataType of node's first input is %d,", passThroughName.c_str(),
          passThroughDesc->GetInputDesc(0).GetDataType());

  // get father node of fused node
  ge::NodePtr passThroughInNode = passThroughNode->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: Get father node[Name:%s Type:%s]", passThroughName.c_str(),
          passThroughInNode->GetName().c_str(), passThroughInNode->GetType().c_str());

  ge::GeTensorDesc passThroughInput = passThroughDesc->GetInputDesc(0);
  ge::GeShape passThroughInputShape = passThroughInput.GetShape();
  vector<int64_t> passThroughInputDimInfo = passThroughInputShape.GetDims();
  ge::Format assitMatrixFormat = passThroughInput.GetFormat();

  FUSION_PASS_CHECK(passThroughInputDimInfo.empty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node[%s]: input shape is null.", passThroughName.c_str()),
                    return FAILED);

  size_t inChannelIdx = -1;
  FUSION_PASS_CHECK(
      SUCCESS != PatternFusionUtil::ParseChannelIdx(passThroughInput, inChannelIdx),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node[%s]: The original format of node's input0 is %s, which is unsupportable.",
              passThroughName.c_str(), ge::TypeUtils::FormatToSerialString(assitMatrixFormat).c_str()),
      return FAILED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: The original format of node's input0 is %s.", passThroughName.c_str(),
          ge::TypeUtils::FormatToSerialString(assitMatrixFormat).c_str());
  size_t inNumberIdx = -1;
  FUSION_PASS_CHECK(
      SUCCESS != ParseNumberIdx(passThroughInput, inNumberIdx),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node[%s]: The original format of node's input0 is %s, which is unsupportable.",
              passThroughName.c_str(), ge::TypeUtils::FormatToSerialString(assitMatrixFormat).c_str()),
      return FAILED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: The original format of node's input0 is %s.", passThroughName.c_str(),
          ge::TypeUtils::FormatToSerialString(assitMatrixFormat).c_str());

  if (PatternFusionUtil::IsUnknownShape(passThroughInputDimInfo[inChannelIdx])) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "PassThroughFusionPass cannot be applied for unknown shape.");
    return FAILED;
  }

  FUSION_PASS_CHECK(
      passThroughInputDimInfo[inChannelIdx] > CIN_THESHOLD,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: Cin[%d] > 64, keep PassThrough for performance, not fusion.",
              passThroughName.c_str(), passThroughInputDimInfo[inChannelIdx]),
      return NOT_CHANGED);

  int64_t passThroughDChannel = passThroughInputDimInfo[inChannelIdx] * stride * stride;
  vector<int64_t> passThroughDInputDimInfo = {passThroughDChannel, passThroughInputDimInfo[inChannelIdx], stride,
                                              stride};
  for (auto printInfo : passThroughDInputDimInfo) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: info of node's input1 is %d.", passThroughName.c_str(), printInfo);
  }
  ge::GeTensorPtr assitPtr = nullptr;
  ge::GeTensorDesc tensorDesc;
  int64_t destSize = passThroughDChannel * passThroughInputDimInfo[inChannelIdx] * stride * stride;
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: Get assit input size %d", passThroughName.c_str(), destSize);

  unique_ptr<uint16_t[]> inputAssit(new (std::nothrow) uint16_t[destSize]());
  FUSION_PASS_CHECK(inputAssit.get() == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node[%s]: inputAssit is NULL", passThroughName.c_str()),
                    return PARAM_INVALID);

  Status ret = NnSet(destSize, UINT_NUM_ZERO, *reinterpret_cast<uint16_t*>(inputAssit.get()));
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node[%s]: NnSet failed.", passThroughName.c_str()),
                    return ret);

  ret = passThroughAssistHelpFP16(destSize, *inputAssit.get(), passThroughDInputDimInfo);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node[%s]: Generate assist matrix failed.", passThroughName.c_str()),
                    return ret);

  // define the shape of auxiliary matrix
  tensorDesc.SetShape(ge::GeShape(passThroughDInputDimInfo));
  tensorDesc.SetFormat(ge::FORMAT_NCHW);
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetOriginFormat(ge::FORMAT_NCHW);
  tensorDesc.SetOriginDataType(ge::DT_FLOAT16);
  tensorDesc.SetOriginShape(ge::GeShape(passThroughDInputDimInfo));
  FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(
                               tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()), destSize * sizeof(uint16_t))),
                          assitPtr = nullptr;
                          return PARAM_INVALID);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: PassThrough has %d input.", passThroughName.c_str(),
          passThroughNode->GetInNodes().size());
  vector<ge::GeTensorPtr> weights = {assitPtr};
  ge::OpDescUtils::SetWeights(passThroughNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(passThroughNode);
  FUSION_PASS_CHECK(constInputNodes.empty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "constInputNodes is null, fusion failed."), return PARAM_INVALID);
  NodePtr constInput = constInputNodes[0];
  constInput->GetOpDesc()->SetType(CONSTANTOP);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: Success to do PassThroughFusionPass.", passThroughName.c_str());
  return SUCCESS;
}
REGISTER_PASS("PassThroughFusionPass", BUILT_IN_GRAPH_PASS, PassThroughFusionPass);
}  // namespace fe
