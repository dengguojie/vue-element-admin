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
 * \file space_to_depth_fusion_pass.cpp
 * \brief
 */
#include "space_to_depth_fusion_pass.h"
#include "fp16_t.hpp"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include <string>
#include <vector>

using namespace ge;
namespace fe {
static const char* SPACE_TO_DEPTH_NODE = "SpaceToDepth";
static const std::string PATTERN_FUSEDNODE = "SpaceToDepth";
static const std::string CONSTANTOP = "Constant";

static const uint16_t UINT_NUM_ZERO = 0;

template <typename Dtype>
Status spaceToDepthAssistHelpFP16(const int32_t n, Dtype& output1, const vector<int64_t> spaceToDepthDInputDimInfo) {
  OP_LOGI("spaceToDepthAssistHelpFP16", "START TO DO spaceToDepthAssistHelpFP16.");
  Dtype* output = &output1;
  fp16_t t;
  t.val = 1;
  int32_t xx = 1;
  t = xx;
  int32_t windowSize = spaceToDepthDInputDimInfo[2] * spaceToDepthDInputDimInfo[3];
  OP_LOGI("spaceToDepthAssistHelpFP16", "START TO DO spaceToDepthAssistHelpFP16 windowSize:%d.", windowSize);
  int32_t channelSize = spaceToDepthDInputDimInfo[1];
  OP_LOGI("spaceToDepthAssistHelpFP16", "START TO DO spaceToDepthAssistHelpFP16 channelSize:%d.", channelSize);
  for (int32_t windowIdx = 0; windowIdx < windowSize; windowIdx++) {
    for (int32_t channelIdx = 0; channelIdx < channelSize; channelIdx++) {
      int32_t strideIdx = channelIdx * (channelSize + 1) * windowSize;
      int32_t fillAssistIdx = strideIdx + (windowIdx * (channelSize * channelSize * windowSize + 1));
      output[fillAssistIdx] = t.val;
      OP_LOGI("spaceToDepthAssitHelpFP16", "=== Idx %d ===", fillAssistIdx);
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

vector<FusionPattern*> SpaceToDepthFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("SpaceToDepthFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {SPACE_TO_DEPTH_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status SpaceToDepthFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr spaceToDepthNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(spaceToDepthNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "spaceToDepthNode is null, fusion failed."), return PARAM_INVALID);
  ge::OpDescPtr spaceToDepthDesc = spaceToDepthNode->GetOpDesc();
  FUSION_PASS_CHECK(spaceToDepthDesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "spaceToDepthDesc is null, fusion failed."), return PARAM_INVALID);
  std::string spaceToDepthName = spaceToDepthNode->GetName();
  // get attr block_size of fused node
  int32_t blockSize = -1;
  FUSION_PASS_CHECK(
      !ge::AttrUtils::GetInt(spaceToDepthDesc, "block_size", blockSize),
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: Cannot get attr:block_size from node.", spaceToDepthName.c_str()),
      return NOT_CHANGED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: get attr:block_size from node is %d.", spaceToDepthName.c_str(), blockSize);
  FUSION_PASS_CHECK(spaceToDepthDesc->GetInputDesc(0).GetDataType() != DT_FLOAT16,
                    OP_LOGI(FUSED_OP_TYPE.c_str(),
                            "Node[%s]: The OriginDataType of node's first input is %d,"
                            "not fp16, cannot do fusion.",
                            spaceToDepthName.c_str(), spaceToDepthDesc->GetInputDesc(0).GetDataType()),
                    return NOT_CHANGED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: The OriginDataType of node's first input is %d,", spaceToDepthName.c_str(),
          spaceToDepthDesc->GetInputDesc(0).GetDataType());

  // get father node of fused node
  ge::NodePtr sapceToDepthInNode = spaceToDepthNode->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: Get father node[Name:%s Type:%s]", spaceToDepthName.c_str(),
          sapceToDepthInNode->GetName().c_str(), sapceToDepthInNode->GetType().c_str());

  ge::GeTensorDesc spaceToDepthInput = spaceToDepthDesc->GetInputDesc(0);
  ge::GeShape spaceToDepthInputShape = spaceToDepthInput.GetShape();
  vector<int64_t> spaceToDepthInputDimInfo = spaceToDepthInputShape.GetDims();
  ge::Format spaceOriginFormat = spaceToDepthInput.GetOriginFormat();
  int64_t spaceCin = 0;
  if (spaceOriginFormat == FORMAT_NHWC) {
    spaceCin = spaceToDepthInputDimInfo[3];
  } else if (spaceOriginFormat == FORMAT_NCHW) {
    spaceCin = spaceToDepthInputDimInfo[1];
  }
  if (spaceCin > 64) {
    OP_LOGI("graph not changed.");
    return NOT_CHANGED;
  }
  ge::Format assitMatrixFormat = spaceToDepthInput.GetFormat();

  size_t inChannelIdx = -1;
  FUSION_PASS_CHECK(
      SUCCESS != PatternFusionUtil::ParseChannelIdx(spaceToDepthInput, inChannelIdx),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Node[%s]: The original format of node's input0 is %s, which is unsupportable.",
              spaceToDepthName.c_str(), ge::TypeUtils::FormatToSerialString(assitMatrixFormat).c_str()),
      return FAILED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: The original format of node's input0 is %s.", spaceToDepthName.c_str(),
          ge::TypeUtils::FormatToSerialString(assitMatrixFormat).c_str());
  size_t inNumberIdx = -1;
  FUSION_PASS_CHECK(
      SUCCESS != ParseNumberIdx(spaceToDepthInput, inNumberIdx),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Node[%s]: The original format of node's input0 is %s, which is unsupportable.",
              spaceToDepthName.c_str(), ge::TypeUtils::FormatToSerialString(assitMatrixFormat).c_str()),
      return FAILED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: The original format of node's input0 is %s.", spaceToDepthName.c_str(),
          ge::TypeUtils::FormatToSerialString(assitMatrixFormat).c_str());
  int64_t spaceToDepthDChannel = spaceToDepthInputDimInfo[inChannelIdx] * blockSize * blockSize;
  vector<int64_t> spaceToDepthDInputDimInfo = {spaceToDepthDChannel, spaceToDepthInputDimInfo[inChannelIdx], blockSize,
                                               blockSize};
  for (auto printInfo : spaceToDepthDInputDimInfo) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: info of node's input1 is %d.", spaceToDepthName.c_str(), printInfo);
  }
  ge::GeTensorPtr assitPtr = nullptr;
  ge::GeTensorDesc tensorDesc;
  int64_t dimSize = spaceToDepthInputDimInfo[inNumberIdx] * spaceToDepthInputDimInfo[inChannelIdx];
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: Get first input size %d", spaceToDepthName.c_str(),
          spaceToDepthInputDimInfo[inNumberIdx]);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: Get first input size %d", spaceToDepthName.c_str(),
          spaceToDepthInputDimInfo[inChannelIdx]);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: Get first input size %d", spaceToDepthName.c_str(), dimSize);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: Get first input size %d", spaceToDepthName.c_str(), spaceToDepthDChannel);
  int64_t destSize = spaceToDepthInputDimInfo[inChannelIdx] * spaceToDepthDChannel * blockSize * blockSize;
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: Get assit input size %d", spaceToDepthName.c_str(), destSize);

  unique_ptr<uint16_t[]> inputAssit(new (std::nothrow) uint16_t[destSize]());
  FUSION_PASS_CHECK(inputAssit.get() == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Node[%s]: inputAssit is NULL", spaceToDepthName.c_str()),
                    return PARAM_INVALID);

  Status ret = NnSet(destSize, UINT_NUM_ZERO, *reinterpret_cast<uint16_t*>(inputAssit.get()));
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Node[%s]: NnSet failed.", spaceToDepthName.c_str()),
                    return ret);

  ret = spaceToDepthAssistHelpFP16(destSize, *inputAssit.get(), spaceToDepthDInputDimInfo);
  FUSION_PASS_CHECK(
      ret != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Node[%s]: Generate assist matrix failed.", spaceToDepthName.c_str()), return ret);

  // define the shape of auxiliary matrix
  tensorDesc.SetShape(ge::GeShape(spaceToDepthDInputDimInfo));
  tensorDesc.SetFormat(ge::FORMAT_NCHW);
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetOriginFormat(ge::FORMAT_NCHW);
  tensorDesc.SetOriginDataType(ge::DT_FLOAT16);
  tensorDesc.SetOriginShape(ge::GeShape(spaceToDepthDInputDimInfo));
  FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(
                               tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()), destSize * sizeof(uint16_t))),
                          assitPtr = nullptr;
                          return PARAM_INVALID);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: SpaceToDepth has %d input.", spaceToDepthName.c_str(),
          spaceToDepthNode->GetInNodes().size());
  vector<ge::GeTensorPtr> weights = {assitPtr};
  ge::OpDescUtils::SetWeights(spaceToDepthNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(spaceToDepthNode);
  NodePtr constInput = constInputNodes[0];
  constInput->GetOpDesc()->SetType(CONSTANTOP);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]: Success to do SpaceToDepthFusionPass.", spaceToDepthName.c_str());
  return SUCCESS;
}
REGISTER_PASS("SpaceToDepthFusionPass", BUILT_IN_GRAPH_PASS, SpaceToDepthFusionPass);
}  // namespace fe
