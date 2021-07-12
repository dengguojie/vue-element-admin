/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file conv_scale_fusion_pass.cpp
 * \brief fuse conv batchnorm, conv scale, batchnorm conv
 */
#include "conv_scale_fusion_pass.h"
#include <climits>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>
#include <string>
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "error_util.h"

namespace fe {
static const char PATTERN_SCALE[] = "scale";
static const char PATTERN_CONV[] = "conv";
static const string FILTER_HOST_TYPE = "ConvScaleFilterHost";
static const string BIAS_HOST_TYPE = "ConvScaleBiasHost";
static const int BIAS_INDEX = 2;
vector<FusionPattern*> ConvScaleFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ConvScaleFusion");
  FUSION_PASS_CHECK(pattern == nullptr, CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);

  // conv2d->scale
  pattern->AddOpDesc(PATTERN_CONV, {CONV2D, DEPTHWISECONV2D})
      .AddOpDesc(PATTERN_SCALE, {SCALE})
      .SetInputs(PATTERN_SCALE, {PATTERN_CONV})
      .SetOutput(PATTERN_SCALE);
  patterns.push_back(pattern);

  return patterns;
}

Status ConvScaleFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr convNode = GetNodeFromMapping(PATTERN_CONV, mapping);
  ge::NodePtr destNode = GetNodeFromMapping(PATTERN_SCALE, mapping);
  FUSION_PASS_CHECK(convNode == nullptr, CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "convNode is null"), return PARAM_INVALID);
  FUSION_PASS_CHECK(destNode == nullptr, CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "destNode is null"), return PARAM_INVALID);
  if (convNode->GetInDataNodes().at(1)->GetType() == QUANTWEIGHTROLLBACK) {
    return NOT_CHANGED;
  }
  if (convNode->GetOutDataNodes().size() > 1) {
    return NOT_CHANGED;
  }
  for (ge::InDataAnchorPtr inDataAnchorPtr : destNode->GetAllInDataAnchors()) {
    if (inDataAnchorPtr == nullptr || inDataAnchorPtr->GetIdx() == 0 ||
        inDataAnchorPtr->GetPeerOutAnchor() == nullptr ||
        inDataAnchorPtr->GetPeerOutAnchor()->GetOwnerNode() == nullptr) {
      continue;
    }
    ge::NodePtr nodePtr = inDataAnchorPtr->GetPeerOutAnchor()->GetOwnerNode();
    std::string type = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(nodePtr);
    FUSION_PASS_CHECK(type != CONSTANT && type != CONSTANTOP,
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "destNode has non const input, can not fusion."),
                      return NOT_CHANGED);
  }
  ge::OpDescPtr convOp = convNode->GetOpDesc();
  ge::OpDescPtr destOp = destNode->GetOpDesc();
  string convNodeName = convNode->GetName();
  string destNodeName = destNode->GetName();

  // 1. get the core's number of convNode
  int filterInputIdx = -1;
  FUSION_PASS_CHECK(GetConvFilterInputIndex(convNode, filterInputIdx) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "ConvNode[%s]: GetConvFilterInputIndex not success, no need fusion.",
                            convNode->GetName().c_str()),
                    return NOT_CHANGED);
  ge::GeTensorDesc filterInputDesc = convOp->GetInputDesc(filterInputIdx);
  ge::Format filterFormat;
  size_t kernelIndex = 0;
  size_t channelIndex = 0;
  FUSION_PASS_CHECK(GetConvKernelIndex(convOp, filterInputDesc, filterFormat, kernelIndex) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "ConvNode[%s]: GetConvKernelIndex not success, no need fusion.",
                            convNodeName.c_str()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(GetConvChannelIndex(convOp, filterInputDesc, filterFormat, channelIndex) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "ConvNode[%s]: GetConvChannelIndex not success, no need fusion.",
                            convNodeName.c_str()),
                    return NOT_CHANGED);
  int64_t kernelNum = 0;
  if (convNode->GetType() == "DepthwiseConv2D") {
    kernelNum = filterInputDesc.GetShape().GetDim(kernelIndex) * filterInputDesc.GetShape().GetDim(channelIndex);
  } else {
    kernelNum = filterInputDesc.GetShape().GetDim(kernelIndex);
  }

  bool hasBiasHost = false;
  for (ge::InDataAnchorPtr inDataAnchorPtr : convNode->GetAllInDataAnchors()) {
    if (inDataAnchorPtr == nullptr || inDataAnchorPtr->GetPeerOutAnchor() == nullptr ||
        inDataAnchorPtr->GetPeerOutAnchor()->GetOwnerNode() == nullptr) {
      continue;
    }
    ge::NodePtr nodePtr = inDataAnchorPtr->GetPeerOutAnchor()->GetOwnerNode();
    if (nodePtr->GetType() == CONVBNBIASHOST) {
      hasBiasHost = true;
    }
  }

  bool hasBias = true;
  if (convNode->GetOpDesc()->GetInputsSize() <= BIAS_INDEX && hasBiasHost == false) {
    AddBiasNode(graph, convNode);
    hasBias = false;
  }
  std::shared_ptr<ge::OpDesc> filterHostOpdesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (filterHostOpdesc =
           std::make_shared<ge::OpDesc>(convNode->GetName() + "_" + destNodeName + "_filter_host", FILTER_HOST_TYPE)),
      return NOT_CHANGED);
  std::shared_ptr<ge::OpDesc> biasHostOpdesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (biasHostOpdesc =
           std::make_shared<ge::OpDesc>(convNode->GetName() + "_" + destNodeName + "_bias_host", BIAS_HOST_TYPE)),
      return NOT_CHANGED);
  vector<ge::GeTensorDesc> conv2dInputs;
  vector<ge::InDataAnchorPtr> conv2dInputAncors;
  vector<ge::GeTensorDesc> conv2dConstOutputs;
  vector<ge::OutDataAnchorPtr> conv2dConstOutputAncors;
  vector<string> conv2dInputsName;

  ge::GeTensorDesc filterOutputDesc =
      convNode->GetInDataAnchor(filterInputIdx)->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc()->GetOutputDesc(0);
  ge::OutDataAnchorPtr filterOutputAnchor = convNode->GetInDataAnchor(filterInputIdx)->GetPeerOutAnchor();

  int biasIndex = convNode->GetOpDesc()->GetInputIndexByName("bias");
  FUSION_PASS_CHECK(biasIndex >= (int)convNode->GetAllInDataAnchorsSize(),
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                            "The index[:%d] of bias of convNode[%s] is greater than size"
                            " of input.",
                            biasIndex, convNode->GetName().c_str()),
                    return FAILED);
  ge::GeTensorDesc biasOutputDesc =
      convNode->GetInDataAnchor(biasIndex)->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc()->GetOutputDesc(0);
  ge::OutDataAnchorPtr biasOutputAnchor = convNode->GetInDataAnchor(biasIndex)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(GetAllConstInput(convNode, conv2dInputs, conv2dInputsName, conv2dInputAncors, conv2dConstOutputs,
                                     conv2dConstOutputAncors) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get const of convNode[%s] not success, fusion failed.",
                            convNode->GetName().c_str()),
                    return FAILED);

  int biasInputIndex = BIAS_INDEX;
  ge::GeTensorDesc inputDesc0 = convNode->GetOpDesc()->GetInputDesc(0);
  ge::Format inputDesc0OriginFormat = inputDesc0.GetOriginFormat();
  ge::GeShape biasShape({kernelNum});
  biasOutputDesc.SetShape(biasShape);
  biasOutputDesc.SetOriginFormat(inputDesc0OriginFormat);
  biasOutputDesc.SetOriginShape(biasShape);
  biasOutputDesc.SetOriginDataType(biasOutputDesc.GetDataType());

  //  update the bias inputDesc of the convOpDesc
  ge::GeTensorDesc biasDesc = convNode->GetOpDesc()->GetInputDesc(biasInputIndex);
  biasDesc.SetShape(biasShape);
  biasDesc.SetOriginFormat(inputDesc0OriginFormat);
  biasDesc.SetOriginShape(biasShape);
  biasDesc.SetOriginDataType(biasDesc.GetDataType());
  FUSION_PASS_CHECK(convNode->GetOpDesc()->UpdateInputDesc(BIAS_INDEX, biasDesc) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "update bias input desc of ConvNode[%s] not success.",
                            convNode->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(!conv2dInputsName.empty() && !conv2dInputsName[0].empty() && !conv2dInputs.empty() &&
                        conv2dInputs.size() > 0 &&
                        filterHostOpdesc->AddInputDesc(conv2dInputsName[0], conv2dInputs[0]) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add const input of ConvScaleFilterHost failed."), return FAILED);
  FUSION_PASS_CHECK(filterHostOpdesc->AddOutputDesc("y", filterOutputDesc) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add const output of ConvScaleFilterHost failed."), return FAILED);
  FUSION_PASS_CHECK(!conv2dInputsName.empty() && !conv2dInputsName[1].empty() && !conv2dInputs.empty() &&
                        conv2dInputs.size() > 1 &&
                        biasHostOpdesc->AddInputDesc(conv2dInputsName[1], conv2dInputs[1]) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add const input of ConvScaleBiasHost failed."), return FAILED);
  FUSION_PASS_CHECK(biasHostOpdesc->AddOutputDesc("y", biasOutputDesc) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add const output of ConvScaleBiasHost failed."), return FAILED);

  vector<ge::GeTensorDesc> destConstInputs;
  vector<ge::InDataAnchorPtr> destdInputAncors;
  vector<ge::GeTensorDesc> destConstOutputs;
  vector<ge::OutDataAnchorPtr> destConstOutputAncors;
  vector<string> descInputsName;
  FUSION_PASS_CHECK(GetAllConstInput(destNode, destConstInputs, descInputsName, destdInputAncors, destConstOutputs,
                                     destConstOutputAncors) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get const of destNode[%s] not success, fusion failed.",
                            destNode->GetName().c_str()),
                    return FAILED);
  for (size_t i = 0; i < destConstInputs.size(); i++) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "descInputsName[%d]=%s", i, descInputsName[i].c_str());
    FUSION_PASS_CHECK(filterHostOpdesc->AddInputDesc(descInputsName[i], destConstInputs[i]) != SUCCESS,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add const input of dest failed."), return FAILED);
    FUSION_PASS_CHECK(biasHostOpdesc->AddInputDesc(descInputsName[i], destConstInputs[i]) != SUCCESS,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add const input of dest failed."), return FAILED);
  }
  ge::NodePtr filterHostNode = graph.AddNode(filterHostOpdesc);
  ge::NodePtr biasHostNode = graph.AddNode(biasHostOpdesc);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(filterHostNode->GetOutDataAnchor(0),
                                            convNode->GetInDataAnchor(filterInputIdx)) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            filterHostNode->GetName().c_str(), convNode->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(biasHostNode->GetOutDataAnchor(0), convNode->GetInDataAnchor(biasIndex)) != SUCCESS,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.", biasHostNode->GetName().c_str(),
              convNode->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(biasOutputAnchor, biasHostNode->GetInDataAnchor(0)) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            biasOutputAnchor->GetOwnerNode()->GetName().c_str(), biasHostNode->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(filterOutputAnchor, filterHostNode->GetInDataAnchor(0)) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            filterOutputAnchor->GetOwnerNode()->GetName().c_str(), filterHostNode->GetName().c_str()),
                    return FAILED);

  for (size_t i = 0; i < destConstOutputAncors.size(); i++) {
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(destConstOutputAncors[i], filterHostNode->GetInDataAnchor(i + 1)) != SUCCESS,
        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                destConstOutputAncors[i]->GetOwnerNode()->GetName().c_str(), filterHostNode->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(destConstOutputAncors[i], biasHostNode->GetInDataAnchor(i + 1)) != SUCCESS,
        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                destConstOutputAncors[i]->GetOwnerNode()->GetName().c_str(), biasHostNode->GetName().c_str()),
        return FAILED);
    if (convNode->GetType() == DEPTHWISECONV2D) {
      (void)ge::AttrUtils::SetBool(filterHostNode->GetOpDesc(), IS_DEPTHWISE_CONV2D, true);
    }
    (void)ge::AttrUtils::SetInt(biasHostNode->GetOpDesc(), KERNEL_NUM, kernelNum);
    (void)ge::AttrUtils::SetBool(biasHostNode->GetOpDesc(), HAS_BIAS, hasBias);
  }
  return DoFusion(graph, convNode, destNode, fusionNodes);
}
REGISTER_PASS("ConvScaleFusionPass", BUILT_IN_GRAPH_PASS, ConvScaleFusionPass);
}  // namespace fe
