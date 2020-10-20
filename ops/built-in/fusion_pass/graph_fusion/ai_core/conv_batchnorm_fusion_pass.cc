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
 * \file conv_batchnorm_fusion_pass.cpp
 * \brief fuse conv batchnorm, conv scale, batchnorm conv
 */
#include "conv_batchnorm_fusion_pass.h"
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

namespace fe {
static const char PATTERN_CONV[] = "conv";
static const char PATTERN_BATCHNORM[] = "batchnorm";
static const char PATTERN_STREAM_SWITCH[] = "streamSwitch";
static const string FILTER_HOST_TYPE = "ConvBnFilterHost";
static const string BIAS_HOST_TYPE = "ConvBnBiasHost";
static const int BIAS_INDEX = 2;
vector<FusionPattern*> ConvBatchnormFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ConvBatchnomFusion");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object not success."), return patterns);

  /*
          Data(input)
              \
               \
                v
    Const(filter)--->Conv2d----->BNInferenceD(cafe)------>output
               ^                  ^       ^
              /                  /       /
             /                  /       /
          Const(bias)         mean  variance
  */
  /*
                 Data(input)
                \
                 \
                  v
      Const(filter)--->Conv2d--->BatchNorm(tensorflow)--->output
                 ^                ^      ^      ^    ^
                /                /      /      /    /
               /                /      /      /    /
            Const(bias)    scale offset mean variance
   */
  pattern->AddOpDesc(PATTERN_CONV, {CONV2D, CONV3D, DEPTHWISECONV2D})
      .AddOpDesc(PATTERN_BATCHNORM, {BATCHNORM, BN_INFERENCE_D})
      .SetInputs(PATTERN_BATCHNORM, {PATTERN_CONV})
      .SetOutput(PATTERN_BATCHNORM);
  patterns.push_back(pattern);
  FusionPattern* pattern1 = new (std::nothrow) FusionPattern("ConvBatchnomFusion1");
  FUSION_PASS_CHECK(
      pattern1 == nullptr, patterns.clear(); if (pattern != nullptr) {
        delete pattern;
        pattern = nullptr;
      } OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object not success."),
                                             return patterns);

  // conv2d+switch->batchnorm
  pattern1->AddOpDesc(PATTERN_CONV, {CONV2D, CONV3D})
      .AddOpDesc(PATTERN_BATCHNORM, {BATCHNORM, BN_INFERENCE_D})
      .AddOpDesc(PATTERN_STREAM_SWITCH, {STREAMSWITCH})
      .SetInputs(PATTERN_BATCHNORM, {PATTERN_STREAM_SWITCH, PATTERN_CONV})
      .SetOutput(PATTERN_BATCHNORM);
  patterns.push_back(pattern1);

  return patterns;
}

Status ConvBatchnormFusionPass::CheckWeights(const ge::NodePtr bnNode) {
  ge::OpDescPtr destOpDesc = bnNode->GetOpDesc();
  FUSION_PASS_CHECK(destOpDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "destOpDesc is null."), return PARAM_INVALID);
  string bnNodeName = destOpDesc->GetName();
  string bnOpType = destOpDesc->GetType();
  // 1. get BatchNormOpParams's WeightDef
  vector<ge::ConstGeTensorPtr> weights = ge::OpDescUtils::GetWeights(bnNode);
  size_t btSize = weights.size();
  if (!weights.empty() && bnOpType == BATCHNORM) {
    // tensorflow: bnScale/bnBias/mean/variance must be required for infer
    FUSION_PASS_CHECK(btSize != BATCHNORM_MAXIMUM_WEIGHT_SIZE,
                      OP_LOGW(FUSED_OP_TYPE.c_str(), "Node[%s]: batch normal weights size %d must be ==4.",
                              bnNodeName.c_str(), btSize),
                      return NOT_CHANGED);
  }
  return SUCCESS;
}

Status ConvBatchnormFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr convNode = GetNodeFromMapping(PATTERN_CONV, mapping);
  ge::NodePtr destNode = GetNodeFromMapping(PATTERN_BATCHNORM, mapping);
  FUSION_PASS_CHECK(convNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new convNode not success."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(destNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new destNode not success."),
                    return PARAM_INVALID);

  FUSION_PASS_CHECK(CheckWeights(destNode) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "BN node can not fusion, exit ConvBatchnormFusionPass."),
                    return NOT_CHANGED);
  if (convNode->GetOutDataNodes().size() > 1) {
    return SUCCESS;
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
  string convNodeName = convNode->GetName();
  Status ret = PatternFusionUtil::LinkControlEdge(destNode, convNode);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "ConvNode[%s]: LinkControlEdge not success.", convNodeName.c_str()),
                    return ret);

  // 1. get the filterFormat and kernelNum
  int filterInputIndex = -1;
  FUSION_PASS_CHECK(
      GetConvFilterInputIndex(convNode, filterInputIndex) != SUCCESS,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "ConvNode[%s]: GetConvFilterInputIndex not success.", convNodeName.c_str()),
      return NOT_CHANGED);
  ge::GeTensorDesc filterInputDesc = convNode->GetOpDesc()->GetInputDesc(filterInputIndex);
  size_t kernerlIndex = 0;
  ge::Format filterFormat;
  FUSION_PASS_CHECK(GetConvKernelIndex(convNode->GetOpDesc(), filterInputDesc, filterFormat, kernerlIndex) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(),
                            "ConvNode[%s]: GetConvKernelIndex not succeess, "
                            "filterInputIndex=[%d].",
                            convNodeName.c_str(), filterInputIndex),
                    return NOT_CHANGED);
  ge::GeShape filterShape = filterInputDesc.GetShape();
  int64_t kernelNum = filterShape.GetDim(kernerlIndex);

  // create host op
  bool hasBias = true;
  if (convNode->GetOpDesc()->GetInputsSize() <= BIAS_INDEX) {
    AddBiasNode(graph, convNode);
    hasBias = false;
  }
  std::shared_ptr<ge::OpDesc> filterHostOpdesc = nullptr;
  filterHostOpdesc =
      std::make_shared<ge::OpDesc>(convNode->GetName() + destNode->GetName() + "_filter_host", FILTER_HOST_TYPE);
  std::shared_ptr<ge::OpDesc> biasHostOpdesc =
      std::make_shared<ge::OpDesc>(convNode->GetName() + destNode->GetName() + "_bias_host", BIAS_HOST_TYPE);
  vector<ge::GeTensorDesc> conv2dInputs;
  vector<ge::InDataAnchorPtr> conv2dInputAncors;
  vector<ge::GeTensorDesc> conv2dConstOutputs;
  vector<ge::OutDataAnchorPtr> conv2dConstOutputAncors;
  vector<string> conv2dInputsName;
  ge::GeTensorDesc filterOutputDesc =
      convNode->GetInDataAnchor(filterInputIndex)->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc()->GetOutputDesc(0);
  ge::OutDataAnchorPtr filterOutputAnchor = convNode->GetInDataAnchor(filterInputIndex)->GetPeerOutAnchor();

  int biasIndex = convNode->GetOpDesc()->GetInputIndexByName("bias");
  FUSION_PASS_CHECK(biasIndex >= (int)convNode->GetAllInDataAnchorsSize(),
                    OP_LOGE(FUSED_OP_TYPE.c_str(),
                            "The index[:%d] of bias of convNode[%s] is greater than size"
                            " of input.",
                            biasIndex, convNode->GetName().c_str()),
                    return FAILED);
  ge::GeTensorDesc biasOutputDesc =
      convNode->GetInDataAnchor(biasIndex)->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc()->GetOutputDesc(0);
  ge::OutDataAnchorPtr biasOutputAnchor = convNode->GetInDataAnchor(biasIndex)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(GetAllConstInput(convNode, conv2dInputs, conv2dInputsName, conv2dInputAncors, conv2dConstOutputs,
                                     conv2dConstOutputAncors) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Get const of convNode[%s] not success, fusion failed.",
                            convNode->GetName().c_str()),
                    return FAILED);
  int biasInputIndex = 2;
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
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "update bias input desc of ConvNode[%s] not success.",
                            convNode->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(!conv2dInputsName.empty() && !conv2dInputsName[0].empty() && !conv2dInputs.empty() &&
                        conv2dInputs.size() > 0 &&
                        filterHostOpdesc->AddInputDesc(conv2dInputsName[0], conv2dInputs[0]) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add const input of ConvBnFilterHost failed."), return FAILED);
  FUSION_PASS_CHECK(filterHostOpdesc->AddOutputDesc("y", filterOutputDesc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add const output of ConvBnFilterHost failed."), return FAILED);
  FUSION_PASS_CHECK(!conv2dInputsName.empty() && !conv2dInputsName[1].empty() && !conv2dInputs.empty() &&
                        conv2dInputs.size() > 1 &&
                        biasHostOpdesc->AddInputDesc(conv2dInputsName[1], conv2dInputs[1]) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add const input of ConvBnBiasHost failed."), return FAILED);
  FUSION_PASS_CHECK(biasHostOpdesc->AddOutputDesc("y", biasOutputDesc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "add const output of ConvBnBiasHost failed."), return FAILED);
  vector<ge::GeTensorDesc> destConstInputs;
  vector<ge::InDataAnchorPtr> destdInputAncors;
  vector<ge::GeTensorDesc> destConstOutputs;
  vector<ge::OutDataAnchorPtr> destConstOutputAncors;
  vector<string> descInputsName;
  FUSION_PASS_CHECK(GetAllConstInput(destNode, destConstInputs, descInputsName, destdInputAncors, destConstOutputs,
                                     destConstOutputAncors) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Get const of destNode[%s] not success, fusion failed.",
                            destNode->GetName().c_str()),
                    return FAILED);
  for (size_t i = 0; i < destConstInputs.size(); i++) {
    FUSION_PASS_CHECK(filterHostOpdesc->AddInputDesc(descInputsName[i], destConstInputs[i]) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add const input of dest failed."), return FAILED);
    FUSION_PASS_CHECK(biasHostOpdesc->AddInputDesc(descInputsName[i], destConstInputs[i]) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add const input of dest failed."), return FAILED);
  }
  ge::NodePtr filterHostNode = graph.AddNode(filterHostOpdesc);
  ge::NodePtr biasHostNode = graph.AddNode(biasHostOpdesc);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(filterHostNode->GetOutDataAnchor(0),
                                            convNode->GetInDataAnchor(filterInputIndex)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            filterHostNode->GetName().c_str(), convNode->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(biasHostNode->GetOutDataAnchor(0), convNode->GetInDataAnchor(biasIndex)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.", biasHostNode->GetName().c_str(),
              convNode->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(biasOutputAnchor, biasHostNode->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            biasOutputAnchor->GetOwnerNode()->GetName().c_str(), biasHostNode->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(filterOutputAnchor, filterHostNode->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                            filterOutputAnchor->GetOwnerNode()->GetName().c_str(), filterHostNode->GetName().c_str()),
                    return FAILED);
  for (size_t i = 0; i < destConstOutputAncors.size(); i++) {
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(destConstOutputAncors[i], filterHostNode->GetInDataAnchor(i + 1)) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                destConstOutputAncors[i]->GetOwnerNode()->GetName().c_str(), filterHostNode->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(destConstOutputAncors[i], biasHostNode->GetInDataAnchor(i + 1)) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                destConstOutputAncors[i]->GetOwnerNode()->GetName().c_str(), biasHostNode->GetName().c_str()),
        return FAILED);
  }
  float eps = 0.0;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetFloat(destNode->GetOpDesc(), ge::BATCHNORM_ATTR_EPSILON, eps),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "DestNode[%s]: get epsilon attr %s not success.",
                            destNode->GetName().c_str(), ge::BATCHNORM_ATTR_EPSILON.c_str()),
                    return FAILED);
  if (eps < EPS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "DestNode[%s]: the epsilon is less than %f, set the default value %f.",
            destNode->GetName().c_str(), EPS, EPS_DEFAULT_FLOAT);
    eps = EPS_DEFAULT_FLOAT;
  }
  (void)ge::AttrUtils::SetFloat(filterHostNode->GetOpDesc(), ge::BATCHNORM_ATTR_EPSILON, eps);
  (void)ge::AttrUtils::SetFloat(biasHostNode->GetOpDesc(), ge::BATCHNORM_ATTR_EPSILON, eps);
  if (convNode->GetType() == DEPTHWISECONV2D) {
    (void)ge::AttrUtils::SetBool(filterHostNode->GetOpDesc(), IS_DEPTHWISE_CONV2D, true);
  }
  (void)ge::AttrUtils::SetInt(biasHostNode->GetOpDesc(), KERNEL_NUM, kernelNum);
  (void)ge::AttrUtils::SetBool(biasHostNode->GetOpDesc(), HAS_BIAS, hasBias);
  bool needAddAndRsqrt = true;
  /* In Default mode, the add and rsqrt will be calculated in
   * conv+bn fusion */
  if (!ge::AttrUtils::GetBool(destNode->GetOpDesc(), NEED_ADD_AND_RSQRT, needAddAndRsqrt)) {
    needAddAndRsqrt = true;
  }

  /* here else is ommited because the needAddAndRsqrt is got from the func
   * GetBool */
  (void)ge::AttrUtils::SetBool(filterHostNode->GetOpDesc(), NEED_ADD_AND_RSQRT, needAddAndRsqrt);
  (void)ge::AttrUtils::SetBool(biasHostNode->GetOpDesc(), NEED_ADD_AND_RSQRT, needAddAndRsqrt);
  // fuse conv and batchnorm
  return DoFusion(graph, convNode, destNode, fusionNodes);
}
REGISTER_PASS("ConvBatchnormFusionPass", BUILT_IN_GRAPH_PASS, ConvBatchnormFusionPass);
}  // namespace fe
