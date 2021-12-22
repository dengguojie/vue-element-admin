/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
 * \file softmax_transpose_fusion_pass.cpp
 * \brief instance norm fusion pass(instance norm --> pure instance norm)
 */
#include "softmax_transpose_fusion_pass.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <algorithm>

#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;
namespace fe {
static const string PATTERN_Softmax = "SoftmaxV2";
static const string SOFTMAX = "SoftmaxV2";
static const string AXIS = "axes";

vector<FusionPattern*> softmaxTransFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("softmaxTransFusionPass");
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter softmaxTransFusionPass::DefinePatterns.");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new an object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_Softmax, {SOFTMAX}).SetOutput(PATTERN_Softmax);
  patterns.push_back(pattern);

  return patterns;
}

Status softmaxTransFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter GoSoftmaxV2");
  ge::NodePtr inNode = GetNodeFromMapping(PATTERN_Softmax, mapping);
  FUSION_PASS_CHECK(inNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Node SoftmaxV2 is null, fusion failed."),
                    return PARAM_INVALID);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "check SoftmaxV2");
  FUSION_PASS_CHECK(CheckParameter(inNode) != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "Check SoftmaxV2 param failed."),
                    return NOT_CHANGED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "fusion SoftmaxV2");
  return INFuison(graph, inNode, newNodes);
}

Status softmaxTransFusionPass::SetAttrValue(const ge::OpDescPtr& OpDescPtr, int64_t shapeSize, int32_t transfer) {
  // transfer the total image axis -2 and axis -1 for this case
  if (transfer == 1) {
    vector<int32_t> permValue;
    for (int32_t i = 0; i < shapeSize; i++) {
      permValue.push_back(i);
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Set SetAttrValue %d", i);
    }

    // reversed aixs -2 and -1
    int temp = permValue.back();
    permValue[shapeSize - 1] = permValue[shapeSize - 2];
    permValue[shapeSize - 2] = temp;

    ge::AttrUtils::SetListInt(OpDescPtr, "perm", permValue);
    return SUCCESS;
  }

  return FAILED;
}

Status softmaxTransFusionPass::CheckParameter(ge::NodePtr& inNodePtr) {
  // get psroipooling node inputs.
  Node::Vistor<NodePtr> inNodes = inNodePtr->GetInDataNodes();
  FUSION_PASS_CHECK((inNodes.size() != 1),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "the input data size num(%lu) != 1",
                                                   inNodes.size()),
                    return PARAM_INVALID);
  return SUCCESS;
}

Status softmaxTransFusionPass::SetAttrValueForNewNode(const ge::OpDescPtr& preOpDescPtr, ge::OpDescPtr& newOpDescPtr,
                                                      int64_t shapeSize) {
  vector<int32_t> axisValue;
  ge::AttrUtils::GetListInt(preOpDescPtr, AXIS, axisValue);

  // change softmax axis at axis == [-1,lens-1, -2 ,lens-2]
  for (uint32_t i = 0; i < axisValue.size(); i++) {
    if (axisValue[i] == -1 || axisValue[i] == (shapeSize - 1)) {
      axisValue[i] = shapeSize - 2;
    } else if (axisValue[i] == -2 || axisValue[i] == (shapeSize - 2)) {
      axisValue[i] = shapeSize - 1;
    }
  }

  ge::AttrUtils::SetListInt(newOpDescPtr, AXIS, axisValue);

  return SUCCESS;
}

bool softmaxTransFusionPass::CheckStatus(ge::OpDescPtr& inOpDescPtr, vector<int64_t> inputShape, uint32_t lastDimsVal,
                                         uint32_t times, uint32_t maxDimsVal) {
  uint32_t shapeSize = inputShape.size();
  vector<int32_t> axisValue;
  vector<int32_t>::iterator iterNegAxis, iterPosAxis;
  ge::AttrUtils::GetListInt(inOpDescPtr, AXIS, axisValue);
  iterNegAxis = find(axisValue.begin(), axisValue.end(), -1);
  iterPosAxis = find(axisValue.begin(), axisValue.end(), shapeSize - 1);
  // The penultimate axis must be more than 10 times the last axis in 4 hd and more than 25 times the last axis in 5 hd
  return shapeSize > 1 && (iterNegAxis != axisValue.end() || iterPosAxis != axisValue.end()) &&
         inputShape.back() < lastDimsVal && inputShape[shapeSize - 1] * times < inputShape[shapeSize - 2] &&
         inputShape[shapeSize - 2] < maxDimsVal;
}

Status softmaxTransFusionPass::INFuison(ge::ComputeGraph& graph, ge::NodePtr& inNodePtr,
                                        vector<ge::NodePtr>& newNodes) {
  ge::OpDescPtr inOpDescPtr = inNodePtr->GetOpDesc();
  FUSION_PASS_CHECK(
      inOpDescPtr == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.",
                                     inOpDescPtr->GetName().c_str()),
      return PARAM_INVALID);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "NODE %s", inOpDescPtr->GetName().c_str());

  ge::GeTensorDesc xInputDesc = inOpDescPtr->GetInputDesc(0);
  vector<int64_t> inputShape = xInputDesc.GetShape().GetDims();
  vector<int64_t> inputOrishape = xInputDesc.GetOriginShape().GetDims();
  ge::Format inputFormat = xInputDesc.GetFormat();
  uint32_t shapeLens = inputShape.size();
  uint32_t oriShapelens = inputOrishape.size();

  FUSION_PASS_CHECK(
      inputShape.empty(),
      OP_LOGW(FUSED_OP_TYPE.c_str(), "Node:%s's input shape is null, fusion failed.", inOpDescPtr->GetName().c_str()),
      return NOT_CHANGED);

  FUSION_PASS_CHECK(inputOrishape.empty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Node:%s's input ori shape is null, fusion failed.",
                            inOpDescPtr->GetName().c_str()),
                    return PARAM_INVALID);
  vector<int64_t> transposeShape;
  uint32_t shapeSize = 0;

  if (inputFormat == ge::FORMAT_NC1HWC0){
      shapeSize = oriShapelens;
      transposeShape = inputOrishape;
  } else{
      shapeSize = shapeLens;
      transposeShape = inputShape;
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "input format is %d", inputFormat);

  if (shapeSize < 2) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "shape_len is less then 2");
    return NOT_CHANGED;
  }
  for (uint64_t i = 0; i < oriShapelens; i++) {
      if (PatternFusionUtil::IsUnknownShape(inputOrishape[i])){
          OP_LOGI(FUSED_OP_TYPE.c_str(), "NODE %s is unknownshape, not changed", inOpDescPtr->GetName().c_str());
          return NOT_CHANGED;
      }
    }

  // normal check
  bool shapeCheck = CheckStatus(inOpDescPtr, inputShape, 16, 10, 600000);

  bool shapeOriCheck = CheckStatus(inOpDescPtr, inputOrishape, 16, 25, 600000);

  if ((shapeCheck && inputFormat != ge::FORMAT_NC1HWC0) || (shapeOriCheck and inputFormat == ge::FORMAT_NC1HWC0)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "begin softmax transpose fusion pass");
    // create softmax opdesc
    std::shared_ptr<ge::OpDesc> SoftmaxV2OpDescPtr = nullptr;
    FUSION_PASS_MAKE_SHARED(
        (SoftmaxV2OpDescPtr = std::make_shared<ge::OpDesc>(inOpDescPtr->GetName() + "_new", "SoftmaxV2")),
        return NOT_CHANGED);

    FUSION_PASS_CHECK(SetAttrValueForNewNode(inOpDescPtr, SoftmaxV2OpDescPtr, shapeSize) != SUCCESS,
                      OP_LOGW(FUSED_OP_TYPE.c_str(), "Update softmax attr failed."), return NOT_CHANGED);

    // create transpose opdesc
    std::shared_ptr<ge::OpDesc> transposeOpDescPtr = nullptr;
    FUSION_PASS_MAKE_SHARED(
        (transposeOpDescPtr = std::make_shared<ge::OpDesc>(inOpDescPtr->GetName() + "_input", "TransposeD")),
        return NOT_CHANGED);

    std::shared_ptr<ge::OpDesc> transposeOutOpDescPtr = nullptr;
    FUSION_PASS_MAKE_SHARED(
        (transposeOutOpDescPtr = std::make_shared<ge::OpDesc>(inOpDescPtr->GetName() + "_out", "TransposeD")),
        return NOT_CHANGED);

    FUSION_PASS_CHECK(SetAttrValue(transposeOpDescPtr, shapeSize, 1) != SUCCESS,
                      OP_LOGW(FUSED_OP_TYPE.c_str(), "set transpose perm failed."), return NOT_CHANGED);
    FUSION_PASS_CHECK(SetAttrValue(transposeOutOpDescPtr, shapeSize, 1) != SUCCESS,
                      OP_LOGW(FUSED_OP_TYPE.c_str(), "set transpose perm failed."), return NOT_CHANGED);

    // get transpose input
    ge::GeTensorDesc tpsXInputTensorDesc = inOpDescPtr->GetInputDesc(0);
    if (inputFormat == ge::FORMAT_NC1HWC0){
        tpsXInputTensorDesc.SetShape(xInputDesc.GetOriginShape());
        tpsXInputTensorDesc.SetOriginShape(xInputDesc.GetOriginShape());
        tpsXInputTensorDesc.SetFormat(FORMAT_ND);
        tpsXInputTensorDesc.SetOriginFormat(FORMAT_ND);
    }
    // fill transpose output TensorDesc
    vector<int64_t> outShape;
    for (uint32_t i = 0; i < shapeSize; i++) {
      outShape.push_back(transposeShape[i]);
    }

    int64_t tempSize = outShape.back();
    outShape[shapeSize - 1] = outShape[shapeSize - 2];
    outShape[shapeSize - 2] = tempSize;

    GeShape out_shape(outShape);

    ge::GeTensorDesc tpsyOutputTensorDesc;
    tpsyOutputTensorDesc = inOpDescPtr->GetOutputDesc(0);
    tpsyOutputTensorDesc.SetShape(out_shape);
    tpsyOutputTensorDesc.SetOriginShape(out_shape);
    tpsyOutputTensorDesc.SetFormat(FORMAT_ND);
    tpsyOutputTensorDesc.SetOriginFormat(FORMAT_ND);

    OP_LOGI(FUSED_OP_TYPE.c_str(), "Set tpsyOutputTensorDesc shape Done");

    ge::GeTensorDesc tpsOutyOutputTensorDesc = inOpDescPtr->GetInputDesc(0);
    if (inputFormat == ge::FORMAT_NC1HWC0){
        tpsOutyOutputTensorDesc.SetShape(xInputDesc.GetOriginShape());
        tpsOutyOutputTensorDesc.SetOriginShape(xInputDesc.GetOriginShape());
        tpsOutyOutputTensorDesc.SetFormat(FORMAT_ND);
        tpsOutyOutputTensorDesc.SetOriginFormat(FORMAT_ND);
    }
    // get SoftmaxV2 output

    ge::GeTensorDesc yOutputTensorDesc;
    yOutputTensorDesc = inOpDescPtr->GetOutputDesc(0);
    yOutputTensorDesc.SetShape(out_shape);
    yOutputTensorDesc.SetOriginShape(out_shape);
    yOutputTensorDesc.SetFormat(FORMAT_ND);
    yOutputTensorDesc.SetOriginFormat(FORMAT_ND);

    transposeOpDescPtr->AddInputDesc("x", tpsXInputTensorDesc);
    transposeOpDescPtr->AddOutputDesc("y", tpsyOutputTensorDesc);

    SoftmaxV2OpDescPtr->AddInputDesc("x", tpsyOutputTensorDesc);
    SoftmaxV2OpDescPtr->AddOutputDesc("y", yOutputTensorDesc);

    transposeOutOpDescPtr->AddInputDesc("x", yOutputTensorDesc);
    transposeOutOpDescPtr->AddOutputDesc("y", tpsOutyOutputTensorDesc);

    OP_LOGI(FUSED_OP_TYPE.c_str(), "Set SoftmaxV2OpDescPtr connect Done");

    // add tranposes and softmaxv2 node to graph
    ge::NodePtr transposeNodePtr = graph.AddNode(transposeOpDescPtr);
    ge::NodePtr SoftmaxV2NodePtr = graph.AddNode(SoftmaxV2OpDescPtr);
    ge::NodePtr transposeOutNodePtr = graph.AddNode(transposeOutOpDescPtr);
    newNodes.push_back(transposeNodePtr);
    newNodes.push_back(SoftmaxV2NodePtr);
    newNodes.push_back(transposeOutNodePtr);

    FUSION_PASS_CHECK(transposeNodePtr == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "fusionNode: transposeNodePtr is null, fusion failed."),
                      return FAILED);
    FUSION_PASS_CHECK(SoftmaxV2NodePtr == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "fusionNode: SoftmaxV2NodePtr is null, fusion failed."),
                      return FAILED);
    FUSION_PASS_CHECK(transposeOutNodePtr == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "fusionNode: transposeOutNodePtr is null, fusion failed."),
                      return FAILED);

    // add the edge
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(inNodePtr->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                         transposeNodePtr->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Add edge from data node:%s to transpose node:%s failed.",
                              inNodePtr->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                              transposeNodePtr->GetName().c_str()),
                      return FAILED);

    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(transposeNodePtr->GetOutAnchor(0), SoftmaxV2NodePtr->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "Add edge from outputedge node:%s to transpose node:%s failed.",
                SoftmaxV2NodePtr->GetName().c_str(), transposeNodePtr->GetName().c_str()),
        return FAILED);

    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(SoftmaxV2NodePtr->GetOutAnchor(0), transposeOutNodePtr->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                       "Add edge from outputedge node:%s to transpose node:%s failed.",
                SoftmaxV2NodePtr->GetName().c_str(), transposeOutNodePtr->GetName().c_str()),
        return FAILED);

    // add the output of transpose edge
    size_t outanchorsize = inNodePtr->GetAllOutDataAnchors().size();
    for (size_t outindex = 0; outindex < outanchorsize; outindex++) {
      for (auto inDataAnchor : inNodePtr->GetOutDataAnchor(outindex)->GetPeerInDataAnchors()) {
        FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(inNodePtr->GetOutDataAnchor(outindex), inDataAnchor) != SUCCESS,
                          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                         "Remove SoftmaxV2 out data edge failed."),
                                                         return FAILED);
        FUSION_PASS_CHECK(
            ge::GraphUtils::AddEdge(transposeOutNodePtr->GetOutDataAnchor(outindex), inDataAnchor) != SUCCESS,
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add SoftmaxV2 out data edge failed."),
                                           return FAILED);
      }
    }

    // remove Normalize from graph
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(inNodePtr),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove inNodePtr node[%s] failed",
                                                     inNodePtr->GetName().c_str()),
                      return FAILED);
    return SUCCESS;
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "not change");
    return NOT_CHANGED;
  }
}

REGISTER_PASS("softmaxTransFusionPass", BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS, softmaxTransFusionPass);
}  // namespace fe
