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
 * \file batchnorm_fusion_pass.cpp
 * \brief Fused Add, Mul(three), Sub of structure:
 *           const     const
 *               \    /
 *                Mul1  const
 *              /   \  /
 *  Conv3d    /     Mul2  const
 *      \   /        |  /
 *       Mul3        Sub
 *         \       /
 *          \    /
 *           Add
 *
 *          or :
 *             const(variance)  const(eps)
 *                      \     /
 *                       Add
 *                        |
 *                      Rsqrt
 *                       |
 *          const       /
 *               \    /
 *                Mul1  const
 *              /   \  /
 *  Conv3d    /     Mul2  const
 *      \   /        |  /
 *       Mul3       Sub
 *         \       /
 *          \    /
 *           Add
 * into batch norm op fusion pass
 *
 */
#include "batchnorm_fusion_pass.h"
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

namespace fe {
static const string ADD = "Add";
static const string MUL = "Mul";
static const string SUB = "Sub";
static const string RSQRT = "Rsqrt";
static const string CONST = "Const";
static const string ADD_PATTERN = "add";
static const string MUL_PATTERN_1 = "mul1";
static const string MUL_PATTERN_2 = "mul2";
static const string MUL_PATTERN_3 = "mul3";
static const string SUB_PATTERN = "sub";
static const string RSQRT_PATTERN = "rsqrt";
static const string CONST_PATTERN = "const_pattern";
static const string ADD_EPS_PATTERN = "add_eps";
static const string CONV3D_PATTERN = "conv3d_pattern";
static const string CONV3D = "Conv3D";
static const int32_t NDHWC_DIM_C = 4;
static const int32_t NCDHW_DIM_C = 1;
static const string BATCHNORM = "BatchNorm";

vector<FusionPattern*> BatchnormFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern1 = new (std::nothrow) FusionPattern("BatchnormFusion1");
  FUSION_PASS_CHECK(pattern1 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  pattern1->AddOpDesc(ADD_PATTERN, {ADD})
      .AddOpDesc(MUL_PATTERN_1, {MUL})
      .AddOpDesc(MUL_PATTERN_2, {MUL})
      .AddOpDesc(MUL_PATTERN_3, {MUL})
      .AddOpDesc(SUB_PATTERN, {SUB})
      .AddOpDesc(CONST_PATTERN, {CONST})
      .AddOpDesc(CONV3D_PATTERN, {CONV3D})
      .SetInputs(MUL_PATTERN_1, {CONST_PATTERN, CONST_PATTERN})
      .SetInputs(MUL_PATTERN_2, {MUL_PATTERN_1})
      .SetInputs(SUB_PATTERN, {MUL_PATTERN_2})
      .SetInputs(MUL_PATTERN_3, {CONV3D_PATTERN, MUL_PATTERN_1})
      .SetInputs(ADD_PATTERN, {MUL_PATTERN_3, SUB_PATTERN})
      .SetOutput(ADD_PATTERN);

  patterns.push_back(pattern1);

  FusionPattern* pattern2 = new (std::nothrow) FusionPattern("BatchnormFusion2");
  FUSION_PASS_CHECK(pattern2 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new an object failed."), return patterns);
  pattern2->AddOpDesc(ADD_PATTERN, {ADD})
      .AddOpDesc(MUL_PATTERN_1, {MUL})
      .AddOpDesc(MUL_PATTERN_2, {MUL})
      .AddOpDesc(MUL_PATTERN_3, {MUL})
      .AddOpDesc(SUB_PATTERN, {SUB})
      .AddOpDesc(RSQRT_PATTERN, {RSQRT})
      .AddOpDesc(ADD_EPS_PATTERN, {ADD})
      .AddOpDesc(CONV3D_PATTERN, {CONV3D})
      .AddOpDesc(CONST_PATTERN, {CONST})
      .SetInputs(ADD_EPS_PATTERN, {CONST_PATTERN, CONST_PATTERN})
      .SetInputs(RSQRT_PATTERN, {ADD_EPS_PATTERN})
      .SetInputs(MUL_PATTERN_1, {RSQRT_PATTERN})
      .SetInputs(MUL_PATTERN_2, {MUL_PATTERN_1})
      .SetInputs(SUB_PATTERN, {MUL_PATTERN_2})
      .SetInputs(MUL_PATTERN_3, {CONV3D_PATTERN, MUL_PATTERN_1})
      .SetInputs(ADD_PATTERN, {MUL_PATTERN_3, SUB_PATTERN})
      .SetOutput(ADD_PATTERN);
  patterns.push_back(pattern2);

  return patterns;
}

int64_t BatchnormFusionPass::GetKernelNumOfOutputOfConv3D(const ge::NodePtr& conv) {
  auto outputDesc = conv->GetOpDesc()->GetOutputDesc(0);
  auto dims = outputDesc.GetShape().GetDims();
  if (dims.size() < 5) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dimNum of %s which is %ld is less than 5", conv->GetName().c_str(), dims.size());
    return 0;
  }

  ge::Format curFormat = outputDesc.GetOriginFormat();
  if (curFormat == ge::FORMAT_NDHWC) {
    return dims.at(NDHWC_DIM_C);
  } else if (curFormat == ge::FORMAT_NCDHW) {
    return dims.at(NCDHW_DIM_C);
  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Node[%s]: don't support the format [%d].", conv->GetName().c_str(), curFormat);
    return 0;
  }
}

Status BatchnormFusionPass::CheckInputTypeValid(const ge::NodePtr& originalNode, const ge::NodePtr& inputNode,
                                                const string& expectOpType) {
  FUSION_PASS_CHECK(inputNode->GetType() != expectOpType,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Match failed! Input of %s is not %s. It is %s, name %s",
                            originalNode->GetName().c_str(), expectOpType.c_str(), inputNode->GetType().c_str(),
                            inputNode->GetName().c_str()),
                    return FAILED);
  return SUCCESS;
}

Status BatchnormFusionPass::CheckInputTensorValid(const ge::GeTensorDesc& tensorDesc, const int64_t& kernelNum) {
  auto dims = tensorDesc.GetShape().GetDims();
  if (dims.size() != 1 || dims.at(0) != kernelNum) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Match failed! Dims is %u and dims[0] is %lu and kernel num is% lu", dims.size(),
            dims.at(0), kernelNum);
    return FAILED;
  }
  return SUCCESS;
}
Status BatchnormFusionPass::CheckPeerInDataAnchors(const ge::OutDataAnchorPtr& outputAnchor,
                                                   const size_t& expectedNum) {
  FUSION_PASS_CHECK(outputAnchor == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "outputAnchor must not be null"),
                    return PARAM_INVALID);
  if (outputAnchor->GetPeerInDataAnchors().size() == expectedNum) {
    return SUCCESS;
  }
  return FAILED;
}

Status BatchnormFusionPass::RemoveSmalleNodes(ge::ComputeGraph& graph, const ge::NodePtr& addNode,
                                              const ge::NodePtr& mulNode1, const ge::NodePtr& mulNode2,
                                              const ge::NodePtr& mulNode3, const ge::NodePtr& subNode) {
  FUSION_PASS_CHECK(graph.RemoveNode(addNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove node %s failed.", addNode->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(graph.RemoveNode(mulNode1) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove node %s failed.", addNode->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(graph.RemoveNode(mulNode2) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove node %s failed.", addNode->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(graph.RemoveNode(mulNode3) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove node %s failed.", addNode->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(graph.RemoveNode(subNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove node %s failed.", addNode->GetName().c_str()),
                    return FAILED);
  return SUCCESS;
}

Status BatchnormFusionPass::AddTensorDescForBn(const ge::OpDescPtr& bnOpdesc, const ge::GeTensorDesc& inputTensor,
                                               const ge::GeTensorDesc& scaleTensor,
                                               const ge::GeTensorDesc& offsetTensor, const ge::GeTensorDesc& meanTensor,
                                               const ge::GeTensorDesc& varianceTensor,
                                               const ge::GeTensorDesc& bnOutTensor) {
  FUSION_PASS_CHECK(bnOpdesc->AddInputDesc(inputTensor) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input0 of bn failed."), return FAILED);
  FUSION_PASS_CHECK(bnOpdesc->AddInputDesc("scale", scaleTensor) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input1 of bn failed."), return FAILED);
  FUSION_PASS_CHECK(bnOpdesc->AddInputDesc("offset", offsetTensor) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input2 of bn failed."), return FAILED);
  FUSION_PASS_CHECK(bnOpdesc->AddInputDesc("mean", meanTensor) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input3 of bn failed."), return FAILED);
  FUSION_PASS_CHECK(bnOpdesc->AddInputDesc("variance", varianceTensor) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add input4 of bn failed."), return FAILED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Size of input of BatchNorm is %u.", bnOpdesc->GetInputsSize());

  /* Output */
  FUSION_PASS_CHECK(bnOpdesc->AddOutputDesc("y", bnOutTensor) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output0 of bn failed."), return FAILED);
  FUSION_PASS_CHECK(bnOpdesc->AddOutputDesc("batch_mean", bnOutTensor) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output1 of bn failed."), return FAILED);
  FUSION_PASS_CHECK(bnOpdesc->AddOutputDesc("batch_variance", bnOutTensor) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add output2 of bn failed."), return FAILED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Size of output of BatchNorm is %u.", bnOpdesc->GetOutputsSize());
  return SUCCESS;
}

Status BatchnormFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter graph fusion BatchnormFusionPass!");
  ge::NodePtr addNode = GetNodeFromMapping(ADD_PATTERN, mapping);
  ge::NodePtr mulNode1 = GetNodeFromMapping(MUL_PATTERN_1, mapping);
  ge::NodePtr mulNode2 = GetNodeFromMapping(MUL_PATTERN_2, mapping);
  ge::NodePtr mulNode3 = GetNodeFromMapping(MUL_PATTERN_3, mapping);
  ge::NodePtr subNode = GetNodeFromMapping(SUB_PATTERN, mapping);
  ge::NodePtr rsqrtNode = GetNodeFromMapping(RSQRT_PATTERN, mapping);
  ge::NodePtr addEpsNode = GetNodeFromMapping(ADD_EPS_PATTERN, mapping);
  ge::NodePtr convNode = GetNodeFromMapping(CONV3D_PATTERN, mapping);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Check node null or not");
  FUSION_PASS_CHECK(convNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "convNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(CheckPeerInDataAnchors(convNode->GetOutDataAnchor(0), 1) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "%s contains more than two peer input", convNode->GetName().c_str()),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(addNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "addNode is null, fusion failed."),
                    return PARAM_INVALID);

  FUSION_PASS_CHECK(mulNode1 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulNode1 is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(CheckPeerInDataAnchors(mulNode1->GetOutDataAnchor(0), 2) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "%s contains more than two peer input", mulNode1->GetName().c_str()),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(mulNode2 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulNode2 is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(CheckPeerInDataAnchors(mulNode2->GetOutDataAnchor(0), 1) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "%s contains more than one peer input", mulNode2->GetName().c_str()),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(mulNode3 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulNode3 is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(CheckPeerInDataAnchors(mulNode3->GetOutDataAnchor(0), 1) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "%s contains more than one peer input", mulNode3->GetName().c_str()),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(subNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "subNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(CheckPeerInDataAnchors(mulNode3->GetOutDataAnchor(0), 1) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "%s contains more than one peer input", mulNode3->GetName().c_str()),
                    return NOT_CHANGED);

  /* 1. Get owner node and previous edge from conv2D's out */
  /* Get input of BatchNorm (x) */
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to get weight of batchnorm");
  ge::OutDataAnchorPtr outAnchorOfConv3D = mulNode3->GetInDataAnchor(0)->GetPeerOutAnchor();
  ge::NodePtr inputNode = outAnchorOfConv3D->GetOwnerNode();
  ge::GeTensorDesc inputTensor = mulNode3->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(CheckInputTypeValid(mulNode3, inputNode, CONV3D) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Input Type invalid."), return NOT_CHANGED);
  uint32_t kernelNum = GetKernelNumOfOutputOfConv3D(inputNode);

  /* Get scale (gamma) */
  FUSION_PASS_CHECK(mulNode1->GetOpDesc()->GetInputsSize() < 2,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mulNode1's %s input size is < 2", mulNode1->GetName().c_str()),
                    return PARAM_INVALID);
  ge::OutDataAnchorPtr outAnchorOfScale = mulNode1->GetInDataAnchor(1)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(outAnchorOfScale == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "outAnchorOfScale must not be null"),
                    return PARAM_INVALID);
  ge::NodePtr scaleNode = outAnchorOfScale->GetOwnerNode();
  FUSION_PASS_CHECK(scaleNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "scaleNode must not be null"),
                    return PARAM_INVALID);
  ge::GeTensorDesc scaleTensor = mulNode1->GetOpDesc()->GetInputDesc(1);
  FUSION_PASS_CHECK(CheckInputTypeValid(mulNode1, scaleNode, CONST) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Input Type invalid."), return NOT_CHANGED);
  FUSION_PASS_CHECK(CheckInputTensorValid(scaleTensor, kernelNum) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Input tensor invalid."), return NOT_CHANGED);

  /* Get offset (beta) */
  ge::OutDataAnchorPtr outAnchorOfOffset = subNode->GetInDataAnchor(0)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(outAnchorOfOffset == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "outAnchorOfOffset must not be null"),
                    return PARAM_INVALID);
  ge::NodePtr offsetNode = outAnchorOfOffset->GetOwnerNode();
  FUSION_PASS_CHECK(offsetNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "offsetNode must not be null"),
                    return PARAM_INVALID);
  ge::GeTensorDesc offsetTensor = subNode->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(CheckInputTypeValid(subNode, offsetNode, CONST) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Input Type invalid."), return NOT_CHANGED);
  FUSION_PASS_CHECK(CheckInputTensorValid(offsetTensor, kernelNum) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Input tensor invalid."), return NOT_CHANGED);

  /* Get mean */
  ge::OutDataAnchorPtr outAnchorOfMean = mulNode2->GetInDataAnchor(0)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(outAnchorOfMean == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "outAnchorOfMean must not be null"),
                    return PARAM_INVALID);
  ge::NodePtr meanNode = outAnchorOfMean->GetOwnerNode();
  FUSION_PASS_CHECK(meanNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "meanNode must not be null"),
                    return PARAM_INVALID);
  ge::GeTensorDesc meanTensor = mulNode2->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(CheckInputTypeValid(mulNode2, meanNode, CONST) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Input Type invalid."), return NOT_CHANGED);
  FUSION_PASS_CHECK(CheckInputTensorValid(meanTensor, kernelNum) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Input tensor invalid."), return NOT_CHANGED);

  /* Get variance */
  /* If the variance mul's first input(input 0) is Rsqrt, we need to do the
   * Adding eps and caculation the reciprocal of square root
   * in Conv + Bn Fusion */
  bool needDeleteAddAndSqrt = false;
  ge::OutDataAnchorPtr outAnchorOfVariance;
  ge::NodePtr varianceNode;
  ge::GeTensorDesc varianceTensor;

  if (rsqrtNode == nullptr) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Rsqrt Node is null, we don't need to add eps and rsqrt for %s",
            mulNode3->GetName().c_str());
    outAnchorOfVariance = mulNode1->GetInDataAnchor(0)->GetPeerOutAnchor();
    FUSION_PASS_CHECK(outAnchorOfVariance == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "outAnchorOfVariance must not be null"), return PARAM_INVALID);
    varianceNode = outAnchorOfVariance->GetOwnerNode();
    FUSION_PASS_CHECK(varianceNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "varianceNode must not be null"),
                      return PARAM_INVALID);
    varianceTensor = mulNode1->GetOpDesc()->GetInputDesc(0);
  } else if (addEpsNode != nullptr && rsqrtNode != nullptr) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "we need to add eps and rsqrt for %s.", mulNode3->GetName().c_str());
    /* input 0 of addEpsNode is variance and input 1 is eps */
    outAnchorOfVariance = addEpsNode->GetInDataAnchor(0)->GetPeerOutAnchor();
    FUSION_PASS_CHECK(
        CheckPeerInDataAnchors(addEpsNode->GetOutDataAnchor(0), 1) != SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "%s contains more than two peer input", addEpsNode->GetName().c_str()),
        return NOT_CHANGED);
    FUSION_PASS_CHECK(
        CheckPeerInDataAnchors(rsqrtNode->GetOutDataAnchor(0), 1) != SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "%s contains more than two peer input", addEpsNode->GetName().c_str()),
        return NOT_CHANGED);
    FUSION_PASS_CHECK(outAnchorOfVariance == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "outAnchorOfVariance must not be null"), return PARAM_INVALID);
    varianceNode = outAnchorOfVariance->GetOwnerNode();
    FUSION_PASS_CHECK(varianceNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "varianceNode must not be null"),
                      return PARAM_INVALID);
    varianceTensor = addEpsNode->GetOpDesc()->GetInputDesc(0);
    needDeleteAddAndSqrt = true;
  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "addEpsNode is null and rsqrtNode is not null for %s", mulNode3->GetName().c_str());
    return FAILED;
  }
  FUSION_PASS_CHECK(CheckInputTypeValid(mulNode1, varianceNode, CONST) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Input Type invalid."), return NOT_CHANGED);
  FUSION_PASS_CHECK(CheckInputTensorValid(varianceTensor, kernelNum) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Input tensor invalid."), return NOT_CHANGED);

  /* 2. Get output node (relu) and edge */
  ge::OutDataAnchorPtr outAnchorOfAdd = addNode->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(outAnchorOfAdd == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "outAnchorOfAdd must not be null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(outAnchorOfAdd->GetPeerInDataAnchors().empty(),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "size of peer in anchor of add is 0"), return FAILED);
  ge::GeTensorDesc bnOutTensor = addNode->GetOpDesc()->GetOutputDesc(0);

  /* 3. Add new Opdesc of BatchNorm*/
  std::shared_ptr<ge::OpDesc> bnOpdesc = nullptr;
  std::string bnNodeName = addNode->GetName() + "_" + "batchnorm";
  bnOpdesc = std::make_shared<ge::OpDesc>(bnNodeName, BATCHNORM);

  FUSION_PASS_CHECK(bnOpdesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "bnOpdesc is null, fusion failed."),
                    return PARAM_INVALID);

  /* 4. Add TensorDesc into new node. Bn must have five inputs and three
   * outputs. If not, we will return false. */
  (void)AddTensorDescForBn(bnOpdesc, inputTensor, scaleTensor, offsetTensor, meanTensor, varianceTensor, bnOutTensor);
  /* 5. Add Node into graph */
  ge::NodePtr bnNode = graph.AddNode(bnOpdesc);
  FUSION_PASS_CHECK(bnNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "bnNode is null, fusion failed."),
                    return PARAM_INVALID);
  newNodes.push_back(bnNode);

  /* 6. remove old edges and add new edges */
  /* remove input from mulNode3 and add it to batchnorm */
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(outAnchorOfConv3D, bnNode->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(outAnchorOfConv3D, mulNode3->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);

  /* remove scale (gamma) from mul node 1 and Add it to batchnorm */
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(outAnchorOfScale, bnNode->GetInDataAnchor(1)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add scale(gamma) edge failed."), return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(outAnchorOfScale, mulNode1->GetInDataAnchor(1)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove scale(gamma) edge failed."), return FAILED);

  /* remove offset (beta) from sub and Add it to batchnorm */
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(outAnchorOfOffset, bnNode->GetInDataAnchor(2)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add offset(beta) edge failed."), return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(outAnchorOfOffset, subNode->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove offset(beta) edge failed."), return FAILED);

  /* remove mean from mul node 3 and Add it to batchnorm */
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(outAnchorOfMean, bnNode->GetInDataAnchor(3)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add mean edge failed."), return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(outAnchorOfMean, mulNode2->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove mean edge failed."), return FAILED);

  /* remove variance from mul node 2 and Add it to batchnorm*/
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(outAnchorOfVariance, bnNode->GetInDataAnchor(4)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add variance edge failed."), return FAILED);

  /* If the last add Node contains more than one peer input,
   * we will first add edge between the peer input and conv's output */
  size_t index = 0;
  for (auto& peerInAnchor : outAnchorOfAdd->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(peerInAnchor == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "peerInAnchor must not be null"),
                      return PARAM_INVALID);
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(outAnchorOfAdd, peerInAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove edge %lu failed.", index), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(bnNode->GetOutDataAnchor(0), peerInAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge %lu failed.", index), return FAILED);
    index++;
  }

  /* remove all small nodes */
  Status ret = RemoveSmalleNodes(graph, addNode, mulNode1, mulNode2, mulNode3, subNode);
  if (ret != SUCCESS) {
    return ret;
  }

  /* Set Attr not_need_adding_eps_and_sqrting and remove add and sqrt node */
  /* Set epsilon */
  float eps = 1e-8;
  if (needDeleteAddAndSqrt) {
    ge::AttrUtils::SetBool(bnOpdesc, "need_adding_eps_and_rsqrting", true);
    /* The edge between Variance and AddEps is not deleted, so the
     * second weight of AddEps is the const of epsilon and there are two
     * weights for this AddEps node. */
    vector<ge::ConstGeTensorPtr> addEpsWeights = ge::OpDescUtils::GetWeights(addEpsNode);
    FUSION_PASS_CHECK(addEpsWeights.size() < 2,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "size of weight which is %lu of %s is less than 2",
                              addEpsWeights.size(), addEpsNode->GetName().c_str()),
                      return NOT_CHANGED);
    eps = *((float*)addEpsWeights[1]->GetData().data());

    FUSION_PASS_CHECK(graph.RemoveNode(rsqrtNode) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove node %s failed.", addNode->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(addEpsNode) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove node %s failed.", addNode->GetName().c_str()),
                      return FAILED);
  } else {
    (void)ge::AttrUtils::SetBool(bnOpdesc, "need_adding_eps_and_rsqrting", false);
  }

  FUSION_PASS_CHECK(!ge::AttrUtils::SetFloat(bnOpdesc, ge::BATCHNORM_ATTR_EPSILON, eps),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "DestNode[%s]: set epsilon attr %s not success.",
                            bnOpdesc->GetName().c_str(), ge::BATCHNORM_ATTR_EPSILON.c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetBool(bnOpdesc, "is_training", false),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set is_traing attr failed."), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "BatchNorm graph fusion success for node %s!", addNode->GetName().c_str());
  return SUCCESS;
}
REGISTER_PASS("ABatchnormFusionPass", BUILT_IN_GRAPH_PASS, BatchnormFusionPass);
}  // namespace fe
