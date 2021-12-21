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
 * \file transdata_confusiontransposed_fusion_pass.cc
 * \brief
 */
#include "transdata_confusiontransposed_fusion_pass.h"

#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <memory>

#include "external/graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"

using namespace std;
using namespace ge;

namespace fe {
static const char PATTERN_TRANSDATA_1[] = "TransData_1";
static const char PATTERN_REFORMAT[] = "ReFormat";
static const char PATTERN_CONFUSIONTRANSPOSE[] = "ConfusionTransposeD";
static const char PATTERN_TRANSDATA_2[] = "TransData_2";

static const char OPTYPE_TRANSDATA[] = "TransData";
static const char OPTYPE_REFORMAT[] = "ReFormat";
static const char OPTYPE_CONFUSIONTRANSPOSE[] = "ConfusionTransposeD";

/*!
 * @brief Define pattern.
 * The graph struct need to adapt is shown as follows:
 *
 * transdata1 --> confusion_transpose_d --> transdata2
 *
 * @return vector<FusionPattern*> All valid patterns.
 */
vector<FusionPattern*> TransDataConfusionTransposeDFusionPass::DefinePatterns() {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define pattern begin");

  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("TransDataConfusionTransposeDFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create an object"),
                    return patterns);

  pattern->AddOpDesc(PATTERN_TRANSDATA_1, {OPTYPE_TRANSDATA})
      .AddOpDesc(PATTERN_REFORMAT, {OPTYPE_REFORMAT})
      .AddOpDesc(PATTERN_CONFUSIONTRANSPOSE, {OPTYPE_CONFUSIONTRANSPOSE})
      .AddOpDesc(PATTERN_TRANSDATA_2, {OPTYPE_TRANSDATA})
      .SetInputs(PATTERN_REFORMAT, {PATTERN_TRANSDATA_1})
      .SetInputs(PATTERN_CONFUSIONTRANSPOSE, {PATTERN_REFORMAT})
      .SetInputs(PATTERN_TRANSDATA_2, {PATTERN_CONFUSIONTRANSPOSE})
      .SetOutput(PATTERN_TRANSDATA_2);
  patterns.push_back(pattern);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define pattern end");
  return patterns;
}

Status TransDataConfusionTransposeDFusionPass::CheckNodeInfo(const ge::NodePtr& transData1, const ge::NodePtr& reformat,
                                                             const ge::NodePtr& confusionTransposeD,
                                                             const ge::NodePtr& transData2) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckNodeInfo begin");

  FUSION_PASS_CHECK(transData1 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transData1 is null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(reformat == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reformat is null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(confusionTransposeD == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "confusionTransposeD is null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(transData2 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transData2 is null"),
                    return PARAM_INVALID);

  OP_LOGD(
      FUSED_OP_TYPE.c_str(),
      "TransData1 node name: %s. ReFormat node name: %s. ConfusionTransposeD node name: %s. TransDatas node name: %s",
      transData1->GetName().c_str(), reformat->GetName().c_str(), confusionTransposeD->GetName().c_str(),
      transData2->GetName().c_str());

  FUSION_PASS_CHECK(!transData1->GetInControlNodes().empty() || !transData1->GetOutControlNodes().empty(),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "TransData 1 has control edge."), return NOT_CHANGED);
  FUSION_PASS_CHECK(transData1->GetOutDataNodesSize() > 1,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "TransData 1 has more than 1 out node."), return NOT_CHANGED);

  FUSION_PASS_CHECK(!reformat->GetInControlNodes().empty() || !reformat->GetOutControlNodes().empty(),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "ReFormat has control edge."), return NOT_CHANGED);
  FUSION_PASS_CHECK(reformat->GetOutDataNodesSize() > 1,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "ReFormat has more than 1 out node."), return NOT_CHANGED);

  FUSION_PASS_CHECK(
      !confusionTransposeD->GetInControlNodes().empty() || !confusionTransposeD->GetOutControlNodes().empty(),
      OP_LOGD(FUSED_OP_TYPE.c_str(), "ConfusionTransposeD has control edge."), return NOT_CHANGED);
  FUSION_PASS_CHECK(confusionTransposeD->GetOutDataNodesSize() > 1,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "ConfusionTransposeD has more than 1 out node."),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(!transData2->GetInControlNodes().empty() || !transData2->GetOutControlNodes().empty(),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "TransData 2 has control edge."), return NOT_CHANGED);
  FUSION_PASS_CHECK(transData2->GetOutDataNodesSize() > 1,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "TransData 2 has more than 1 out node."), return NOT_CHANGED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckNodeInfo end");
  return SUCCESS;
}

Status TransDataConfusionTransposeDFusionPass::CheckOpInfo(const ge::NodePtr& transData1,
                                                           const ge::NodePtr& confusionTransposeD,
                                                           const ge::NodePtr& transData2) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckOpInfo begin");

  ge::OpDescPtr transData1OpDesc = transData1->GetOpDesc();
  FUSION_PASS_CHECK(transData1OpDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transData1 op desc is null"),
                    return PARAM_INVALID);

  ge::GeTensorDesc transData1InputTensor = transData1OpDesc->GetInputDesc(0);
  ge::GeTensorDesc transData1OutputTensor = transData1OpDesc->GetOutputDesc(0);
  FUSION_PASS_CHECK(
      transData1OutputTensor.GetFormat() != ge::FORMAT_ND ||
          ge::GetPrimaryFormat(transData1InputTensor.GetFormat()) != ge::FORMAT_FRACTAL_NZ,
      OP_LOGD(FUSED_OP_TYPE.c_str(),
              "For the first TransData node, input format should be FRACTAL_NZ, and output format should be ND"),
      return NOT_CHANGED);

  ge::OpDescPtr transData2OpDesc = transData2->GetOpDesc();
  FUSION_PASS_CHECK(transData2OpDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transData2 op desc is null"),
                    return PARAM_INVALID);

  ge::GeTensorDesc transData2InputTensor = transData2OpDesc->GetInputDesc(0);
  ge::GeTensorDesc transData2OutputTensor = transData2OpDesc->GetOutputDesc(0);
  FUSION_PASS_CHECK(
      transData2InputTensor.GetFormat() != ge::FORMAT_ND ||
          ge::GetPrimaryFormat(transData2OutputTensor.GetFormat()) != ge::FORMAT_FRACTAL_NZ,
      OP_LOGD(FUSED_OP_TYPE.c_str(),
              "For the second TransData node, input format should be ND, and output fromat should be FRACTAL_NZ"),
      return NOT_CHANGED);

  Operator confusionTransposeDOp = ge::OpDescUtils::CreateOperatorFromNode(confusionTransposeD);
  std::vector<int64_t> permList;
  FUSION_PASS_CHECK(confusionTransposeDOp.GetAttr("perm", permList) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to get perm of ConfusionTransposeD!"),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(permList.size() != 4, OP_LOGD(FUSED_OP_TYPE.c_str(), "Length of perm should be 4!"),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(permList[0] != 0 || permList[1] != 2 || permList[2] != 3 || permList[3] != 1,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Support perm of ConfusionTransposeD is (0, 2, 3, 1)"),
                    return NOT_CHANGED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckOpInfo end");
  return SUCCESS;
}

Status TransDataConfusionTransposeDFusionPass::CheckShapeInfo(const ge::NodePtr& transData1,
                                                              const ge::NodePtr& confusionTransposeD,
                                                              const ge::NodePtr& transData2) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckShapeInfo begin");

  ge::OpDescPtr transData1OpDesc = transData1->GetOpDesc();
  ge::GeTensorDesc transData1InputTensor = transData1OpDesc->GetInputDesc(0);
  ge::GeShape inputOriginShape = transData1InputTensor.GetOriginShape();
  FUSION_PASS_CHECK(inputOriginShape.IsUnknownShape(),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "TransData 1 input origin shape is unkown shape."),
                    return NOT_CHANGED);
  ge::GeShape inputShape = transData1InputTensor.GetShape();
  FUSION_PASS_CHECK(inputShape.IsUnknownShape(),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "TransData 1 input shape is unkown shape."), return NOT_CHANGED);

  ge::OpDescPtr transData2OpDesc = transData2->GetOpDesc();
  ge::GeTensorDesc transData2OutputTensor = transData2OpDesc->GetOutputDesc(0);
  ge::GeShape outputOriginShape = transData2OutputTensor.GetOriginShape();
  FUSION_PASS_CHECK(outputOriginShape.IsUnknownShape(),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "TransData 2 input origin shape is unkown shape."),
                    return NOT_CHANGED);

  ge::GeShape outputShape = transData2OutputTensor.GetShape();
  FUSION_PASS_CHECK(outputShape.IsUnknownShape(),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "TransData 2 input shape is unkown shape."), return NOT_CHANGED);

  size_t inputOriginShapeDimNum = 3;
  size_t inputShapeDimNum = 5;
  size_t outputOriginShapeDimNum = 4;
  size_t outputShapeDimNum = 6;
  FUSION_PASS_CHECK(
      inputOriginShape.GetDimNum() != inputOriginShapeDimNum || inputShape.GetDimNum() != inputShapeDimNum ||
          outputOriginShape.GetDimNum() != outputOriginShapeDimNum || outputShape.GetDimNum() != outputShapeDimNum,
      OP_LOGD(FUSED_OP_TYPE.c_str(), "The dim num not match the fusion condition."), return NOT_CHANGED);

  DataType transData1DataType = transData1InputTensor.GetDataType();
  DataType transData2DataType = transData2OutputTensor.GetDataType();
  FUSION_PASS_CHECK(DT_FLOAT16 != transData1DataType || DT_FLOAT16 != transData2DataType,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Data type should be float16."), return NOT_CHANGED);

  FUSION_PASS_CHECK(inputOriginShape.GetDim(1) % 16 != 0 || inputOriginShape.GetDim(2) % 16 != 0,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "The last 2 dims of transdata 1 shape should be multiple of 16."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(
      inputOriginShape.GetDim(0) != inputShape.GetDim(0) ||
          inputOriginShape.GetDim(1) != inputShape.GetDim(2) * inputShape.GetDim(3) ||
          inputOriginShape.GetDim(2) != inputShape.GetDim(1) * inputShape.GetDim(4),
      OP_LOGD(FUSED_OP_TYPE.c_str(), "The Shape and OriginShape of transdata1 not match the fusion condition."),
      return NOT_CHANGED);

  FUSION_PASS_CHECK(
      outputOriginShape.GetDim(1) == 0,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "The origin shape of transdata 2 contains 0."),
      return NOT_CHANGED);
  FUSION_PASS_CHECK(outputShape.GetDim(1) == 0,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "The shape of transdata 2 contains 0."),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(outputOriginShape.GetDim(2) % 16 != 0 || outputOriginShape.GetDim(3) % 16 != 0,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "The last 2 dims of transdata 2 shape should be multiple of 16."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(
      outputOriginShape.GetDim(0) != outputShape.GetDim(0) || outputOriginShape.GetDim(1) != outputShape.GetDim(1) ||
          outputOriginShape.GetDim(2) != outputShape.GetDim(3) * outputShape.GetDim(4) ||
          outputOriginShape.GetDim(3) != outputShape.GetDim(2) * outputShape.GetDim(5),
      OP_LOGD(FUSED_OP_TYPE.c_str(), "The Shape and OriginShape of transdata2 not match the fusion condition."),
      return NOT_CHANGED);

  FUSION_PASS_CHECK(
      inputOriginShape.GetDim(0) != outputOriginShape.GetDim(0) ||
          inputOriginShape.GetDim(1) != outputOriginShape.GetDim(3) ||
          inputOriginShape.GetDim(2) != outputOriginShape.GetDim(1) * outputOriginShape.GetDim(2),
      OP_LOGD(FUSED_OP_TYPE.c_str(),
              "The OriginShape compare between transdata1 and transdata2 not match the fusion condition."),
      return NOT_CHANGED);

  FUSION_PASS_CHECK(inputShape.GetDim(0) != outputShape.GetDim(0) ||
                        inputShape.GetDim(1) != outputShape.GetDim(1) * outputShape.GetDim(3) ||
                        inputShape.GetDim(2) != outputShape.GetDim(2) ||
                        inputShape.GetDim(3) != outputShape.GetDim(5) || inputShape.GetDim(4) != outputShape.GetDim(4),
                    OP_LOGD(FUSED_OP_TYPE.c_str(),
                            "The Shape compare between transdata1 and transdata2 not match the fusion condition."),
                    return NOT_CHANGED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "CheckShapeInfo end");
  return SUCCESS;
}

Status TransDataConfusionTransposeDFusionPass::InsertTransposeDNode(ge::ComputeGraph& graph,
                                                                   const ge::NodePtr& transData1,
                                                                   const ge::NodePtr& transData2,
                                                                   ge::NodePtr& transposeNode) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "InsertTransposeDNode begin");

  auto transData1InAnchor = transData1->GetInDataAnchor(0);
  auto preNodeOutAnchor = transData1InAnchor->GetPeerOutAnchor();
  ge::NodePtr preNode = preNodeOutAnchor->GetOwnerNode();
  ge::OpDescPtr preNodeOpDesc = preNode->GetOpDesc();
  std::string nodeName = preNodeOpDesc->GetName() + "/TransposeD";
  ge::OpDescPtr transposeDNodeOpDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((transposeDNodeOpDesc = std::make_shared<ge::OpDesc>(nodeName, "TransposeD")),
                          return INTERNAL_ERROR);

  ge::OpDescPtr transData1OpDesc = transData1->GetOpDesc();
  ge::OpDescPtr transData2OpDesc = transData2->GetOpDesc();

  ge::GeTensorDesc xDesc = transData1OpDesc->GetInputDesc(0).Clone();
  FUSION_PASS_CHECK(transposeDNodeOpDesc->AddInputDesc("x", xDesc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to add input x for transpose node."), return FAILED);
  transposeDNodeOpDesc->MutableInputDesc(0)->SetFormat(ge::FORMAT_ND);

  std::vector<int64_t> transData1InputOriginShape = transData1OpDesc->GetInputDesc(0).GetOriginShape().GetDims();
  std::vector<int64_t> transData2OutputOriginShape = transData2OpDesc->GetOutputDesc(0).GetOriginShape().GetDims();
  std::vector<int64_t> xOriginShape;
  xOriginShape.push_back(transData1InputOriginShape.at(0));
  xOriginShape.push_back(transData2OutputOriginShape.at(1));
  xOriginShape.push_back(transData1InputOriginShape.at(1));
  xOriginShape.push_back(transData1InputOriginShape.at(2) / transData2OutputOriginShape.at(1));
  transposeDNodeOpDesc->MutableInputDesc(0)->SetOriginShape(ge::GeShape(xOriginShape));

  std::vector<int64_t> transData1InputShape = transData1OpDesc->GetInputDesc(0).GetShape().GetDims();
  std::vector<int64_t> transData2OutputShape = transData2OpDesc->GetOutputDesc(0).GetShape().GetDims();
  std::vector<int64_t> xShape;
  xShape.push_back(transData1InputShape.at(0));
  xShape.push_back(transData2OutputShape.at(1));
  xShape.push_back(transData1InputShape.at(1) / transData2OutputShape.at(1));
  xShape.push_back(transData1InputShape.at(2));
  xShape.push_back(transData1InputShape.at(3));
  xShape.push_back(transData1InputShape.at(4));
  transposeDNodeOpDesc->MutableInputDesc(0)->SetShape(ge::GeShape(xShape));
  
  std::vector<int64_t> perm = {0, 1, 3, 2, 5, 4};
  ge::AttrUtils::SetListInt(transposeDNodeOpDesc, "perm", perm);

  ge::GeTensorDesc yDesc = transData2OpDesc->GetOutputDesc(0).Clone();
  FUSION_PASS_CHECK(transposeDNodeOpDesc->AddOutputDesc("y", yDesc) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to add output y for transpose node."), return FAILED);
  transposeDNodeOpDesc->MutableOutputDesc(0)->SetFormat(ge::FORMAT_ND);

  transposeNode = graph.AddNode(transposeDNodeOpDesc);
  FUSION_PASS_CHECK(transposeNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to add transpose node."),
                    return FAILED);

  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(preNodeOutAnchor, transposeNode->GetInDataAnchor(0)),
                    VECTOR_FUSION_INNER_ERR_REPORT(
                        FUSED_OP_TYPE.c_str(), "Failed to add edge from pre node's output to transpose node's input."),
                    return FAILED);

  auto postInAnchors = transData2->GetOutDataAnchor(0)->GetPeerInDataAnchors();
  if (postInAnchors.size() > 0) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "The output edge of TransData2 is [%d].", postInAnchors.size());
    for (InDataAnchorPtr inAnchorPtr : postInAnchors) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(transposeNode->GetOutDataAnchor(0), inAnchorPtr),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                         "Failed to add edge from transpose node's output to post node's input."),
          return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Success to add edge from transpose node's output to post node's input.");
    }
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "InsertTransposeNode end");
  return SUCCESS;
}

Status TransDataConfusionTransposeDFusionPass::RemoveNode(ge::ComputeGraph& graph, const ge::NodePtr& node) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "RemoveNode begin. Node name %s", node->GetName().c_str());

  for (size_t i = 0; i < node->GetAllInDataAnchors().size(); ++i) {
    auto inDataAnchor = node->GetInDataAnchor(i);
    FUSION_PASS_CHECK(
        inDataAnchor == nullptr,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove node for inDataAnchor is null."),
        return FAILED);
    auto preOutDataAnchor = inDataAnchor->GetPeerOutAnchor();

    FUSION_PASS_CHECK(
        preOutDataAnchor == nullptr,
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove node for preOutDataAnchor is null."),
        return FAILED);

    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(preOutDataAnchor, inDataAnchor) != ge::GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove node."), return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "remove edge %u of node %s", i, node->GetName().c_str());
  }

  FUSION_PASS_CHECK(graph.RemoveNode(node) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove node"), return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "RemoveNode end");
  return SUCCESS;
}

Status TransDataConfusionTransposeDFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                      vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Fusion begin");

  ge::NodePtr transData1 = GetNodeFromMapping(PATTERN_TRANSDATA_1, mapping);
  ge::NodePtr reformat = GetNodeFromMapping(PATTERN_REFORMAT, mapping);
  ge::NodePtr confusionTransposeD = GetNodeFromMapping(PATTERN_CONFUSIONTRANSPOSE, mapping);
  ge::NodePtr transData2 = GetNodeFromMapping(PATTERN_TRANSDATA_2, mapping);

  Status checkNodeRet = CheckNodeInfo(transData1, reformat, confusionTransposeD, transData2);
  FUSION_PASS_CHECK(SUCCESS != checkNodeRet, OP_LOGD(FUSED_OP_TYPE.c_str(), "Failed to check node info."),
                    return checkNodeRet);

  Status checkOpRet = CheckOpInfo(transData1, confusionTransposeD, transData2);
  FUSION_PASS_CHECK(SUCCESS != checkOpRet, OP_LOGD(FUSED_OP_TYPE.c_str(), "Failed to check node info."),
                    return checkOpRet);

  Status checkShapeRet = CheckShapeInfo(transData1, confusionTransposeD, transData2);
  FUSION_PASS_CHECK(SUCCESS != checkShapeRet, OP_LOGD(FUSED_OP_TYPE.c_str(), "Unsupport parameters. Fusion end."),
                    return checkShapeRet);

  ge::NodePtr transposeNode = nullptr;
  Status insertTransposeRet = InsertTransposeDNode(graph, transData1, transData2, transposeNode);
  FUSION_PASS_CHECK(SUCCESS != insertTransposeRet, OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to insert Transpose node."),
                    return insertTransposeRet);
  newNodes.push_back(transposeNode);

  FUSION_PASS_CHECK(graph.RemoveNode(transData1) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove transData1 node"),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(transData2) != GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove transData2 node"),
                    return FAILED);
  FUSION_PASS_CHECK(RemoveNode(graph, confusionTransposeD) == FAILED,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove confusionTransposeD node"),
                    return FAILED);
  FUSION_PASS_CHECK(RemoveNode(graph, reformat) == FAILED,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove reformat node"),
                    return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Fusion end");
  return SUCCESS;
}

REGISTER_PASS("TransDataTransposeFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS,
              TransDataConfusionTransposeDFusionPass);
}  // namespace fe
