/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file mean_grad_fusion_pass.cpp
 * \brief
 */
#include "mean_grad_fusion_pass.h"

#include "graph/debug/ge_attr_define.h"
#include "graph/ge_attr_value.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "external/graph/types.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;
namespace fe {

static const char* RESHAPE = "Reshape";
static const char* TILE = "Tile";
static const char* MULTIPLY = "Multiply";

static const string PATTERN_RESHAPE = "reshape";
static const string PATTERN_TILE = "tile";
static const string PATTERN_TRUEDIV = "truediv";

/*MeanGradInput*/
static const std::string MEAN_GRAD_OUTPUT_SHAPE_VALUE = "mean_grad_output_shape_value";
static const std::string MEAN_GRAD_OUTPUT_SHAPE_FORMAT = "mean_grad_output_shape_format";

const int32_t OUTPUT_W_INDEX = 2;

/* before:
 * shape(const)  floordiv(const)  recip(const)
 *     \             \                \
 * data-->reshepe -> tile --> truediv(mul)-->data
 *
 * after:
 * data-->mean_grad-->data
 */
vector<FusionPattern*> MeanGradFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("MeanGradFusion");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new an object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_RESHAPE, {RESHAPE})
      .AddOpDesc(PATTERN_TILE, {TILE})
      .AddOpDesc(PATTERN_TRUEDIV, {MULTIPLY})
      .SetInputs(PATTERN_TILE, {PATTERN_RESHAPE})
      .SetInputs(PATTERN_TRUEDIV, {PATTERN_TILE})
      .SetOutput(PATTERN_TRUEDIV);
  patterns.push_back(pattern);

  return patterns;
}

Status MeanGradFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter MeanGradFusionPass!");

  ge::NodePtr reshapeNode = GetNodeFromMapping(PATTERN_RESHAPE, mapping);
  ge::NodePtr tileNode = GetNodeFromMapping(PATTERN_TILE, mapping);
  ge::NodePtr truedivNode = GetNodeFromMapping(PATTERN_TRUEDIV, mapping);
  FUSION_PASS_CHECK(reshapeNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "reshapeNode is nullptr."),
                    return FAILED);
  FUSION_PASS_CHECK(tileNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "tileNode is nullptr."),
                    return FAILED);
  FUSION_PASS_CHECK(truedivNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "truedivNode is nullptr."),
                    return FAILED);
  int32_t outputN = 0;
  int32_t outputH = 0;
  int32_t outputW = 0;
  int32_t outputC = 0;

  string type;

  // extract para from input const node of tileNode and remove const node
  FUSION_PASS_CHECK(
      !(tileNode->GetInDataNodes().size() == 2),
      OP_LOGW(FUSED_OP_TYPE.c_str(),
              "Input nodes of tile node not equal to 2, exit MeanGradFusionPass with status: not changed."),
      return NOT_CHANGED);
  FUSION_PASS_CHECK(
      !(truedivNode->GetInDataNodes().size() == 2),
      OP_LOGW(FUSED_OP_TYPE.c_str(),
              "Input nodes of truediv node not equal to 2, exit MeanGradFusionPass with status: not changed."),
      return NOT_CHANGED);
  FUSION_PASS_CHECK(
      !(reshapeNode->GetInDataNodes().size() == 2),
      OP_LOGW(FUSED_OP_TYPE.c_str(),
              "Input nodes of reshape node not equal to 2, exit MeanGradFusionPass with status: not changed."),
      return NOT_CHANGED);
  FUSION_PASS_CHECK(
      !(truedivNode->GetAllOutDataAnchors().size() == 1),
      OP_LOGW(FUSED_OP_TYPE.c_str(),
              "Output anchor of truediv node not equal to 1, exit MeanGradFusionPass with status: not changed."),
      return NOT_CHANGED);
  Status ret;
  bool extractParaSuccess = false;
  for (auto inNode : tileNode->GetInDataNodes()) {
    string nodeType = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(inNode);
    if ((nodeType == "Constant") || (nodeType == "Const")) {
      ret = ParseParaFromConst(inNode, outputH, 1);
      FUSION_PASS_CHECK(ret != SUCCESS,
                        OP_LOGI(FUSED_OP_TYPE.c_str(), "get reshape output height failed, not changed."),
                        return NOT_CHANGED);
      ret = ParseParaFromConst(inNode, outputW, OUTPUT_W_INDEX);
      FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGI(FUSED_OP_TYPE.c_str(), "get reshape output width failed, not changed."),
                        return NOT_CHANGED);
      extractParaSuccess = true;
      ret = RemoveConstOpInput(graph, tileNode);
      FUSION_PASS_CHECK(ret != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove %s node failed.",
                                                       inNode->GetName().c_str()),
                        return ret);
    }
  }
  FUSION_PASS_CHECK(!extractParaSuccess, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "extract factor failed."),
                    return FAILED);
  extractParaSuccess = false;
  // remove input const node of truedivNode
  for (auto inNode : truedivNode->GetInDataNodes()) {
    string nodeType = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(inNode);
    if ((nodeType == "Constant") || (nodeType == "Const")) {
      ret = RemoveConstOpInput(graph, truedivNode);
      FUSION_PASS_CHECK(ret != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove %s node failed.",
                                                       inNode->GetName().c_str()),
                        return ret);
    }
  }

  // extract para from input const node of reshapeNode and remove const node
  for (auto inNode : reshapeNode->GetInDataNodes()) {
    string nodeType = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(inNode);
    if ((nodeType == "Constant") || (nodeType == "Const")) {
      ret = ParseParaFromConst(inNode, outputN, 0);
      FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                        "get reshape output batch failed."),
                        return ret);
      ret = ParseParaFromConst(inNode, outputC, 3);
      FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                        "get reshape output channel failed."),
                        return ret);
      ret = RemoveConstOpInput(graph, reshapeNode);
      FUSION_PASS_CHECK(ret != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove %s node failed.",
                                                       inNode->GetName().c_str()),
                        return ret);
      extractParaSuccess = true;
    }
  }
  FUSION_PASS_CHECK(!extractParaSuccess, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "extract factor failed."),
                    return FAILED);
  // add link from reshape node to the output node of truedivNode
  for (auto outDataAnchor : truedivNode->GetAllOutDataAnchors()) {
    for (auto inDataAnchor : outDataAnchor->GetPeerInDataAnchors()) {
      ret = ge::GraphUtils::RemoveEdge(outDataAnchor, inDataAnchor);
      FUSION_PASS_CHECK(ret != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                       "Remove edge between node %s. and node %s failed.",
                                truedivNode->GetName().c_str(), inDataAnchor->GetOwnerNode()->GetName().c_str()),
                        return ret);
      ret = ge::GraphUtils::AddEdge(reshapeNode->GetOutDataAnchor(0), inDataAnchor);
      FUSION_PASS_CHECK(ret != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                       "Add edge between node %s. and node %s failed.",
                                reshapeNode->GetName().c_str(), inDataAnchor->GetOwnerNode()->GetName().c_str()),
                        return ret);
    }
  }

  ret = graph.RemoveNode(tileNode);
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove tileNode failed."),
                    return ret);
  ret = graph.RemoveNode(truedivNode);
  FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove truedivNode failed."),
                    return ret);
  reshapeNode->GetOpDesc()->SetType("MeanGrad");

  ge::AttrUtils::SetInt(reshapeNode->GetOpDesc(), MEAN_GRAD_OUTPUT_SHAPE_FORMAT, ge::FORMAT_NHWC);

  vector<int64_t> shapeValue;
  shapeValue.push_back(outputN);
  shapeValue.push_back(outputH);
  shapeValue.push_back(outputW);
  shapeValue.push_back(outputC);

  ge::AttrUtils::SetListInt(reshapeNode->GetOpDesc(), MEAN_GRAD_OUTPUT_SHAPE_VALUE, shapeValue);
  // update input and output tensor_desc
  ge::GeTensorDesc inputDesc = reshapeNode->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc outputDesc = reshapeNode->GetOpDesc()->GetOutputDesc(0);
  inputDesc.SetFormat(ge::FORMAT_NHWC);
  inputDesc.SetDataType(ge::DT_FLOAT);
  inputDesc.SetOriginFormat(ge::FORMAT_NHWC);
  outputDesc.SetFormat(ge::FORMAT_NHWC);
  outputDesc.SetDataType(ge::DT_FLOAT);
  outputDesc.SetShape(GeShape(shapeValue));
  outputDesc.SetOriginFormat(ge::FORMAT_NHWC);
  FUSION_PASS_CHECK(reshapeNode->GetOpDesc()->UpdateInputDesc(0, inputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Update inputDesc failed!"), return FAILED);
  FUSION_PASS_CHECK(reshapeNode->GetOpDesc()->UpdateOutputDesc(0, outputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Update outputDesc failed!"), return FAILED);
  ge::GeTensorDesc inputTensorDesc;
  ge::GeTensorDesc outputTensorDesc;
  ge::AttrUtils::GetTensorDesc(reshapeNode->GetOpDesc(), "input_desc_reshape", inputTensorDesc);
  ge::AttrUtils::GetTensorDesc(reshapeNode->GetOpDesc(), "output_desc_reshape", outputTensorDesc);
  inputTensorDesc.SetFormat(ge::FORMAT_NHWC);
  inputTensorDesc.SetOriginFormat(ge::FORMAT_NHWC);
  outputTensorDesc.SetFormat(ge::FORMAT_NHWC);
  outputTensorDesc.SetOriginFormat(ge::FORMAT_NHWC);
  ge::AttrUtils::SetTensorDesc(reshapeNode->GetOpDesc(), "input_desc_reshape", inputTensorDesc);
  ge::AttrUtils::SetTensorDesc(reshapeNode->GetOpDesc(), "output_desc_reshape", outputTensorDesc);

  fusionNodes.push_back(reshapeNode);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "MeanGradFusionPass success!");
  return SUCCESS;
}

// Delete the constant input node connected to the specified node
// (if the constant node is also connected to other nodes, only the edge is deleted)
// Since the Const node of the training network is converted to CONSTANTOP,
// when the Remove node is called, the Const input node cannot be automatically deleted
// You must delete the Const node of the node by yourself
Status MeanGradFusionPass::RemoveConstOpInput(ge::ComputeGraph& graph, ge::NodePtr node) {
  FUSION_PASS_CHECK(node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "node is nullptr."),
                    return FAILED);
  for (auto inDataAnchor : node->GetAllInDataAnchors()) {
    int idx = inDataAnchor->GetIdx();
    auto outDataAnchor = inDataAnchor->GetPeerOutAnchor();
    if (outDataAnchor == nullptr || outDataAnchor->GetOwnerNode() == nullptr) {
      continue;
    }
    string nodeType = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(outDataAnchor->GetOwnerNode());
    if (nodeType == "Constant" || nodeType == "Const") {
      Status ret;
      if (outDataAnchor->GetOwnerNode()->GetOutDataNodes().size() == 1) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "Remove const op %s.", outDataAnchor->GetOwnerNode()->GetName().c_str());
        ge::NodeUtils::ClearInDataAnchor(node, inDataAnchor);
        ge::OpDescUtils::ClearInputDesc(node->GetOpDesc(), (uint32_t)idx);
        ret = graph.RemoveNode(outDataAnchor->GetOwnerNode());
        FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                          "Remove const node failed"),
                          return ret);
      } else {
        ret = ge::GraphUtils::RemoveEdge(outDataAnchor, inDataAnchor);
        ge::NodeUtils::ClearInDataAnchor(node, inDataAnchor);
        ge::OpDescUtils::ClearInputDesc(node->GetOpDesc(), (uint32_t)idx);
        FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                          "Remove edge from const op failed"),
                          return ret);
      }
    }
  }
  return SUCCESS;
}

Status MeanGradFusionPass::ParseParaFromConst(ge::NodePtr node, int32_t& param, int index) {
  string nodeType = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(node);
  if (nodeType == "Const" || nodeType == "Constant") {
    vector<ge::ConstGeTensorPtr> weights_vec = ge::OpDescUtils::GetWeights(node);
    FUSION_PASS_CHECK(weights_vec.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get weights failed"),
                      return PARAM_INVALID);
    const ge::GeTensor* tensor = weights_vec[0].get();
    // get data from tensor
    int64_t dataType = 0;
    FUSION_PASS_CHECK(!ge::AttrUtils::GetInt(node->GetOpDesc(), "dtype", dataType),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get dtype attr failed."),
                                                     return PARAM_INVALID);
    if (dataType == ge::DT_INT32) {
      FUSION_PASS_CHECK(tensor->GetData().size() < (sizeof(int32_t) * (index + 1)),
                        OP_LOGI(FUSED_OP_TYPE.c_str(), "data size too small, may not support, just not changed."),
                        return PARAM_INVALID);
      FUSION_PASS_CHECK(tensor->GetData().data() == nullptr,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "tensor->GetData().data() is a null."),
                        return FAILED);
      param = *((int32_t*)tensor->GetData().data() + index);
    } else {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "data type not supported");
      return FAILED;
    }
    return SUCCESS;
  }
  return FAILED;
}

REGISTER_PASS("MeanGradFusion", BUILT_IN_GRAPH_PASS, MeanGradFusionPass);
}  // namespace fe
