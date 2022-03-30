/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
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
 * \file double_ascend_antiquant_add_relu_ascend_quant_fusion_pass.cc
 * \brief
 *   AscendAntiQuant AscendAntiQuant
 *                \   /
 *                 Add
 *                  |
 *                 Relu
 *                  |
 *              AscendQuant
 *                                                 Cast
 *                             Cast                 |
 *                              |                  Adds
 *  AscendAntiQuant  ->        Adds       or        |
 *                              |                  Muls
 *                             Muls                 |
 *                                                 Muls
 *
 *
 *                             Muls
 *                              |
 *                             Muls                           Muls
 *                              |                              |
 *  AscendQuant      ->        Adds            or             Adds
 *                              |                              |
 *                 Round / Ceil / Floor / None    Round / Ceil / Floor / None
 *                              |                              |
 *                             Cast                           Cast
 */
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"

#include "double_ascend_antiquant_add_relu_ascend_quant_fusion_pass.h"

namespace fe {
// op type
static const char* QUANT = "AscendQuant";
static const char* ANTIQUANT = "AscendAntiQuant";
static const char* ADD = "Add";
static const char* RELU = "Relu";
static const char* CAST = "Cast";
static const char* ADDS = "Adds";
static const char* MULS = "Muls";
static const char* ROUND = "Round";
static const char* CEIL = "Ceil";
static const char* FLOOR = "Floor";

// node pattern
static const char* PATTERN_QUANT = "AscendQuant";
static const char* PATTERN_ANTIQUANT_1 = "AscendAntiQuant1";
static const char* PATTERN_ANTIQUANT_2 = "AscendAntiQuant2";
static const char* PATTERN_ADD = "Add";
static const char* PATTERN_RELU = "Relu";

// attributes
static const char* ATTR_SCALE = "scale";
static const char* ATTR_OFFSET = "offset";
static const char* ATTR_SQRT_MODE = "sqrt_mode";
static const char* ATTR_ROUND_MODE = "round_mode";
static const char* ATTR_DST_TYPE = "dst_type";
static const char* ATTR_VALUE = "value";

static const char* FUSED_OP_TYPE = "DoubleAscendAntiQuantAddReluAscendQuantFusionPass";

/**
 * @brief : initialize input attribute of quant op
 *
 * @return Status : SUCCESS or FAILED
 */
Status XQuantFusionBasePass::GetInputAttributes() {
  VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(FUSED_OP_TYPE, this->node, FAILED,
                                          "AscendQuant / AscendAntiQuant node is null");

  ge::Operator op = ge::OpDescUtils::CreateOperatorFromNode(this->node);
  if (op.GetAttr(ATTR_SCALE, this->scale) != GRAPH_SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE,
                                   "Get attribute scale of AscendQuant / AscendAntiQuant [%s] operator failed.",
                                   this->node->GetName().c_str());
    return FAILED;
  }

  if (op.GetAttr(ATTR_OFFSET, this->offset) != GRAPH_SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE,
                                   "Get attribute offset of AscendQuant / AscendAntiQuant [%s] operator failed.",
                                   this->node->GetName().c_str());
    return FAILED;
  }

  // sqrt_mode is an optional attribute. fail will be considered as default value: false
  op.GetAttr(ATTR_SQRT_MODE, this->sqrtMode);

  return SUCCESS;
}

/**
 * @brief
 *
 * @param name : name for the new desc
 * @param type : type for the new desc
 * @param inputDesc : input for the new desc
 * @param outputDesc : output for the new desc
 * @return ge::OpDescPtr : created desc for the new node
 */
ge::OpDescPtr XQuantFusionBasePass::CreateOpDesc(const std::string& name, const std::string& type,
                                                 const ge::GeTensorDesc& inputDesc,
                                                 const ge::GeTensorDesc& outputDesc) const {
  ge::OpDescPtr desc = nullptr;
  FUSION_PASS_MAKE_SHARED((desc = std::make_shared<ge::OpDesc>(name, type)), return nullptr);

  if (desc->AddInputDesc("x", inputDesc) != GRAPH_SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "add input x to %s failed.", type.c_str());
    return nullptr;
  }

  if (desc->AddOutputDesc("y", outputDesc) != GRAPH_SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "add input y to %s failed.", type.c_str());
    return nullptr;
  }

  return desc;
}

/**
 * @brief create an `Adds` node and add to graph including pushing to `fusionNodes`
 *
 * @param graph : the graph waiting for pass level optimization
 * @param fusionNodes  : new op nodes after fusion
 * @param name  : name of the new node
 * @param inputDesc : input desc for Adds node
 * @return ge::NodePtr : created node
 */
ge::NodePtr XQuantFusionBasePass::CreateAddsNode(ge::ComputeGraph& graph, vector<ge::NodePtr>& fusionNodes,
                                                 const std::string& name, const ge::GeTensorDesc& inputDesc) const {
  ge::OpDescPtr desc = this->CreateOpDesc(name, ADDS, inputDesc, inputDesc);
  FUSION_PASS_CHECK(desc == nullptr, void(), return nullptr);

  auto ret = ge::AttrUtils::SetFloat(desc, ATTR_VALUE, this->offset);
  FUSION_PASS_CHECK(ret != true,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "set value attribute to Adds failed."),
                    return nullptr);

  ge::NodePtr node = graph.AddNode(desc);
  VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(FUSED_OP_TYPE, node, nullptr, "add fusion node (Adds) to graph failed.");

  fusionNodes.push_back(node);
  return node;
}

/**
 * @brief create a `Muls` node and add to graph including pushing to `fusionNodes`
 *
 * @param graph : the graph waiting for pass level optimization
 * @param fusionNodes  : new op nodes after fusion
 * @param name  : name of the new node
 * @param inputDesc : input desc for the new node
 * @return ge::NodePtr : created node
 */
ge::NodePtr XQuantFusionBasePass::CreateMulsNode(ge::ComputeGraph& graph, vector<ge::NodePtr>& fusionNodes,
                                                 const std::string& name, const ge::GeTensorDesc& inputDesc) const {
  ge::OpDescPtr desc = this->CreateOpDesc(name, MULS, inputDesc, inputDesc);
  FUSION_PASS_CHECK(desc == nullptr, void(), return nullptr);

  auto ret = ge::AttrUtils::SetFloat(desc, ATTR_VALUE, this->scale);
  FUSION_PASS_CHECK(ret != true,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "set value attribute to Muls failed."),
                    return nullptr);

  ge::NodePtr node = graph.AddNode(desc);
  VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(FUSED_OP_TYPE, node, nullptr, "add fusion node (Muls) to graph failed.");

  fusionNodes.push_back(node);
  return node;
}

/**
 * @brief create a `Cast` node and add to graph including pushing to `fusionNodes`
 *
 * @param graph : the graph waiting for pass level optimization
 * @param fusionNodes  : new op nodes after fusion
 * @param name  : name of the new node
 * @param inputDesc : input desc for cast
 * @param outputDesc : output desc for cast
 * @return ge::NodePtr : created node
 */
ge::NodePtr XQuantFusionBasePass::CreateCastNode(ge::ComputeGraph& graph, vector<ge::NodePtr>& fusionNodes,
                                                 const std::string& name,  const ge::GeTensorDesc& inputDesc,
                                                 const ge::GeTensorDesc& outputDesc) const {
  ge::OpDescPtr desc = this->CreateOpDesc(name, CAST, inputDesc, outputDesc);
  FUSION_PASS_CHECK(desc == nullptr, void(), return nullptr);

  auto ret = ge::AttrUtils::SetInt(desc, ATTR_DST_TYPE, outputDesc.GetDataType());
  FUSION_PASS_CHECK(ret != true,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "set dst_type attribute to Cast failed."),
                    return nullptr);

  ge::NodePtr node = graph.AddNode(desc);
  VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(FUSED_OP_TYPE, node, nullptr, "add fusion node (Cast) to graph failed.");
  fusionNodes.push_back(node);
  return node;
}

/**
 * @brief simplely link two node `from` to `to` by `AddEdge`
 *
 * @param from : from node
 * @param to : to node
 * @return Status : SUCCESS / FAILED
 */
Status XQuantFusionBasePass::LinkNode(const ge::NodePtr from, const ge::NodePtr to) const {
  VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(FUSED_OP_TYPE, from, FAILED, "from node is null !");
  VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(FUSED_OP_TYPE, to, FAILED, "to node is null !");

  if (ge::GraphUtils::AddEdge(from->GetOutDataAnchor(0), to->GetInDataAnchor(0)) != ge::GRAPH_SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "Failed to add edge between %s and %s.",
                                   from->GetName().c_str(), to->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

/**
 * @brief replace one node with two nodes in graph
 *
 * @param originalNode : original node to be replaced
 * @param newFirstNode : head node to replace
 * @param newLastNode : tail node to replace
 * @return Status : SUCCESS / FAILED
 */
Status XQuantFusionBasePass::ReplaceNode(const ge::NodePtr originalNode,
                                         const ge::NodePtr newFirstNode, const ge::NodePtr newLastNode) const {
  VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(FUSED_OP_TYPE, originalNode, FAILED, "originalNode node is null !");
  VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(FUSED_OP_TYPE, newFirstNode, FAILED, "newFirstNode node is null !");
  VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(FUSED_OP_TYPE, newLastNode, FAILED, "newLastNode node is null !");

  // input link
  auto inAnchor = originalNode->GetInDataAnchor(0);
  VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(FUSED_OP_TYPE, inAnchor, FAILED,
                                          "Failed to get input anchor of %s.", originalNode->GetName().c_str());

  auto peerOutAnchor = inAnchor->GetPeerOutAnchor();
  if (ge::GraphUtils::RemoveEdge(peerOutAnchor, inAnchor) != ge::GRAPH_SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "Failed to remove input edge of %s.",
                                   originalNode->GetName().c_str());
    return FAILED;
  }

  if (ge::GraphUtils::AddEdge(peerOutAnchor, newFirstNode->GetInDataAnchor(0)) != ge::GRAPH_SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "Failed to add edge between input and %s.",
                                   newFirstNode->GetName().c_str());
    return FAILED;
  }

  // output link
  auto outAnchor = originalNode->GetOutDataAnchor(0);
  VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(FUSED_OP_TYPE, outAnchor, FAILED,
                                          "Failed to get output anchor of %s.", originalNode->GetName().c_str());

  for (ge::InDataAnchorPtr& peerInAnchor : outAnchor->GetPeerInDataAnchors()) {
    if (ge::GraphUtils::RemoveEdge(outAnchor, peerInAnchor) != ge::GRAPH_SUCCESS) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "Failed to remove output edge of %s.",
                                     originalNode->GetName().c_str());
      return FAILED;
    }

    if (ge::GraphUtils::AddEdge(newLastNode->GetOutDataAnchor(0), peerInAnchor) != ge::GRAPH_SUCCESS) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "Failed to add output edge of %s.",
                                     newLastNode->GetName().c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

/**
 * @brief whether match fussion scenario
 *
 * @return true: match, false: not match
 */
bool AscendAntiQuantFusionPass::IsMatch() const {
  // check output link
  auto outAnchor = this->node->GetOutDataAnchor(0);
  VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(FUSED_OP_TYPE, outAnchor, false,
                                          "Get output anchor of AscendAntiQuant failed.");

  size_t outputPeerSize = outAnchor->GetPeerAnchorsSize();
  if (outputPeerSize != 1) {
    OP_LOGI(FUSED_OP_TYPE, "AscendAntiQuant node [%s] output size is [%zu], which should be 1.",
            this->node->GetName().c_str(), outputPeerSize);
    return false;
  }

  return true;
}

/**
 * @brief fusion procedure for AscendAntiQuant
 *
 * @param graph : the graph waiting for pass level optimization
 * @param fusionNodes : new op nodes after fusion
 * @return Status SUCCESS / FAILED
 */
Status AscendAntiQuantFusionPass::Fusion(ge::ComputeGraph& graph, vector<ge::NodePtr>& fusionNodes) {
  // create new node: cast / adds / muls
  auto ret = this->CreateNodes(graph, fusionNodes);
  FUSION_PASS_CHECK(ret != SUCCESS, void(), return FAILED);

  // add / remove edge
  ret = this->UpdateEdges();
  FUSION_PASS_CHECK(ret != SUCCESS, void(), return FAILED);

  if (graph.RemoveNode(this->node) != SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "Remove AscendAntiQuant [%s] from graph failed.",
                                   this->node->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

/**
 * @brief create new node: cast / adds / muls
 *
 * @return Status : SUCCESS / FAILED
 */
Status AscendAntiQuantFusionPass::CreateNodes(ge::ComputeGraph& graph, vector<ge::NodePtr>& fusionNodes) {
  const std::string antiQuantNodeName = this->node->GetName();

  ge::OpDescPtr opDesc = this->node->GetOpDesc();
  VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(FUSED_OP_TYPE, opDesc, FAILED,
                                          "Try to get operation desc of AscendAntiQuant failed.");

  auto xDesc = opDesc->GetInputDesc(0).Clone();
  auto yDesc = opDesc->GetOutputDesc(0).Clone();
  yDesc.SetShape(xDesc.GetShape());

  // create new node: cast / adds / muls
  this->castNode = this->CreateCastNode(graph, fusionNodes, antiQuantNodeName + "_cast", xDesc, yDesc);
  FUSION_PASS_CHECK(this->castNode == nullptr, void(), return FAILED);

  this->addsNode = this->CreateAddsNode(graph, fusionNodes, antiQuantNodeName + "_adds", yDesc);
  FUSION_PASS_CHECK(this->addsNode == nullptr, void(), return FAILED);

  this->mulsNode1 = this->CreateMulsNode(graph, fusionNodes, antiQuantNodeName + "_muls1", yDesc);
  FUSION_PASS_CHECK(this->mulsNode1 == nullptr, void(), return FAILED);

  if (this->sqrtMode == true) {
    this->mulsNode2 = this->CreateMulsNode(graph, fusionNodes, antiQuantNodeName + "_muls2", yDesc);
    FUSION_PASS_CHECK(this->mulsNode2 == nullptr, void(), return FAILED);
  }

  return SUCCESS;
}

/**
 * @brief Add / remove edges between nodes
 *
 * @return Status  : SUCCESS / FAILED
 */
Status AscendAntiQuantFusionPass::UpdateEdges() const {
  // link output of `Cast` to `Adds`
  auto ret = this->LinkNode(this->castNode, this->addsNode);
  FUSION_PASS_CHECK(ret != SUCCESS, void(), return FAILED);

  // link output of `Adds` to `Muls`(1)
  ret = this->LinkNode(this->addsNode, this->mulsNode1);
  FUSION_PASS_CHECK(ret != SUCCESS, void(), return FAILED);

  ge::NodePtr lastNode = this->mulsNode1;
  // link output of `Muls`(1) to `Muls`(2) if needed
  if (this->sqrtMode == true) {
    ret = this->LinkNode(this->mulsNode1, this->mulsNode2);
    FUSION_PASS_CHECK(ret != SUCCESS, void(), return FAILED);
    lastNode = this->mulsNode2;
  }

  // link input node of `AscendAntiQuant` to `Cast`
  // link output of `Muls` to ouput nodes of `AscendAntiQuant`
  ret = this->ReplaceNode(this->node, this->castNode, lastNode);
  FUSION_PASS_CHECK(ret != SUCCESS, void(), return FAILED);

  return SUCCESS;
}

/**
 * @brief : initialize input attribute of AscendQuant op
 *
 * @return Status : SUCCESS or FAILED
 */
Status AscendQuantFusionPass::GetInputAttributes() {
  if (XQuantFusionBasePass::GetInputAttributes() != SUCCESS) {
    return FAILED;
  }

  ge::Operator op = ge::OpDescUtils::CreateOperatorFromNode(this->node);
  op.GetAttr(ATTR_ROUND_MODE, this->roundMode);
  op.GetAttr(ATTR_DST_TYPE, this->dstType);

  return SUCCESS;
}

/**
 * @brief whether match fussion scenario
 *
 * @return true: match, false: not match
 */
bool AscendQuantFusionPass::IsMatch() const {
  // dst_type should be 2, which means int8
  if (this->dstType != ge::DataType::DT_INT8) {
    OP_LOGI(FUSED_OP_TYPE,
            "Attribute dst_type of AscendQuant [%s] is [%ld], which should be 2 (INT8).",
            this->node->GetName().c_str(), this->dstType);
    return false;
  }

  return true;
}

/**
 * @brief create a `Round` / `Ceil` / `Floor` node and add to graph including pushing to `fusionNodes`
 *
 * @param graph : the graph waiting for pass level optimization
 * @param fusionNodes  : new op nodes after fusion
 * @param name  : name of the new node
 * @param inputDesc : input desc for the new node
 * @return ge::NodePtr : created node
 */
ge::NodePtr AscendQuantFusionPass::CreateConvertNode(ge::ComputeGraph& graph, vector<ge::NodePtr>& fusionNodes,
                                                     const std::string& name,
                                                     const ge::GeTensorDesc& inputDesc) const {
  const char* opType = ROUND;
  if (this->roundMode == "Ceil") {
    opType = CEIL;
  } else if (this->roundMode == "Floor") {
    opType = FLOOR;
  }

  ge::OpDescPtr desc = this->CreateOpDesc(name, opType, inputDesc, inputDesc);
  FUSION_PASS_CHECK(desc == nullptr, void(), return nullptr);

  ge::NodePtr node = graph.AddNode(desc);
  VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(FUSED_OP_TYPE, node, nullptr,
                                          "add fusion node (Round / Ceil / Floor) to graph failed.");

  fusionNodes.push_back(node);
  return node;
}

/**
 * @brief fusion procedure for AscendQuant
 *
 * @param graph : the graph waiting for pass level optimization
 * @param fusionNodes : new op nodes after fusion
 * @return Status SUCCESS / FAILED
 */
Status AscendQuantFusionPass::Fusion(ge::ComputeGraph& graph, vector<ge::NodePtr>& fusionNodes) {
  // create new node: convert / cast / adds / muls
  auto ret = this->CreateNodes(graph, fusionNodes);
  FUSION_PASS_CHECK(ret != SUCCESS, void(), return FAILED);

  // add / remove edge
  ret = this->UpdateEdges();
  FUSION_PASS_CHECK(ret != SUCCESS, void(), return FAILED);

  if (graph.RemoveNode(this->node) != SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "Remove AscendQuant [%s] from graph failed.",
                                   this->node->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

/**
 * @brief create new node: convert / cast / adds / muls
 *
 * @return Status : SUCCESS / FAILED
 */
Status AscendQuantFusionPass::CreateNodes(ge::ComputeGraph& graph, vector<ge::NodePtr>& fusionNodes) {
  const std::string quantNodeName = this->node->GetName();

  ge::OpDescPtr opDesc = this->node->GetOpDesc();
  VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(FUSED_OP_TYPE, opDesc, FAILED,
                                          "Try to get operation desc of AscendQuant failed.");

  auto xDesc = opDesc->GetInputDesc(0).Clone();
  auto yDesc = opDesc->GetOutputDesc(0).Clone();
  xDesc.SetShape(yDesc.GetShape());

  // create new node: convert / cast / adds / muls
  this->mulsNode1 = this->CreateMulsNode(graph, fusionNodes, quantNodeName + "_muls1", xDesc);
  FUSION_PASS_CHECK(this->mulsNode1 == nullptr, void(), return FAILED);

  if (this->sqrtMode == true) {
    this->mulsNode2 = this->CreateMulsNode(graph, fusionNodes, quantNodeName + "_muls2", xDesc);
    FUSION_PASS_CHECK(this->mulsNode2 == nullptr, void(), return FAILED);
  }

  this->addsNode = this->CreateAddsNode(graph, fusionNodes, quantNodeName + "_adds", xDesc);
  FUSION_PASS_CHECK(this->addsNode == nullptr, void(), return FAILED);

  if (this->roundMode != "Trunc") {
    this->convertNode = this->CreateConvertNode(graph, fusionNodes, quantNodeName + "_conv", xDesc);
    FUSION_PASS_CHECK(this->convertNode == nullptr, void(), return FAILED);
  }

  this->castNode = this->CreateCastNode(graph, fusionNodes, quantNodeName + "_cast", xDesc, yDesc);
  FUSION_PASS_CHECK(this->castNode == nullptr, void(), return FAILED);

  return SUCCESS;
}

/**
 * @brief Add / remove edges between nodes
 *
 * @return Status  : SUCCESS / FAILED
 */
Status AscendQuantFusionPass::UpdateEdges() const {
  Status ret = SUCCESS;
  // link output of `Muls`(1) to `Muls`(2) if needed
  ge::NodePtr lastMulsNode = this->mulsNode1;
  if (this->sqrtMode == true) {
    ret = this->LinkNode(this->mulsNode1, this->mulsNode2);
    FUSION_PASS_CHECK(ret != SUCCESS, void(), return FAILED);
    lastMulsNode = this->mulsNode2;
  }

  // link output of `Muls` to `Adds`
  ret = this->LinkNode(lastMulsNode, this->addsNode);
  FUSION_PASS_CHECK(ret != SUCCESS, void(), return FAILED);

  // link output of `Adds` to `Round / Ceil / Floor`
  ge::NodePtr nodeBeforCast = this->addsNode;
  if (this->convertNode != nullptr) {
    ret = this->LinkNode(this->addsNode, this->convertNode);
    FUSION_PASS_CHECK(ret != SUCCESS, void(), return FAILED);
    nodeBeforCast = this->convertNode;
  }

  // link output to `Cast`
  ret = this->LinkNode(nodeBeforCast, this->castNode);
  FUSION_PASS_CHECK(ret != SUCCESS, void(), return FAILED);

  // link input node of `AscendQuant` to `Muls`
  // link output of `Cast` to ouput nodes of `AscendQuant`
  ret = this->ReplaceNode(this->node, this->mulsNode1, this->castNode);
  FUSION_PASS_CHECK(ret != SUCCESS, void(), return FAILED);

  return SUCCESS;
}

/**
 * @brief Define fussion scenario patterns
 *
 * @return vector<FusionPattern*> defined pattern
 */
vector<FusionPattern*> DoubleAscendAntiQuantAddReluAscendQuantFusionPass::DefinePatterns() {
  OP_LOGD(FUSED_OP_TYPE, "Define pattern begin");
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern(FUSED_OP_TYPE);
  VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(FUSED_OP_TYPE, pattern, patterns,
                                          "New a pattern object failed.");

  pattern->AddOpDesc(PATTERN_ANTIQUANT_1, {ANTIQUANT})
      .AddOpDesc(PATTERN_ANTIQUANT_2, {ANTIQUANT})
      .AddOpDesc(PATTERN_ADD, {ADD})
      .AddOpDesc(PATTERN_RELU, {RELU})
      .AddOpDesc(PATTERN_QUANT, {QUANT})
      .SetInputs(PATTERN_ADD, {PATTERN_ANTIQUANT_1, PATTERN_ANTIQUANT_2})
      .SetInputs(PATTERN_RELU, {PATTERN_ADD})
      .SetInputs(PATTERN_QUANT, {PATTERN_RELU})
      .SetOutput(PATTERN_QUANT);

  patterns.push_back(pattern);

  OP_LOGD(FUSED_OP_TYPE, "Define pattern end");
  return patterns;
}

/**
 * @brief fusion procedure
 *
 * @param graph : the graph waiting for pass level optimization
 * @param mapping : result matched by patterns defined in DefinePatterns()
 * @param fusionNodes : new op nodes after fusion
 * @return Status :  SUCCESS / NOT_CHANGED / FAILED
 */
Status DoubleAscendAntiQuantAddReluAscendQuantFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                                 vector<ge::NodePtr>& fusionNodes) {
  OP_LOGD(FUSED_OP_TYPE, "Fusion begin");

  this->antiQuantFP1.SetNode(GetNodeFromMapping(PATTERN_ANTIQUANT_1, mapping));
  this->antiQuantFP2.SetNode(GetNodeFromMapping(PATTERN_ANTIQUANT_2, mapping));
  this->quantFP.SetNode(GetNodeFromMapping(PATTERN_QUANT, mapping));
  this->addNode = GetNodeFromMapping(PATTERN_ADD, mapping);
  this->reluNode = GetNodeFromMapping(PATTERN_RELU, mapping);

  // check
  auto match = this->IsMatch();
  FUSION_PASS_CHECK(!match, void(), return NOT_CHANGED);

  // dynamic shape not supported currently
  auto to_check_nodes = {this->antiQuantFP1.GetNode(), this->antiQuantFP2.GetNode()};
  NOT_CHANGED_WITH_DYNAMIC_NODE(to_check_nodes);

  // do fussion:
  auto ret = this->antiQuantFP1.Fusion(graph, fusionNodes);
  FUSION_PASS_CHECK(ret != SUCCESS, void(), return FAILED);

  ret = this->antiQuantFP2.Fusion(graph, fusionNodes);
  FUSION_PASS_CHECK(ret != SUCCESS, void(), return FAILED);

  ret = this->quantFP.Fusion(graph, fusionNodes);
  FUSION_PASS_CHECK(ret != SUCCESS, void(), return FAILED);

  // change input/output shape for original Add / Relu node
  ret = this->ChangeAddReluShape();
  FUSION_PASS_CHECK(ret != SUCCESS, void(), return FAILED);

  OP_LOGD(FUSED_OP_TYPE, "Fusion end");
  return SUCCESS;
}

/**
 * @brief check the pattern matched subgraph
 *
 * @return true: match, false: not match
 */
bool DoubleAscendAntiQuantAddReluAscendQuantFusionPass::IsMatch() {
  VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(FUSED_OP_TYPE, this->addNode, false,
                                          "Add node is null, will not fusion.");
  VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(FUSED_OP_TYPE, this->reluNode, false,
                                          "Relu node is null, will not fusion.");

  // get input attribute
  FUSION_PASS_CHECK(this->antiQuantFP1.GetInputAttributes() != SUCCESS, void(), return false);
  FUSION_PASS_CHECK(this->antiQuantFP2.GetInputAttributes() != SUCCESS, void(), return false);
  FUSION_PASS_CHECK(this->quantFP.GetInputAttributes() != SUCCESS, void(), return false);

  // check AscendAntiQuant output link
  FUSION_PASS_CHECK(!this->antiQuantFP1.IsMatch(), void(), return false);
  FUSION_PASS_CHECK(!this->antiQuantFP2.IsMatch(), void(), return false);

  // check AscendQuant dst_type
  FUSION_PASS_CHECK(!this->quantFP.IsMatch(), void(), return false);

  // check add / relu output link
  FUSION_PASS_CHECK(!this->IsAddReluOutputMatch(), void(), return false);

  return true;
}

/**
 * @brief ensure that there is only one output link for add / relu op
 *
 * @return true : match / false: no
 */
bool DoubleAscendAntiQuantAddReluAscendQuantFusionPass::IsAddReluOutputMatch() const {
  size_t outputPeerSize = this->addNode->GetOutDataAnchor(0)->GetPeerAnchorsSize();
  if (outputPeerSize != 1) {
    OP_LOGI(FUSED_OP_TYPE, "Add node [%s] output size is [%zu], which should be 1.",
            this->addNode->GetName().c_str(), outputPeerSize);
    return false;
  }

  outputPeerSize = this->reluNode->GetOutDataAnchor(0)->GetPeerAnchorsSize();
  if (outputPeerSize != 1) {
    OP_LOGI(FUSED_OP_TYPE, "Relu node [%s] output size is [%zu], which should be 1.",
            this->reluNode->GetName().c_str(), outputPeerSize);
    return false;
  }

  return true;
}

/**
 * @brief change input / output shape of Add / Relu to C0 = 32
 *
 * @return Status : SUCCESS / FAILED
 */
Status DoubleAscendAntiQuantAddReluAscendQuantFusionPass::ChangeAddReluShape() {
  ge::NodePtr antiQuantNode = this->antiQuantFP1.GetNode();
  VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(FUSED_OP_TYPE, antiQuantNode, FAILED,
                                          "AscendAntiQuant node 1 is null, which should never happen.");

  ge::OpDescPtr antiQuantOpDesc = antiQuantNode->GetOpDesc();
  VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(FUSED_OP_TYPE, antiQuantOpDesc, FAILED,
                                          "Try to get operation desc of AscendAntiQuant node failed.");
  auto xDesc = antiQuantOpDesc->GetInputDesc(0).Clone();
  auto shape = xDesc.GetShape();

  ge::OpDescPtr addOpDesc = this->addNode->GetOpDesc();
  VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(FUSED_OP_TYPE, addOpDesc, FAILED,
                                          "Try to get operation desc of Add node failed.");
  addOpDesc->MutableInputDesc(0)->SetShape(shape);
  addOpDesc->MutableInputDesc(1)->SetShape(shape);
  addOpDesc->MutableOutputDesc(0)->SetShape(shape);

  ge::OpDescPtr reluOpDesc = this->reluNode->GetOpDesc();
  VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(FUSED_OP_TYPE, reluOpDesc, FAILED,
                                          "Try to get operation desc of Relu node failed.");
  reluOpDesc->MutableInputDesc(0)->SetShape(shape);
  reluOpDesc->MutableOutputDesc(0)->SetShape(shape);

  return SUCCESS;
}

REGISTER_PASS(FUSED_OP_TYPE, SECOND_ROUND_BUILT_IN_GRAPH_PASS,
              DoubleAscendAntiQuantAddReluAscendQuantFusionPass);
}  // namespace fe
