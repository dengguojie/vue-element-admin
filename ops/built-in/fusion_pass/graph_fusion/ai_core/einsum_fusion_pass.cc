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
 * \file einsum_fusion_pass.cc
 * \brief
 */
#include "einsum_fusion_pass.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe {
static const string PATTERN_FUSED_NODE = "EinSum";
static const string EINSUM = "EinSum";
static const string RESHAPE = "Reshape";
static const string TRANSPOSED = "TransposeD";
static const string MATMUL = "MatMulV2";
static const string BATCHMATMUL = "BatchMatMul";

/*
    einsum -> reshape + matmul + reshape
    einsum -> transpose + batchmatmul + transpose
*/

vector<FusionPattern*> EinSumPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (nothrow) FusionPattern("EinSumPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSED_NODE, {EINSUM}).SetOutput(PATTERN_FUSED_NODE);
  patterns.push_back(pattern);
  return patterns;
}

// vector<NodePtr> &fusion_nodes: Store fusion nodes,
//       including newly added nodes and fused but not deleted nodes
Status EinSumPass::Fusion(ComputeGraph& graph, Mapping& mapping, vector<NodePtr>& fusion_nodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "EinSumPass fusion in!");
  // get node
  NodePtr node = GetNodeFromMapping(PATTERN_FUSED_NODE, mapping);
  FUSION_PASS_CHECK(node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "einsum node is null, fusion failed."),
                    return PARAM_INVALID);
  // get op desc
  OpDescPtr op_desc = node->GetOpDesc();
  FUSION_PASS_CHECK(op_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "einsum desc is null, fusion failed."),
                    return PARAM_INVALID);

  // check input link relation
  FUSION_PASS_CHECK(node->GetInDataNodes().size() != 2,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Input node of einsum node size is [%lu], which not equal to 2.",
                            node->GetInDataNodes().size()),
                    return NOT_CHANGED);

  // get input
  GeTensorDesc x0_desc = op_desc->GetInputDesc(0);
  GeShape x0_shape = x0_desc.GetShape();
  std::vector<int64_t> x0_dims = x0_shape.GetDims();
  GeTensorDesc x1_desc = op_desc->GetInputDesc(1);
  GeShape x1_shape = x1_desc.GetShape();
  std::vector<int64_t> x1_dims = x1_shape.GetDims();

  // check dynamic shape
  FUSION_PASS_CHECK(IsUnknownShape(x0_dims) || IsUnknownShape(x1_dims),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "EinSum is dynamic."), return NOT_CHANGED);

  // get attr equation
  std::string equation;
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  if (op.GetAttr("equation", equation) != GRAPH_SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Get attr equation failed.");
    return NOT_CHANGED;
  }

  // common vars
  std::vector<int64_t> tmp_dims;
  auto x0_anchor = node->GetInDataAnchor(0);
  auto x0_anchor_peer_anchor = x0_anchor->GetPeerOutAnchor();
  auto x0_anchor_peer_node = x0_anchor_peer_anchor->GetOwnerNode();
  auto x1_anchor = node->GetInDataAnchor(1);
  auto x1_anchor_peer_anchor = x1_anchor->GetPeerOutAnchor();
  auto x1_anchor_peer_node = x1_anchor_peer_anchor->GetOwnerNode();
  auto out_anchor = node->GetOutDataAnchor(0);
  auto out_anchor_peer_anchors = out_anchor->GetPeerInDataAnchors();

  // select equation model
  if (equation == "abc,cde->abde") {
    FUSION_PASS_CHECK((x0_dims.size() != 3) && (x1_dims.size() != 3),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "input dims size must be three and three."), return NOT_CHANGED);
    // create matmul op desc
    std::shared_ptr<ge::OpDesc> matmul_desc = nullptr;
    matmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/MatMul", MATMUL);
    // add matmul op input and output desc
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0] * x0_dims[1]);
    tmp_dims.push_back(x0_dims[2]);
    x0_desc.SetShape(GeShape(tmp_dims));
    x0_desc.SetOriginShape(GeShape(tmp_dims));
    matmul_desc->AddInputDesc("x1", x0_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x1_dims[0]);
    tmp_dims.push_back(x1_dims[1] * x1_dims[2]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    matmul_desc->AddInputDesc("x2", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x1_dims[1]);
    tmp_dims.push_back(x1_dims[2]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    matmul_desc->AddOutputDesc("y", x1_desc);
    // create matmul op
    NodePtr matmul_node = graph.AddNode(matmul_desc);
    // set matmul op attr
    Operator matmul_op = OpDescUtils::CreateOperatorFromNode(matmul_node);
    matmul_op.SetAttr("transpose_x1", false);
    matmul_op.SetAttr("transpose_x2", false);
    // unlink
    for (auto inAnchor : node->GetAllInDataAnchors()) {
      if (inAnchor != nullptr) {
        inAnchor->UnlinkAll();
      }
    }
    for (auto outAnchor : node->GetAllOutDataAnchors()) {
      if (outAnchor != nullptr) {
        outAnchor->UnlinkAll();
      }
    }
    // add edge
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, matmul_node->GetInDataAnchor(0)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x0_anchor_peer_node->GetName().c_str(), matmul_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, matmul_node->GetInDataAnchor(1)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x1_anchor_peer_node->GetName().c_str(), matmul_node->GetName().c_str()),
                      return FAILED);
    for (uint64_t i = 0; i < out_anchor_peer_anchors.size(); ++i) {
      auto out_anchor_peer_anchor = out_anchor_peer_anchors.at(i);
      auto out_anchor_peer_node = out_anchor_peer_anchor->GetOwnerNode();
      FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), out_anchor_peer_anchor),
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                matmul_node->GetName().c_str(), out_anchor_peer_node->GetName().c_str()),
                        return FAILED);
    }
    // remove node
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "remove einsum node failed"), return FAILED);
  } else if (equation == "BTNH,BFNH->BNFT") {
    FUSION_PASS_CHECK((x0_dims.size() != 4) && (x1_dims.size() != 4),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "input dims size must be four and four."), return NOT_CHANGED);
    // create transposed op desc
    std::shared_ptr<ge::OpDesc> transpose_1_desc = nullptr;
    transpose_1_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Transpose1", TRANSPOSED);
    std::shared_ptr<ge::OpDesc> transpose_2_desc = nullptr;
    transpose_2_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Transpose2", TRANSPOSED);
    std::shared_ptr<ge::OpDesc> transpose_3_desc = nullptr;
    transpose_3_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Transpose3", TRANSPOSED);
    // create batchmatmulv2 op desc
    std::shared_ptr<ge::OpDesc> matmul_desc = nullptr;
    matmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/BatchMatMul", BATCHMATMUL);
    // add input and output desc
    transpose_1_desc->AddInputDesc("x", x0_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[2]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x0_dims[3]);
    x0_desc.SetShape(GeShape(tmp_dims));
    x0_desc.SetOriginShape(GeShape(tmp_dims));
    transpose_1_desc->AddOutputDesc("y", x0_desc);
    matmul_desc->AddInputDesc("x1", x0_desc);
    transpose_2_desc->AddInputDesc("x", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x1_dims[0]);
    tmp_dims.push_back(x1_dims[2]);
    tmp_dims.push_back(x1_dims[3]);
    tmp_dims.push_back(x1_dims[1]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    transpose_2_desc->AddOutputDesc("y", x1_desc);
    matmul_desc->AddInputDesc("x2", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[2]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x1_dims[1]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    matmul_desc->AddOutputDesc("y", x1_desc);
    transpose_3_desc->AddInputDesc("x", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[2]);
    tmp_dims.push_back(x1_dims[1]);
    tmp_dims.push_back(x0_dims[1]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    transpose_3_desc->AddOutputDesc("y", x1_desc);
    // create transposed op
    NodePtr transpose_1_node = graph.AddNode(transpose_1_desc);
    NodePtr transpose_2_node = graph.AddNode(transpose_2_desc);
    NodePtr transpose_3_node = graph.AddNode(transpose_3_desc);
    // create batchmatmulv2 op
    NodePtr matmul_node = graph.AddNode(matmul_desc);
    // set op attr
    Operator transpose1_op = OpDescUtils::CreateOperatorFromNode(transpose_1_node);
    tmp_dims.clear();
    tmp_dims.push_back(0);
    tmp_dims.push_back(2);
    tmp_dims.push_back(1);
    tmp_dims.push_back(3);
    transpose1_op.SetAttr("perm", tmp_dims);
    Operator transpose2_op = OpDescUtils::CreateOperatorFromNode(transpose_2_node);
    tmp_dims.clear();
    tmp_dims.push_back(0);
    tmp_dims.push_back(2);
    tmp_dims.push_back(3);
    tmp_dims.push_back(1);
    transpose2_op.SetAttr("perm", tmp_dims);
    Operator transpose3_op = OpDescUtils::CreateOperatorFromNode(transpose_3_node);
    tmp_dims.clear();
    tmp_dims.push_back(0);
    tmp_dims.push_back(1);
    tmp_dims.push_back(3);
    tmp_dims.push_back(2);
    transpose3_op.SetAttr("perm", tmp_dims);
    Operator matmul_op = OpDescUtils::CreateOperatorFromNode(matmul_node);
    matmul_op.SetAttr("adj_x1", false);
    matmul_op.SetAttr("adj_x2", false);
    // unlink
    for (auto inAnchor : node->GetAllInDataAnchors()) {
      if (inAnchor != nullptr) {
        inAnchor->UnlinkAll();
      }
    }
    for (auto outAnchor : node->GetAllOutDataAnchors()) {
      if (outAnchor != nullptr) {
        outAnchor->UnlinkAll();
      }
    }
    // add edge
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, transpose_1_node->GetInDataAnchor(0)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x0_anchor_peer_node->GetName().c_str(), transpose_1_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, transpose_2_node->GetInDataAnchor(0)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x1_anchor_peer_node->GetName().c_str(), transpose_2_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(transpose_1_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0)),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                transpose_1_node->GetName().c_str(), matmul_node->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(transpose_2_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1)),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                transpose_2_node->GetName().c_str(), matmul_node->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), transpose_3_node->GetInDataAnchor(0)),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                matmul_node->GetName().c_str(), transpose_3_node->GetName().c_str()),
        return FAILED);
    for (uint64_t i = 0; i < out_anchor_peer_anchors.size(); ++i) {
      auto out_anchor_peer_anchor = out_anchor_peer_anchors.at(i);
      auto out_anchor_peer_node = out_anchor_peer_anchor->GetOwnerNode();
      FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(transpose_3_node->GetOutDataAnchor(0), out_anchor_peer_anchor),
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                transpose_3_node->GetName().c_str(), out_anchor_peer_node->GetName().c_str()),
                        return FAILED);
    }
    // remove node
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "remove einsum node failed"), return FAILED);
  } else if (equation == "BNFT,BTNH->BFNH") {
    FUSION_PASS_CHECK((x0_dims.size() != 4) && (x1_dims.size() != 4),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "input dims size must be four and four."), return NOT_CHANGED);
    // create transposed op desc
    std::shared_ptr<ge::OpDesc> transpose_1_desc = nullptr;
    transpose_1_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Transpose1", TRANSPOSED);
    std::shared_ptr<ge::OpDesc> transpose_2_desc = nullptr;
    transpose_2_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Transpose2", TRANSPOSED);
    // create batchmatmulv2 op desc
    std::shared_ptr<ge::OpDesc> matmul_desc = nullptr;
    matmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/BatchMatMul", BATCHMATMUL);
    // add input and output desc
    transpose_1_desc->AddInputDesc("x", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x1_dims[0]);
    tmp_dims.push_back(x1_dims[2]);
    tmp_dims.push_back(x1_dims[1]);
    tmp_dims.push_back(x1_dims[3]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    transpose_1_desc->AddOutputDesc("y", x1_desc);
    matmul_desc->AddInputDesc("x1", x0_desc);
    matmul_desc->AddInputDesc("x2", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x0_dims[2]);
    tmp_dims.push_back(x1_dims[3]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    matmul_desc->AddOutputDesc("y", x1_desc);
    transpose_2_desc->AddInputDesc("x", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[2]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x1_dims[3]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    transpose_2_desc->AddOutputDesc("y", x1_desc);
    // create transposed op
    NodePtr transpose_1_node = graph.AddNode(transpose_1_desc);
    NodePtr transpose_2_node = graph.AddNode(transpose_2_desc);
    // create batchmatmulv2 op
    NodePtr matmul_node = graph.AddNode(matmul_desc);
    // set op attr
    Operator transpose1_op = OpDescUtils::CreateOperatorFromNode(transpose_1_node);
    tmp_dims.clear();
    tmp_dims.push_back(0);
    tmp_dims.push_back(2);
    tmp_dims.push_back(1);
    tmp_dims.push_back(3);
    transpose1_op.SetAttr("perm", tmp_dims);
    Operator transpose2_op = OpDescUtils::CreateOperatorFromNode(transpose_2_node);
    tmp_dims.clear();
    tmp_dims.push_back(0);
    tmp_dims.push_back(2);
    tmp_dims.push_back(1);
    tmp_dims.push_back(3);
    transpose2_op.SetAttr("perm", tmp_dims);
    Operator matmul_op = OpDescUtils::CreateOperatorFromNode(matmul_node);
    matmul_op.SetAttr("adj_x1", false);
    matmul_op.SetAttr("adj_x2", false);
    // unlink
    for (auto inAnchor : node->GetAllInDataAnchors()) {
      if (inAnchor != nullptr) {
        inAnchor->UnlinkAll();
      }
    }
    for (auto outAnchor : node->GetAllOutDataAnchors()) {
      if (outAnchor != nullptr) {
        outAnchor->UnlinkAll();
      }
    }
    // add edge
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, matmul_node->GetInDataAnchor(0)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x0_anchor_peer_node->GetName().c_str(), matmul_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, transpose_1_node->GetInDataAnchor(0)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x1_anchor_peer_node->GetName().c_str(), transpose_1_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(transpose_1_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1)),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                transpose_1_node->GetName().c_str(), matmul_node->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), transpose_2_node->GetInDataAnchor(0)),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                matmul_node->GetName().c_str(), transpose_2_node->GetName().c_str()),
        return FAILED);
    for (uint64_t i = 0; i < out_anchor_peer_anchors.size(); ++i) {
      auto out_anchor_peer_anchor = out_anchor_peer_anchors.at(i);
      auto out_anchor_peer_node = out_anchor_peer_anchor->GetOwnerNode();
      FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(transpose_2_node->GetOutDataAnchor(0), out_anchor_peer_anchor),
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                transpose_2_node->GetName().c_str(), out_anchor_peer_node->GetName().c_str()),
                        return FAILED);
    }
    // remove node
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "remove einsum node failed"), return FAILED);
  } else if (equation == "abcd,cde->abe") {
    FUSION_PASS_CHECK((x0_dims.size() != 4) && (x1_dims.size() != 3),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "input dims size must be four and three."), return NOT_CHANGED);
    // create matmul op desc
    std::shared_ptr<ge::OpDesc> matmul_desc = nullptr;
    matmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/MatMul", MATMUL);
    // add matmul op input and output desc
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0] * x0_dims[1]);
    tmp_dims.push_back(x0_dims[2] * x0_dims[3]);
    x0_desc.SetShape(GeShape(tmp_dims));
    x0_desc.SetOriginShape(GeShape(tmp_dims));
    matmul_desc->AddInputDesc("x1", x0_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x1_dims[0] * x1_dims[1]);
    tmp_dims.push_back(x1_dims[2]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    matmul_desc->AddInputDesc("x2", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x1_dims[2]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    matmul_desc->AddOutputDesc("y", x1_desc);
    // create matmul op
    NodePtr matmul_node = graph.AddNode(matmul_desc);
    // set matmul op attr
    Operator matmul_op = OpDescUtils::CreateOperatorFromNode(matmul_node);
    matmul_op.SetAttr("transpose_x1", false);
    matmul_op.SetAttr("transpose_x2", false);
    // unlink
    for (auto inAnchor : node->GetAllInDataAnchors()) {
      if (inAnchor != nullptr) {
        inAnchor->UnlinkAll();
      }
    }
    for (auto outAnchor : node->GetAllOutDataAnchors()) {
      if (outAnchor != nullptr) {
        outAnchor->UnlinkAll();
      }
    }
    // add edge
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, matmul_node->GetInDataAnchor(0)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x0_anchor_peer_node->GetName().c_str(), matmul_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, matmul_node->GetInDataAnchor(1)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x1_anchor_peer_node->GetName().c_str(), matmul_node->GetName().c_str()),
                      return FAILED);
    for (uint64_t i = 0; i < out_anchor_peer_anchors.size(); ++i) {
      auto out_anchor_peer_anchor = out_anchor_peer_anchors.at(i);
      auto out_anchor_peer_node = out_anchor_peer_anchor->GetOwnerNode();
      FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), out_anchor_peer_anchor),
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                matmul_node->GetName().c_str(), out_anchor_peer_node->GetName().c_str()),
                        return FAILED);
    }
    // remove node
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "remove einsum node failed"), return FAILED);
  } else if (equation == "abc,cd->abd") {
    FUSION_PASS_CHECK((x0_dims.size() != 3) && (x1_dims.size() != 2),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "input dims size must be three and two."), return NOT_CHANGED);
    // create matmul op desc
    std::shared_ptr<ge::OpDesc> matmul_desc = nullptr;
    matmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/MatMul", MATMUL);
    // add matmul op input and output desc
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0] * x0_dims[1]);
    tmp_dims.push_back(x0_dims[2]);
    x0_desc.SetShape(GeShape(tmp_dims));
    x0_desc.SetOriginShape(GeShape(tmp_dims));
    matmul_desc->AddInputDesc("x1", x0_desc);
    matmul_desc->AddInputDesc("x2", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x1_dims[1]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    matmul_desc->AddOutputDesc("y", x1_desc);
    // create matmul op
    NodePtr matmul_node = graph.AddNode(matmul_desc);
    // set matmul op attr
    Operator matmul_op = OpDescUtils::CreateOperatorFromNode(matmul_node);
    matmul_op.SetAttr("transpose_x1", false);
    matmul_op.SetAttr("transpose_x2", false);
    // unlink
    for (auto inAnchor : node->GetAllInDataAnchors()) {
      if (inAnchor != nullptr) {
        inAnchor->UnlinkAll();
      }
    }
    for (auto outAnchor : node->GetAllOutDataAnchors()) {
      if (outAnchor != nullptr) {
        outAnchor->UnlinkAll();
      }
    }
    // add edge
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, matmul_node->GetInDataAnchor(0)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x0_anchor_peer_node->GetName().c_str(), matmul_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, matmul_node->GetInDataAnchor(1)),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x1_anchor_peer_node->GetName().c_str(), matmul_node->GetName().c_str()),
                      return FAILED);
    for (uint64_t i = 0; i < out_anchor_peer_anchors.size(); ++i) {
      auto out_anchor_peer_anchor = out_anchor_peer_anchors.at(i);
      auto out_anchor_peer_node = out_anchor_peer_anchor->GetOwnerNode();
      FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), out_anchor_peer_anchor),
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                matmul_node->GetName().c_str(), out_anchor_peer_node->GetName().c_str()),
                        return FAILED);
    }
    // remove node
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "remove einsum node failed"), return FAILED);
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "equation relu is not match.");
    return NOT_CHANGED;
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "EinSumPass fusion success!");
  return SUCCESS;
}
REGISTER_PASS("EinSumPass", BUILT_IN_GRAPH_PASS, EinSumPass);
}  // namespace fe
