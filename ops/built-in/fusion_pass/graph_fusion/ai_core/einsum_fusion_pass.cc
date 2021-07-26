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
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe {
static const string PATTERN_FUSED_NODE = "Einsum";
static const string EINSUM = "Einsum";
static const string RESHAPE = "Reshape";
static const string TRANSPOSE = "TransposeD";
static const string MATMUL = "MatMulV2";
static const string BATCHMATMUL = "BatchMatMul";

/*
    einsum -> reshape + matmul + reshape
    einsum -> transpose + batchmatmul + transpose
*/

static void AssistIntHelp(const vector<int64_t>& const_vec, int32_t* output) {
  for (size_t i = 0; i < const_vec.size(); ++i) {
    output[i] = const_vec[i];
  }
}

vector<FusionPattern*> EinsumPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (nothrow) FusionPattern("EinsumPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSED_NODE, {EINSUM}).SetOutput(PATTERN_FUSED_NODE);
  patterns.push_back(pattern);
  return patterns;
}

// vector<NodePtr> &fusion_nodes: Store fusion nodes,
//       including newly added nodes and fused but not deleted nodes
Status EinsumPass::Fusion(ComputeGraph& graph, Mapping& mapping, vector<NodePtr>& fusion_nodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "EinsumPass fusion in!");
  // get node
  NodePtr node = GetNodeFromMapping(PATTERN_FUSED_NODE, mapping);
  FUSION_PASS_CHECK(node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "einsum node is null, fusion failed."),
                    return PARAM_INVALID);
  // get op desc
  OpDescPtr op_desc = node->GetOpDesc();
  FUSION_PASS_CHECK(op_desc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "einsum desc is null, fusion failed."),
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

  // generate assist
  GeTensorDesc assist_desc;
  assist_desc.SetDataType(DT_INT32);
  assist_desc.SetFormat(FORMAT_ND);
  GeTensorPtr assist_ptr_1 = nullptr;
  GeTensorPtr assist_ptr_2 = nullptr;
  GeTensorPtr assist_ptr_3 = nullptr;
  GeTensorPtr assist_ptr_4 = nullptr;
  GeTensorPtr assist_ptr_5 = nullptr;

  // geneate is_input_const
  vector<bool> is_input_const;
  is_input_const.push_back(false);
  is_input_const.push_back(true);

  // create matmul op desc
  std::shared_ptr<ge::OpDesc> matmul_desc = nullptr;
  matmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/MatMul", MATMUL);
  // create batchmatmulv2 op desc
  std::shared_ptr<ge::OpDesc> batchmatmul_desc = nullptr;
  batchmatmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/BatchMatMul", BATCHMATMUL);
  // create transpose op desc
  std::shared_ptr<ge::OpDesc> transpose_1_desc = nullptr;
  transpose_1_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Transpose1", TRANSPOSE);
  std::shared_ptr<ge::OpDesc> transpose_2_desc = nullptr;
  transpose_2_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Transpose2", TRANSPOSE);
  std::shared_ptr<ge::OpDesc> transpose_3_desc = nullptr;
  transpose_3_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Transpose3", TRANSPOSE);
  // create reshape op desc
  std::shared_ptr<ge::OpDesc> reshape_1_desc = nullptr;
  reshape_1_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Reshape1", RESHAPE);
  std::shared_ptr<ge::OpDesc> reshape_2_desc = nullptr;
  reshape_2_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Reshape2", RESHAPE);
  std::shared_ptr<ge::OpDesc> reshape_3_desc = nullptr;
  reshape_3_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Reshape3", RESHAPE);

  // select equation model
  if (equation == "abc,cde->abde") {  // 001:reshape+reshape+matmul+reshape
    FUSION_PASS_CHECK((x0_dims.size() != 3) && (x1_dims.size() != 3),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "input dims size must be three and three."), return NOT_CHANGED);
    // init const
    unique_ptr<int32_t[]> input_assist_1(new (nothrow) int32_t[2]());
    FUSION_PASS_CHECK(input_assist_1.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_assist is NULL"),
                      return PARAM_INVALID);
    unique_ptr<int32_t[]> input_assist_2(new (nothrow) int32_t[2]());
    FUSION_PASS_CHECK(input_assist_2.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_assist is NULL"),
                      return PARAM_INVALID);
    unique_ptr<int32_t[]> input_assist_3(new (nothrow) int32_t[4]());
    FUSION_PASS_CHECK(input_assist_3.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_assist is NULL"),
                      return PARAM_INVALID);
    // add input and output desc
    reshape_1_desc->AddInputDesc("x", x0_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0] * x0_dims[1]);
    tmp_dims.push_back(x0_dims[2]);
    AssistIntHelp(tmp_dims, input_assist_1.get());
    x0_desc.SetShape(GeShape(tmp_dims));
    x0_desc.SetOriginShape(GeShape(tmp_dims));
    reshape_1_desc->AddOutputDesc("y", x0_desc);
    matmul_desc->AddInputDesc("x1", x0_desc);
    reshape_2_desc->AddInputDesc("x", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x1_dims[0]);
    tmp_dims.push_back(x1_dims[1] * x1_dims[2]);
    AssistIntHelp(tmp_dims, input_assist_2.get());
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    reshape_2_desc->AddOutputDesc("y", x1_desc);
    matmul_desc->AddInputDesc("x2", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0] * x0_dims[1]);
    tmp_dims.push_back(x1_dims[1] * x1_dims[2]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    matmul_desc->AddOutputDesc("y", x1_desc);
    reshape_3_desc->AddInputDesc("x", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x1_dims[1]);
    tmp_dims.push_back(x1_dims[2]);
    AssistIntHelp(tmp_dims, input_assist_3.get());
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    reshape_3_desc->AddOutputDesc("y", x1_desc);
    // create matmul and reshape node
    NodePtr matmul_node = graph.AddNode(matmul_desc);
    NodePtr reshape_1_node = graph.AddNode(reshape_1_desc);
    NodePtr reshape_2_node = graph.AddNode(reshape_2_desc);
    NodePtr reshape_3_node = graph.AddNode(reshape_3_desc);
    // set matmul op attr
    Operator matmul_op = OpDescUtils::CreateOperatorFromNode(matmul_node);
    matmul_op.SetAttr("transpose_x1", false);
    matmul_op.SetAttr("transpose_x2", false);
    // add const
    assist_desc.SetShape(GeShape({2}));
    FUSION_PASS_MAKE_SHARED((assist_ptr_1 = make_shared<GeTensor>(
                                 assist_desc, reinterpret_cast<uint8_t*>(input_assist_1.get()), 2 * sizeof(int32_t))),
                            assist_ptr_1 = nullptr;
                            return PARAM_INVALID);
    vector<GeTensorPtr> weights_1 = {assist_ptr_1};
    OpDescUtils::SetWeights(reshape_1_node, weights_1);
    auto const_nodes_1 = OpDescUtils::GetConstInputs(reshape_1_node);
    NodePtr const_node_1 = const_nodes_1[0];
    const_node_1->GetOpDesc()->SetType("Constant");
    reshape_1_desc->SetIsInputConst(is_input_const);
    assist_desc.SetShape(GeShape({2}));
    FUSION_PASS_MAKE_SHARED((assist_ptr_2 = make_shared<GeTensor>(
                                 assist_desc, reinterpret_cast<uint8_t*>(input_assist_2.get()), 2 * sizeof(int32_t))),
                            assist_ptr_2 = nullptr;
                            return PARAM_INVALID);
    vector<GeTensorPtr> weights_2 = {assist_ptr_2};
    OpDescUtils::SetWeights(reshape_2_node, weights_2);
    auto const_nodes_2 = OpDescUtils::GetConstInputs(reshape_2_node);
    NodePtr const_node_2 = const_nodes_2[0];
    const_node_2->GetOpDesc()->SetType("Constant");
    reshape_2_desc->SetIsInputConst(is_input_const);
    assist_desc.SetShape(GeShape({4}));
    FUSION_PASS_MAKE_SHARED((assist_ptr_3 = make_shared<GeTensor>(
                                 assist_desc, reinterpret_cast<uint8_t*>(input_assist_3.get()), 4 * sizeof(int32_t))),
                            assist_ptr_3 = nullptr;
                            return PARAM_INVALID);
    vector<GeTensorPtr> weights_3 = {assist_ptr_3};
    OpDescUtils::SetWeights(reshape_3_node, weights_3);
    auto const_nodes_3 = OpDescUtils::GetConstInputs(reshape_3_node);
    NodePtr const_node_3 = const_nodes_3[0];
    const_node_3->GetOpDesc()->SetType("Constant");
    reshape_3_desc->SetIsInputConst(is_input_const);
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
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, reshape_1_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x0_anchor_peer_node->GetName().c_str(), reshape_1_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, reshape_2_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x1_anchor_peer_node->GetName().c_str(), reshape_2_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(reshape_1_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                reshape_1_node->GetName().c_str(), matmul_node->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(reshape_2_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                reshape_2_node->GetName().c_str(), matmul_node->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), reshape_3_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                matmul_node->GetName().c_str(), reshape_3_node->GetName().c_str()),
        return FAILED);
    for (uint64_t i = 0; i < out_anchor_peer_anchors.size(); ++i) {
      auto out_anchor_peer_anchor = out_anchor_peer_anchors.at(i);
      auto out_anchor_peer_node = out_anchor_peer_anchor->GetOwnerNode();
      FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(reshape_3_node->GetOutDataAnchor(0), out_anchor_peer_anchor),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                reshape_3_node->GetName().c_str(), out_anchor_peer_node->GetName().c_str()),
                        return FAILED);
    }
    // remove node
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove einsum node failed"), return FAILED);
  } else if (equation == "BTNH,BFNH->BNFT") {  // 002 transpose+transpose+batchmatmul(swap input)
    FUSION_PASS_CHECK((x0_dims.size() != 4) && (x1_dims.size() != 4),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "input dims size must be four and four."), return NOT_CHANGED);
    // add input and output desc
    transpose_2_desc->AddInputDesc("x", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x1_dims[0]);
    tmp_dims.push_back(x1_dims[2]);
    tmp_dims.push_back(x1_dims[1]);
    tmp_dims.push_back(x1_dims[3]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    transpose_2_desc->AddOutputDesc("y", x1_desc);
    batchmatmul_desc->AddInputDesc("x1", x1_desc);
    transpose_1_desc->AddInputDesc("x", x0_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[2]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x0_dims[3]);
    x0_desc.SetShape(GeShape(tmp_dims));
    x0_desc.SetOriginShape(GeShape(tmp_dims));
    transpose_1_desc->AddOutputDesc("y", x0_desc);
    batchmatmul_desc->AddInputDesc("x2", x0_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[2]);
    tmp_dims.push_back(x1_dims[1]);
    tmp_dims.push_back(x0_dims[1]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    batchmatmul_desc->AddOutputDesc("y", x1_desc);
    // create transpose and batchmatmul node
    NodePtr transpose_1_node = graph.AddNode(transpose_1_desc);
    NodePtr transpose_2_node = graph.AddNode(transpose_2_desc);
    NodePtr batchmatmul_node = graph.AddNode(batchmatmul_desc);
    // set op attr
    Operator batchmatmul_op = OpDescUtils::CreateOperatorFromNode(batchmatmul_node);
    batchmatmul_op.SetAttr("adj_x1", false);
    batchmatmul_op.SetAttr("adj_x2", true);
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
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x0_anchor_peer_node->GetName().c_str(), transpose_1_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, transpose_2_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x1_anchor_peer_node->GetName().c_str(), transpose_2_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(transpose_1_node->GetOutDataAnchor(0), batchmatmul_node->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                transpose_1_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(transpose_2_node->GetOutDataAnchor(0), batchmatmul_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                transpose_2_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
        return FAILED);
    for (uint64_t i = 0; i < out_anchor_peer_anchors.size(); ++i) {
      auto out_anchor_peer_anchor = out_anchor_peer_anchors.at(i);
      auto out_anchor_peer_node = out_anchor_peer_anchor->GetOwnerNode();
      FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(batchmatmul_node->GetOutDataAnchor(0), out_anchor_peer_anchor),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                batchmatmul_node->GetName().c_str(), out_anchor_peer_node->GetName().c_str()),
                        return FAILED);
    }
    // remove node
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove einsum node failed"), return FAILED);
  } else if (equation == "BNFT,BTNH->BFNH") {  // 003:transpose+batchmatmul+transpose
    FUSION_PASS_CHECK((x0_dims.size() != 4) && (x1_dims.size() != 4),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "input dims size must be four and four."), return NOT_CHANGED);
    // add input and output desc
    batchmatmul_desc->AddInputDesc("x1", x0_desc);
    transpose_1_desc->AddInputDesc("x", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x1_dims[0]);
    tmp_dims.push_back(x1_dims[2]);
    tmp_dims.push_back(x1_dims[1]);
    tmp_dims.push_back(x1_dims[3]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    transpose_1_desc->AddOutputDesc("y", x1_desc);
    batchmatmul_desc->AddInputDesc("x2", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x0_dims[2]);
    tmp_dims.push_back(x1_dims[3]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    batchmatmul_desc->AddOutputDesc("y", x1_desc);
    transpose_2_desc->AddInputDesc("x", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[2]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x1_dims[3]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    transpose_2_desc->AddOutputDesc("y", x1_desc);
    // create batchmatmul and transpose node
    NodePtr transpose_1_node = graph.AddNode(transpose_1_desc);
    NodePtr transpose_2_node = graph.AddNode(transpose_2_desc);
    NodePtr batchmatmul_node = graph.AddNode(batchmatmul_desc);
    // set op attr
    Operator batchmatmul_op = OpDescUtils::CreateOperatorFromNode(batchmatmul_node);
    batchmatmul_op.SetAttr("adj_x1", false);
    batchmatmul_op.SetAttr("adj_x2", false);
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
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, batchmatmul_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x0_anchor_peer_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, transpose_1_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x1_anchor_peer_node->GetName().c_str(), transpose_1_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(transpose_1_node->GetOutDataAnchor(0), batchmatmul_node->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                transpose_1_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(batchmatmul_node->GetOutDataAnchor(0), transpose_2_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                batchmatmul_node->GetName().c_str(), transpose_2_node->GetName().c_str()),
        return FAILED);
    for (uint64_t i = 0; i < out_anchor_peer_anchors.size(); ++i) {
      auto out_anchor_peer_anchor = out_anchor_peer_anchors.at(i);
      auto out_anchor_peer_node = out_anchor_peer_anchor->GetOwnerNode();
      FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(transpose_2_node->GetOutDataAnchor(0), out_anchor_peer_anchor),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                transpose_2_node->GetName().c_str(), out_anchor_peer_node->GetName().c_str()),
                        return FAILED);
    }
    // remove node
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove einsum node failed"), return FAILED);
  } else if (equation == "abcd,cde->abe") {  // 004:reshape+reshape+matmul+reshape-->reshape+batchmatmul
    FUSION_PASS_CHECK((x0_dims.size() != 4) && (x1_dims.size() != 3),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "input dims size must be four and three."), return NOT_CHANGED);
    // init const
    unique_ptr<int32_t[]> input_assist_1(new (nothrow) int32_t[3]());
    FUSION_PASS_CHECK(input_assist_1.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_assist is NULL"),
                      return PARAM_INVALID);
    unique_ptr<int32_t[]> input_assist_2(new (nothrow) int32_t[2]());
    FUSION_PASS_CHECK(input_assist_2.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_assist is NULL"),
                      return PARAM_INVALID);
    // add input and output desc
    reshape_1_desc->AddInputDesc("x", x0_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x0_dims[2] * x0_dims[3]);
    AssistIntHelp(tmp_dims, input_assist_1.get());
    x0_desc.SetShape(GeShape(tmp_dims));
    x0_desc.SetOriginShape(GeShape(tmp_dims));
    reshape_1_desc->AddOutputDesc("y", x0_desc);
    batchmatmul_desc->AddInputDesc("x1", x0_desc);
    reshape_2_desc->AddInputDesc("x", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x1_dims[0] * x1_dims[1]);
    tmp_dims.push_back(x1_dims[2]);
    AssistIntHelp(tmp_dims, input_assist_2.get());
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    reshape_2_desc->AddOutputDesc("y", x1_desc);
    batchmatmul_desc->AddInputDesc("x2", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x1_dims[2]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    batchmatmul_desc->AddOutputDesc("y", x1_desc);
    // create batchmatmul and reshape node
    NodePtr batchmatmul_node = graph.AddNode(batchmatmul_desc);
    NodePtr reshape_1_node = graph.AddNode(reshape_1_desc);
    NodePtr reshape_2_node = graph.AddNode(reshape_2_desc);
    // set batchmatmul op attr
    Operator batchmatmul_op = OpDescUtils::CreateOperatorFromNode(batchmatmul_node);
    batchmatmul_op.SetAttr("adj_x1", false);
    batchmatmul_op.SetAttr("adj_x2", false);
    // add const
    assist_desc.SetShape(GeShape({3}));
    FUSION_PASS_MAKE_SHARED((assist_ptr_1 = make_shared<GeTensor>(
                                 assist_desc, reinterpret_cast<uint8_t*>(input_assist_1.get()), 3 * sizeof(int32_t))),
                            assist_ptr_1 = nullptr;
                            return PARAM_INVALID);
    vector<GeTensorPtr> weights_1 = {assist_ptr_1};
    OpDescUtils::SetWeights(reshape_1_node, weights_1);
    auto const_nodes_1 = OpDescUtils::GetConstInputs(reshape_1_node);
    NodePtr const_node_1 = const_nodes_1[0];
    const_node_1->GetOpDesc()->SetType("Constant");
    reshape_1_desc->SetIsInputConst(is_input_const);
    assist_desc.SetShape(GeShape({2}));
    FUSION_PASS_MAKE_SHARED((assist_ptr_2 = make_shared<GeTensor>(
                                 assist_desc, reinterpret_cast<uint8_t*>(input_assist_2.get()), 2 * sizeof(int32_t))),
                            assist_ptr_2 = nullptr;
                            return PARAM_INVALID);
    vector<GeTensorPtr> weights_2 = {assist_ptr_2};
    OpDescUtils::SetWeights(reshape_2_node, weights_2);
    auto const_nodes_2 = OpDescUtils::GetConstInputs(reshape_2_node);
    NodePtr const_node_2 = const_nodes_2[0];
    const_node_2->GetOpDesc()->SetType("Constant");
    reshape_2_desc->SetIsInputConst(is_input_const);
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
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, reshape_1_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x0_anchor_peer_node->GetName().c_str(), reshape_1_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, reshape_2_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x1_anchor_peer_node->GetName().c_str(), reshape_2_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(reshape_1_node->GetOutDataAnchor(0), batchmatmul_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                reshape_1_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(reshape_2_node->GetOutDataAnchor(0), batchmatmul_node->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                reshape_2_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
        return FAILED);
    for (uint64_t i = 0; i < out_anchor_peer_anchors.size(); ++i) {
      auto out_anchor_peer_anchor = out_anchor_peer_anchors.at(i);
      auto out_anchor_peer_node = out_anchor_peer_anchor->GetOwnerNode();
      FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(batchmatmul_node->GetOutDataAnchor(0), out_anchor_peer_anchor),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                batchmatmul_node->GetName().c_str(), out_anchor_peer_node->GetName().c_str()),
                        return FAILED);
    }
    // remove node
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove einsum node failed"), return FAILED);
  } else if (equation == "abc,cd->abd") {  // 005:reshape+matmul+reshape-->batchmatmul
    FUSION_PASS_CHECK((x0_dims.size() != 3) && (x1_dims.size() != 2),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "input dims size must be three and two."), return NOT_CHANGED);

    // add input and output desc
    batchmatmul_desc->AddInputDesc("x1", x0_desc);
    batchmatmul_desc->AddInputDesc("x2", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x1_dims[1]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    batchmatmul_desc->AddOutputDesc("y", x1_desc);
    // create batchmatmul node
    NodePtr batchmatmul_node = graph.AddNode(batchmatmul_desc);
    // set batchmatmul op attr
    Operator batchmatmul_op = OpDescUtils::CreateOperatorFromNode(batchmatmul_node);
    batchmatmul_op.SetAttr("adj_x1", false);
    batchmatmul_op.SetAttr("adj_x2", false);
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
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, batchmatmul_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x0_anchor_peer_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, batchmatmul_node->GetInDataAnchor(1)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x1_anchor_peer_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
                      return FAILED);
    for (uint64_t i = 0; i < out_anchor_peer_anchors.size(); ++i) {
      auto out_anchor_peer_anchor = out_anchor_peer_anchors.at(i);
      auto out_anchor_peer_node = out_anchor_peer_anchor->GetOwnerNode();
      FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(batchmatmul_node->GetOutDataAnchor(0), out_anchor_peer_anchor),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                batchmatmul_node->GetName().c_str(), out_anchor_peer_node->GetName().c_str()),
                        return FAILED);
    }
    // remove node
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove einsum node failed"), return FAILED);
  } else if (equation == "abd,cd->abc") {  // 006:reshape+matmul+reshape-->batchmatmul
    FUSION_PASS_CHECK((x0_dims.size() != 3) && (x1_dims.size() != 2),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "input dims size must be three and two."), return NOT_CHANGED);
    // add input and output desc
    batchmatmul_desc->AddInputDesc("x1", x0_desc);
    batchmatmul_desc->AddInputDesc("x2", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x1_dims[0]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    batchmatmul_desc->AddOutputDesc("y", x1_desc);
    // create batchmatmul and reshape node
    NodePtr batchmatmul_node = graph.AddNode(batchmatmul_desc);
    // set matmul op attr
    Operator batchmatmul_op = OpDescUtils::CreateOperatorFromNode(batchmatmul_node);
    batchmatmul_op.SetAttr("adj_x1", false);
    batchmatmul_op.SetAttr("adj_x2", true);
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
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, batchmatmul_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x0_anchor_peer_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, batchmatmul_node->GetInDataAnchor(1)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x1_anchor_peer_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
                      return FAILED);
    for (uint64_t i = 0; i < out_anchor_peer_anchors.size(); ++i) {
      auto out_anchor_peer_anchor = out_anchor_peer_anchors.at(i);
      auto out_anchor_peer_node = out_anchor_peer_anchor->GetOwnerNode();
      FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(batchmatmul_node->GetOutDataAnchor(0), out_anchor_peer_anchor),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                batchmatmul_node->GetName().c_str(), out_anchor_peer_node->GetName().c_str()),
                        return FAILED);
    }
    // remove node
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove einsum node failed"), return FAILED);
  } else if (equation == "abd,abc->cd") {  // 007:reshape+reshape+matmul(swap input)
    FUSION_PASS_CHECK((x0_dims.size() != 3) && (x1_dims.size() != 3),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "input dims size must be three and three."), return NOT_CHANGED);
    // init const
    unique_ptr<int32_t[]> input_assist_1(new (nothrow) int32_t[2]());
    FUSION_PASS_CHECK(input_assist_1.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_assist is NULL"),
                      return PARAM_INVALID);
    unique_ptr<int32_t[]> input_assist_2(new (nothrow) int32_t[2]());
    FUSION_PASS_CHECK(input_assist_2.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_assist is NULL"),
                      return PARAM_INVALID);
    // add input and output desc
    reshape_2_desc->AddInputDesc("x", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x1_dims[0] * x1_dims[1]);
    tmp_dims.push_back(x1_dims[2]);
    AssistIntHelp(tmp_dims, input_assist_2.get());
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    reshape_2_desc->AddOutputDesc("y", x1_desc);
    matmul_desc->AddInputDesc("x1", x1_desc);
    reshape_1_desc->AddInputDesc("x", x0_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0] * x0_dims[1]);
    tmp_dims.push_back(x0_dims[2]);
    AssistIntHelp(tmp_dims, input_assist_1.get());
    x0_desc.SetShape(GeShape(tmp_dims));
    x0_desc.SetOriginShape(GeShape(tmp_dims));
    reshape_1_desc->AddOutputDesc("y", x0_desc);
    matmul_desc->AddInputDesc("x2", x0_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x1_dims[2]);
    tmp_dims.push_back(x0_dims[2]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    matmul_desc->AddOutputDesc("y", x1_desc);
    // create matmul and reshape node
    NodePtr matmul_node = graph.AddNode(matmul_desc);
    NodePtr reshape_1_node = graph.AddNode(reshape_1_desc);
    NodePtr reshape_2_node = graph.AddNode(reshape_2_desc);
    // set op attr
    Operator matmul_op = OpDescUtils::CreateOperatorFromNode(matmul_node);
    matmul_op.SetAttr("transpose_x1", true);
    matmul_op.SetAttr("transpose_x2", false);
    // add const
    assist_desc.SetShape(GeShape({2}));
    FUSION_PASS_MAKE_SHARED((assist_ptr_1 = make_shared<GeTensor>(
                                 assist_desc, reinterpret_cast<uint8_t*>(input_assist_1.get()), 2 * sizeof(int32_t))),
                            assist_ptr_1 = nullptr;
                            return PARAM_INVALID);
    vector<GeTensorPtr> weights_1 = {assist_ptr_1};
    OpDescUtils::SetWeights(reshape_1_node, weights_1);
    auto const_nodes_1 = OpDescUtils::GetConstInputs(reshape_1_node);
    NodePtr const_node_1 = const_nodes_1[0];
    const_node_1->GetOpDesc()->SetType("Constant");
    reshape_1_desc->SetIsInputConst(is_input_const);
    assist_desc.SetShape(GeShape({2}));
    FUSION_PASS_MAKE_SHARED((assist_ptr_2 = make_shared<GeTensor>(
                                 assist_desc, reinterpret_cast<uint8_t*>(input_assist_2.get()), 2 * sizeof(int32_t))),
                            assist_ptr_2 = nullptr;
                            return PARAM_INVALID);
    vector<GeTensorPtr> weights_2 = {assist_ptr_2};
    OpDescUtils::SetWeights(reshape_2_node, weights_2);
    auto const_nodes_2 = OpDescUtils::GetConstInputs(reshape_2_node);
    NodePtr const_node_2 = const_nodes_2[0];
    const_node_2->GetOpDesc()->SetType("Constant");
    reshape_2_desc->SetIsInputConst(is_input_const);
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
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, reshape_1_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x0_anchor_peer_node->GetName().c_str(), reshape_1_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(reshape_1_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                reshape_1_node->GetName().c_str(), matmul_node->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, reshape_2_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x1_anchor_peer_node->GetName().c_str(), reshape_2_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(reshape_2_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                reshape_2_node->GetName().c_str(), matmul_node->GetName().c_str()),
        return FAILED);
    for (uint64_t i = 0; i < out_anchor_peer_anchors.size(); ++i) {
      auto out_anchor_peer_anchor = out_anchor_peer_anchors.at(i);
      auto out_anchor_peer_node = out_anchor_peer_anchor->GetOwnerNode();
      FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), out_anchor_peer_anchor),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                matmul_node->GetName().c_str(), out_anchor_peer_node->GetName().c_str()),
                        return FAILED);
    }
    // remove node
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove einsum node failed"), return FAILED);
  } else if (equation == "abe,cde->abcd") {  // 008:reshape+reshape+matmul+reshape
    FUSION_PASS_CHECK((x0_dims.size() != 3) && (x1_dims.size() != 3),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "input dims size must be three and three."), return NOT_CHANGED);
    // init const
    unique_ptr<int32_t[]> input_assist_1(new (nothrow) int32_t[2]());
    FUSION_PASS_CHECK(input_assist_1.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_assist is NULL"),
                      return PARAM_INVALID);
    unique_ptr<int32_t[]> input_assist_2(new (nothrow) int32_t[2]());
    FUSION_PASS_CHECK(input_assist_2.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_assist is NULL"),
                      return PARAM_INVALID);
    unique_ptr<int32_t[]> input_assist_3(new (nothrow) int32_t[4]());
    FUSION_PASS_CHECK(input_assist_3.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_assist is NULL"),
                      return PARAM_INVALID);
    // add input and output desc
    reshape_1_desc->AddInputDesc("x", x0_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0] * x0_dims[1]);
    tmp_dims.push_back(x0_dims[2]);
    AssistIntHelp(tmp_dims, input_assist_1.get());
    x0_desc.SetShape(GeShape(tmp_dims));
    x0_desc.SetOriginShape(GeShape(tmp_dims));
    reshape_1_desc->AddOutputDesc("y", x0_desc);
    matmul_desc->AddInputDesc("x1", x0_desc);
    reshape_2_desc->AddInputDesc("x", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x1_dims[0] * x1_dims[1]);
    tmp_dims.push_back(x1_dims[2]);
    AssistIntHelp(tmp_dims, input_assist_2.get());
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    reshape_2_desc->AddOutputDesc("y", x1_desc);
    matmul_desc->AddInputDesc("x2", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0] * x0_dims[1]);
    tmp_dims.push_back(x1_dims[0] * x1_dims[1]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    matmul_desc->AddOutputDesc("y", x1_desc);
    reshape_3_desc->AddInputDesc("x", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x1_dims[0]);
    tmp_dims.push_back(x1_dims[1]);
    AssistIntHelp(tmp_dims, input_assist_3.get());
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    reshape_3_desc->AddOutputDesc("y", x1_desc);
    // create matmul and reshape node
    NodePtr matmul_node = graph.AddNode(matmul_desc);
    NodePtr reshape_1_node = graph.AddNode(reshape_1_desc);
    NodePtr reshape_2_node = graph.AddNode(reshape_2_desc);
    NodePtr reshape_3_node = graph.AddNode(reshape_3_desc);
    // set op attr
    Operator matmul_op = OpDescUtils::CreateOperatorFromNode(matmul_node);
    matmul_op.SetAttr("transpose_x1", false);
    matmul_op.SetAttr("transpose_x2", true);
    // add const
    assist_desc.SetShape(GeShape({2}));
    FUSION_PASS_MAKE_SHARED((assist_ptr_1 = make_shared<GeTensor>(
                                 assist_desc, reinterpret_cast<uint8_t*>(input_assist_1.get()), 2 * sizeof(int32_t))),
                            assist_ptr_1 = nullptr;
                            return PARAM_INVALID);
    vector<GeTensorPtr> weights_1 = {assist_ptr_1};
    OpDescUtils::SetWeights(reshape_1_node, weights_1);
    auto const_nodes_1 = OpDescUtils::GetConstInputs(reshape_1_node);
    NodePtr const_node_1 = const_nodes_1[0];
    const_node_1->GetOpDesc()->SetType("Constant");
    reshape_1_desc->SetIsInputConst(is_input_const);
    assist_desc.SetShape(GeShape({2}));
    FUSION_PASS_MAKE_SHARED((assist_ptr_2 = make_shared<GeTensor>(
                                 assist_desc, reinterpret_cast<uint8_t*>(input_assist_2.get()), 2 * sizeof(int32_t))),
                            assist_ptr_2 = nullptr;
                            return PARAM_INVALID);
    vector<GeTensorPtr> weights_2 = {assist_ptr_2};
    OpDescUtils::SetWeights(reshape_2_node, weights_2);
    auto const_nodes_2 = OpDescUtils::GetConstInputs(reshape_2_node);
    NodePtr const_node_2 = const_nodes_2[0];
    const_node_2->GetOpDesc()->SetType("Constant");
    reshape_2_desc->SetIsInputConst(is_input_const);
    assist_desc.SetShape(GeShape({4}));
    FUSION_PASS_MAKE_SHARED((assist_ptr_3 = make_shared<GeTensor>(
                                 assist_desc, reinterpret_cast<uint8_t*>(input_assist_3.get()), 4 * sizeof(int32_t))),
                            assist_ptr_3 = nullptr;
                            return PARAM_INVALID);
    vector<GeTensorPtr> weights_3 = {assist_ptr_3};
    OpDescUtils::SetWeights(reshape_3_node, weights_3);
    auto const_nodes_3 = OpDescUtils::GetConstInputs(reshape_3_node);
    NodePtr const_node_3 = const_nodes_3[0];
    const_node_3->GetOpDesc()->SetType("Constant");
    reshape_3_desc->SetIsInputConst(is_input_const);
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
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, reshape_1_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x0_anchor_peer_node->GetName().c_str(), reshape_1_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(reshape_1_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                reshape_1_node->GetName().c_str(), matmul_node->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, reshape_2_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x1_anchor_peer_node->GetName().c_str(), reshape_2_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(reshape_2_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                reshape_2_node->GetName().c_str(), matmul_node->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), reshape_3_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                matmul_node->GetName().c_str(), reshape_3_node->GetName().c_str()),
        return FAILED);
    for (uint64_t i = 0; i < out_anchor_peer_anchors.size(); ++i) {
      auto out_anchor_peer_anchor = out_anchor_peer_anchors.at(i);
      auto out_anchor_peer_node = out_anchor_peer_anchor->GetOwnerNode();
      FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(reshape_3_node->GetOutDataAnchor(0), out_anchor_peer_anchor),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                reshape_3_node->GetName().c_str(), out_anchor_peer_node->GetName().c_str()),
                        return FAILED);
    }
    // remove node
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove einsum node failed"), return FAILED);
  } else if (equation == "abe,abcd->cde") {  // 009:reshape+reshape+matmul+reshape(swap input)
    FUSION_PASS_CHECK((x0_dims.size() != 3) && (x1_dims.size() != 4),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "input dims size must be three and four."), return NOT_CHANGED);
    // init const
    unique_ptr<int32_t[]> input_assist_1(new (nothrow) int32_t[2]());
    FUSION_PASS_CHECK(input_assist_1.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_assist is NULL"),
                      return PARAM_INVALID);
    unique_ptr<int32_t[]> input_assist_2(new (nothrow) int32_t[2]());
    FUSION_PASS_CHECK(input_assist_2.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_assist is NULL"),
                      return PARAM_INVALID);
    unique_ptr<int32_t[]> input_assist_3(new (nothrow) int32_t[3]());
    FUSION_PASS_CHECK(input_assist_3.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_assist is NULL"),
                      return PARAM_INVALID);
    // add input and output desc
    reshape_2_desc->AddInputDesc("x", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x1_dims[0] * x1_dims[1]);
    tmp_dims.push_back(x1_dims[2] * x1_dims[3]);
    AssistIntHelp(tmp_dims, input_assist_2.get());
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    reshape_2_desc->AddOutputDesc("y", x1_desc);
    matmul_desc->AddInputDesc("x1", x1_desc);
    reshape_1_desc->AddInputDesc("x", x0_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0] * x0_dims[1]);
    tmp_dims.push_back(x0_dims[2]);
    AssistIntHelp(tmp_dims, input_assist_1.get());
    x0_desc.SetShape(GeShape(tmp_dims));
    x0_desc.SetOriginShape(GeShape(tmp_dims));
    reshape_1_desc->AddOutputDesc("y", x0_desc);
    matmul_desc->AddInputDesc("x2", x0_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x1_dims[2] * x1_dims[3]);
    tmp_dims.push_back(x0_dims[2]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    matmul_desc->AddOutputDesc("y", x1_desc);
    reshape_3_desc->AddInputDesc("x", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x1_dims[2]);
    tmp_dims.push_back(x1_dims[3]);
    tmp_dims.push_back(x0_dims[2]);
    AssistIntHelp(tmp_dims, input_assist_3.get());
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    reshape_3_desc->AddOutputDesc("y", x1_desc);
    // create matmul and reshape node
    NodePtr matmul_node = graph.AddNode(matmul_desc);
    NodePtr reshape_1_node = graph.AddNode(reshape_1_desc);
    NodePtr reshape_2_node = graph.AddNode(reshape_2_desc);
    NodePtr reshape_3_node = graph.AddNode(reshape_3_desc);
    // set op attr
    Operator matmul_op = OpDescUtils::CreateOperatorFromNode(matmul_node);
    matmul_op.SetAttr("transpose_x1", true);
    matmul_op.SetAttr("transpose_x2", false);
    // add const
    assist_desc.SetShape(GeShape({2}));
    FUSION_PASS_MAKE_SHARED((assist_ptr_1 = make_shared<GeTensor>(
                                 assist_desc, reinterpret_cast<uint8_t*>(input_assist_1.get()), 2 * sizeof(int32_t))),
                            assist_ptr_1 = nullptr;
                            return PARAM_INVALID);
    vector<GeTensorPtr> weights_1 = {assist_ptr_1};
    OpDescUtils::SetWeights(reshape_1_node, weights_1);
    auto const_nodes_1 = OpDescUtils::GetConstInputs(reshape_1_node);
    NodePtr const_node_1 = const_nodes_1[0];
    const_node_1->GetOpDesc()->SetType("Constant");
    reshape_1_desc->SetIsInputConst(is_input_const);
    assist_desc.SetShape(GeShape({2}));
    FUSION_PASS_MAKE_SHARED((assist_ptr_2 = make_shared<GeTensor>(
                                 assist_desc, reinterpret_cast<uint8_t*>(input_assist_2.get()), 2 * sizeof(int32_t))),
                            assist_ptr_2 = nullptr;
                            return PARAM_INVALID);
    vector<GeTensorPtr> weights_2 = {assist_ptr_2};
    OpDescUtils::SetWeights(reshape_2_node, weights_2);
    auto const_nodes_2 = OpDescUtils::GetConstInputs(reshape_2_node);
    NodePtr const_node_2 = const_nodes_2[0];
    const_node_2->GetOpDesc()->SetType("Constant");
    reshape_2_desc->SetIsInputConst(is_input_const);
    assist_desc.SetShape(GeShape({3}));
    FUSION_PASS_MAKE_SHARED((assist_ptr_3 = make_shared<GeTensor>(
                                 assist_desc, reinterpret_cast<uint8_t*>(input_assist_3.get()), 3 * sizeof(int32_t))),
                            assist_ptr_3 = nullptr;
                            return PARAM_INVALID);
    vector<GeTensorPtr> weights_3 = {assist_ptr_3};
    OpDescUtils::SetWeights(reshape_3_node, weights_3);
    auto const_nodes_3 = OpDescUtils::GetConstInputs(reshape_3_node);
    NodePtr const_node_3 = const_nodes_3[0];
    const_node_3->GetOpDesc()->SetType("Constant");
    reshape_3_desc->SetIsInputConst(is_input_const);
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
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, reshape_1_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x0_anchor_peer_node->GetName().c_str(), reshape_1_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(reshape_1_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                reshape_1_node->GetName().c_str(), matmul_node->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, reshape_2_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x1_anchor_peer_node->GetName().c_str(), reshape_2_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(reshape_2_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                reshape_2_node->GetName().c_str(), matmul_node->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), reshape_3_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                matmul_node->GetName().c_str(), reshape_3_node->GetName().c_str()),
        return FAILED);
    for (uint64_t i = 0; i < out_anchor_peer_anchors.size(); ++i) {
      auto out_anchor_peer_anchor = out_anchor_peer_anchors.at(i);
      auto out_anchor_peer_node = out_anchor_peer_anchor->GetOwnerNode();
      FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(reshape_3_node->GetOutDataAnchor(0), out_anchor_peer_anchor),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                reshape_3_node->GetName().c_str(), out_anchor_peer_node->GetName().c_str()),
                        return FAILED);
    }
    // remove node
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove einsum node failed"), return FAILED);
  } else if (equation == "BFNH,BTNH->BNFT") {  // 010:transpose+batchmatmul+transpose
    FUSION_PASS_CHECK((x0_dims.size() != 4) && (x1_dims.size() != 4),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "input dims size must be four and four."),
                      return NOT_CHANGED);  // add input and output desc
    transpose_1_desc->AddInputDesc("x", x0_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[2]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x0_dims[3]);
    x0_desc.SetShape(GeShape(tmp_dims));
    x0_desc.SetOriginShape(GeShape(tmp_dims));
    transpose_1_desc->AddOutputDesc("y", x0_desc);
    batchmatmul_desc->AddInputDesc("x1", x0_desc);
    transpose_2_desc->AddInputDesc("x", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x1_dims[0]);
    tmp_dims.push_back(x1_dims[2]);
    tmp_dims.push_back(x1_dims[1]);
    tmp_dims.push_back(x1_dims[3]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    transpose_2_desc->AddOutputDesc("y", x1_desc);
    batchmatmul_desc->AddInputDesc("x2", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[2]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x1_dims[1]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    batchmatmul_desc->AddOutputDesc("y", x1_desc);
    // create transpose and batchmatmul node
    NodePtr transpose_1_node = graph.AddNode(transpose_1_desc);
    NodePtr transpose_2_node = graph.AddNode(transpose_2_desc);
    NodePtr batchmatmul_node = graph.AddNode(batchmatmul_desc);
    // set op attr
    Operator batchmatmul_op = OpDescUtils::CreateOperatorFromNode(batchmatmul_node);
    batchmatmul_op.SetAttr("adj_x1", false);
    batchmatmul_op.SetAttr("adj_x2", true);
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
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x0_anchor_peer_node->GetName().c_str(), transpose_1_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, transpose_2_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x1_anchor_peer_node->GetName().c_str(), transpose_2_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(transpose_1_node->GetOutDataAnchor(0), batchmatmul_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                transpose_1_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(transpose_2_node->GetOutDataAnchor(0), batchmatmul_node->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                transpose_2_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
        return FAILED);
    for (uint64_t i = 0; i < out_anchor_peer_anchors.size(); ++i) {
      auto out_anchor_peer_anchor = out_anchor_peer_anchors.at(i);
      auto out_anchor_peer_node = out_anchor_peer_anchor->GetOwnerNode();
      FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(batchmatmul_node->GetOutDataAnchor(0), out_anchor_peer_anchor),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                batchmatmul_node->GetName().c_str(), out_anchor_peer_node->GetName().c_str()),
                        return FAILED);
    }
    // remove node
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove einsum node failed"), return FAILED);
  } else if (equation == "BFNH,BNFT->BTNH") {  // 011:transpose+batchmatmul+transpose(swap input)
    FUSION_PASS_CHECK((x0_dims.size() != 4) && (x1_dims.size() != 4),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "input dims size must be four and four."), return NOT_CHANGED);
    // add input and output desc
    batchmatmul_desc->AddInputDesc("x1", x1_desc);
    transpose_1_desc->AddInputDesc("x", x0_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[2]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x0_dims[3]);
    x0_desc.SetShape(GeShape(tmp_dims));
    x0_desc.SetOriginShape(GeShape(tmp_dims));
    transpose_1_desc->AddOutputDesc("y", x0_desc);
    batchmatmul_desc->AddInputDesc("x2", x0_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x1_dims[0]);
    tmp_dims.push_back(x1_dims[1]);
    tmp_dims.push_back(x1_dims[3]);
    tmp_dims.push_back(x0_dims[3]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    batchmatmul_desc->AddOutputDesc("y", x1_desc);
    transpose_2_desc->AddInputDesc("x", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x1_dims[0]);
    tmp_dims.push_back(x1_dims[3]);
    tmp_dims.push_back(x1_dims[1]);
    tmp_dims.push_back(x0_dims[3]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    transpose_2_desc->AddOutputDesc("y", x1_desc);
    // create transpose and batchmatmul node
    NodePtr transpose_1_node = graph.AddNode(transpose_1_desc);
    NodePtr transpose_2_node = graph.AddNode(transpose_2_desc);
    NodePtr batchmatmul_node = graph.AddNode(batchmatmul_desc);
    // set op attr
    Operator batchmatmul_op = OpDescUtils::CreateOperatorFromNode(batchmatmul_node);
    batchmatmul_op.SetAttr("adj_x1", true);
    batchmatmul_op.SetAttr("adj_x2", false);
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
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x0_anchor_peer_node->GetName().c_str(), transpose_1_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(transpose_1_node->GetOutDataAnchor(0), batchmatmul_node->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                transpose_1_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, batchmatmul_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x1_anchor_peer_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(batchmatmul_node->GetOutDataAnchor(0), transpose_2_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                batchmatmul_node->GetName().c_str(), transpose_2_node->GetName().c_str()),
        return FAILED);
    for (uint64_t i = 0; i < out_anchor_peer_anchors.size(); ++i) {
      auto out_anchor_peer_anchor = out_anchor_peer_anchors.at(i);
      auto out_anchor_peer_node = out_anchor_peer_anchor->GetOwnerNode();
      FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(transpose_2_node->GetOutDataAnchor(0), out_anchor_peer_anchor),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                transpose_2_node->GetName().c_str(), out_anchor_peer_node->GetName().c_str()),
                        return FAILED);
    }
    // remove node
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove einsum node failed"), return FAILED);
  } else if (equation == "abde,cde->abc") {  // 012:reshape+reshape+matmul+reshape-->reshape+batchmatmul
    FUSION_PASS_CHECK((x0_dims.size() != 4) && (x1_dims.size() != 3),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "input dims size must be four and three."), return NOT_CHANGED);
    // init const
    unique_ptr<int32_t[]> input_assist_1(new (nothrow) int32_t[3]());
    FUSION_PASS_CHECK(input_assist_1.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_assist is NULL"),
                      return PARAM_INVALID);
    unique_ptr<int32_t[]> input_assist_2(new (nothrow) int32_t[2]());
    FUSION_PASS_CHECK(input_assist_2.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_assist is NULL"),
                      return PARAM_INVALID);
    // add input and output desc
    reshape_1_desc->AddInputDesc("x", x0_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x0_dims[2] * x0_dims[3]);
    AssistIntHelp(tmp_dims, input_assist_1.get());
    x0_desc.SetShape(GeShape(tmp_dims));
    x0_desc.SetOriginShape(GeShape(tmp_dims));
    reshape_1_desc->AddOutputDesc("y", x0_desc);
    batchmatmul_desc->AddInputDesc("x1", x0_desc);
    reshape_2_desc->AddInputDesc("x", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x1_dims[0]);
    tmp_dims.push_back(x1_dims[1] * x1_dims[2]);
    AssistIntHelp(tmp_dims, input_assist_2.get());
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    reshape_2_desc->AddOutputDesc("y", x1_desc);
    batchmatmul_desc->AddInputDesc("x2", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x1_dims[0]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    batchmatmul_desc->AddOutputDesc("y", x1_desc);
    // create batchmatmul and reshape node
    NodePtr batchmatmul_node = graph.AddNode(batchmatmul_desc);
    NodePtr reshape_1_node = graph.AddNode(reshape_1_desc);
    NodePtr reshape_2_node = graph.AddNode(reshape_2_desc);
    // set op attr
    Operator batchmatmul_op = OpDescUtils::CreateOperatorFromNode(batchmatmul_node);
    batchmatmul_op.SetAttr("adj_x1", false);
    batchmatmul_op.SetAttr("adj_x2", true);
    // add const
    assist_desc.SetShape(GeShape({3}));
    FUSION_PASS_MAKE_SHARED((assist_ptr_1 = make_shared<GeTensor>(
                                 assist_desc, reinterpret_cast<uint8_t*>(input_assist_1.get()), 3 * sizeof(int32_t))),
                            assist_ptr_1 = nullptr;
                            return PARAM_INVALID);
    vector<GeTensorPtr> weights_1 = {assist_ptr_1};
    OpDescUtils::SetWeights(reshape_1_node, weights_1);
    auto const_nodes_1 = OpDescUtils::GetConstInputs(reshape_1_node);
    NodePtr const_node_1 = const_nodes_1[0];
    const_node_1->GetOpDesc()->SetType("Constant");
    reshape_1_desc->SetIsInputConst(is_input_const);
    assist_desc.SetShape(GeShape({2}));
    FUSION_PASS_MAKE_SHARED((assist_ptr_2 = make_shared<GeTensor>(
                                 assist_desc, reinterpret_cast<uint8_t*>(input_assist_2.get()), 2 * sizeof(int32_t))),
                            assist_ptr_2 = nullptr;
                            return PARAM_INVALID);
    vector<GeTensorPtr> weights_2 = {assist_ptr_2};
    OpDescUtils::SetWeights(reshape_2_node, weights_2);
    auto const_nodes_2 = OpDescUtils::GetConstInputs(reshape_2_node);
    NodePtr const_node_2 = const_nodes_2[0];
    const_node_2->GetOpDesc()->SetType("Constant");
    reshape_2_desc->SetIsInputConst(is_input_const);
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
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, reshape_1_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x0_anchor_peer_node->GetName().c_str(), reshape_1_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(reshape_1_node->GetOutDataAnchor(0), batchmatmul_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                reshape_1_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, reshape_2_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x1_anchor_peer_node->GetName().c_str(), reshape_2_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(reshape_2_node->GetOutDataAnchor(0), batchmatmul_node->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                reshape_2_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
        return FAILED);
    for (uint64_t i = 0; i < out_anchor_peer_anchors.size(); ++i) {
      auto out_anchor_peer_anchor = out_anchor_peer_anchors.at(i);
      auto out_anchor_peer_node = out_anchor_peer_anchor->GetOwnerNode();
      FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(batchmatmul_node->GetOutDataAnchor(0), out_anchor_peer_anchor),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                batchmatmul_node->GetName().c_str(), out_anchor_peer_node->GetName().c_str()),
                        return FAILED);
    }
    // remove node
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove einsum node failed"), return FAILED);
  } else if (equation == "abde,abc->cde") {  // 013:reshape+reshape+matmul+reshape(swap input)
    FUSION_PASS_CHECK((x0_dims.size() != 4) && (x1_dims.size() != 3),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "input dims size must be four and three."), return NOT_CHANGED);
    // init const
    unique_ptr<int32_t[]> input_assist_1(new (nothrow) int32_t[2]());
    FUSION_PASS_CHECK(input_assist_1.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_assist is NULL"),
                      return PARAM_INVALID);
    unique_ptr<int32_t[]> input_assist_2(new (nothrow) int32_t[2]());
    FUSION_PASS_CHECK(input_assist_2.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_assist is NULL"),
                      return PARAM_INVALID);
    unique_ptr<int32_t[]> input_assist_3(new (nothrow) int32_t[3]());
    FUSION_PASS_CHECK(input_assist_3.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_assist is NULL"),
                      return PARAM_INVALID);
    // add input and output desc
    reshape_2_desc->AddInputDesc("x", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x1_dims[0] * x1_dims[1]);
    tmp_dims.push_back(x1_dims[2]);
    AssistIntHelp(tmp_dims, input_assist_2.get());
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    reshape_2_desc->AddOutputDesc("y", x1_desc);
    matmul_desc->AddInputDesc("x1", x1_desc);
    reshape_1_desc->AddInputDesc("x", x0_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0] * x0_dims[1]);
    tmp_dims.push_back(x0_dims[2] * x0_dims[3]);
    AssistIntHelp(tmp_dims, input_assist_1.get());
    x0_desc.SetShape(GeShape(tmp_dims));
    x0_desc.SetOriginShape(GeShape(tmp_dims));
    reshape_1_desc->AddOutputDesc("y", x0_desc);
    matmul_desc->AddInputDesc("x2", x0_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x1_dims[2]);
    tmp_dims.push_back(x0_dims[2] * x0_dims[3]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    matmul_desc->AddOutputDesc("y", x1_desc);
    reshape_3_desc->AddInputDesc("x", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x1_dims[2]);
    tmp_dims.push_back(x0_dims[2]);
    tmp_dims.push_back(x0_dims[3]);
    AssistIntHelp(tmp_dims, input_assist_3.get());
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    reshape_3_desc->AddOutputDesc("y", x1_desc);
    // create matmul reshape node
    NodePtr matmul_node = graph.AddNode(matmul_desc);
    NodePtr reshape_1_node = graph.AddNode(reshape_1_desc);
    NodePtr reshape_2_node = graph.AddNode(reshape_2_desc);
    NodePtr reshape_3_node = graph.AddNode(reshape_3_desc);
    // set op attr
    Operator matmul_op = OpDescUtils::CreateOperatorFromNode(matmul_node);
    matmul_op.SetAttr("transpose_x1", true);
    matmul_op.SetAttr("transpose_x2", false);
    // add const
    assist_desc.SetShape(GeShape({2}));
    FUSION_PASS_MAKE_SHARED((assist_ptr_1 = make_shared<GeTensor>(
                                 assist_desc, reinterpret_cast<uint8_t*>(input_assist_1.get()), 2 * sizeof(int32_t))),
                            assist_ptr_1 = nullptr;
                            return PARAM_INVALID);
    vector<GeTensorPtr> weights_1 = {assist_ptr_1};
    OpDescUtils::SetWeights(reshape_1_node, weights_1);
    auto const_nodes_1 = OpDescUtils::GetConstInputs(reshape_1_node);
    NodePtr const_node_1 = const_nodes_1[0];
    const_node_1->GetOpDesc()->SetType("Constant");
    reshape_1_desc->SetIsInputConst(is_input_const);
    assist_desc.SetShape(GeShape({2}));
    FUSION_PASS_MAKE_SHARED((assist_ptr_2 = make_shared<GeTensor>(
                                 assist_desc, reinterpret_cast<uint8_t*>(input_assist_2.get()), 2 * sizeof(int32_t))),
                            assist_ptr_2 = nullptr;
                            return PARAM_INVALID);
    vector<GeTensorPtr> weights_2 = {assist_ptr_2};
    OpDescUtils::SetWeights(reshape_2_node, weights_2);
    auto const_nodes_2 = OpDescUtils::GetConstInputs(reshape_2_node);
    NodePtr const_node_2 = const_nodes_2[0];
    const_node_2->GetOpDesc()->SetType("Constant");
    reshape_2_desc->SetIsInputConst(is_input_const);
    assist_desc.SetShape(GeShape({3}));
    FUSION_PASS_MAKE_SHARED((assist_ptr_3 = make_shared<GeTensor>(
                                 assist_desc, reinterpret_cast<uint8_t*>(input_assist_3.get()), 3 * sizeof(int32_t))),
                            assist_ptr_3 = nullptr;
                            return PARAM_INVALID);
    vector<GeTensorPtr> weights_3 = {assist_ptr_3};
    OpDescUtils::SetWeights(reshape_3_node, weights_3);
    auto const_nodes_3 = OpDescUtils::GetConstInputs(reshape_3_node);
    NodePtr const_node_3 = const_nodes_3[0];
    const_node_3->GetOpDesc()->SetType("Constant");
    reshape_3_desc->SetIsInputConst(is_input_const);
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
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, reshape_1_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x0_anchor_peer_node->GetName().c_str(), reshape_1_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(reshape_1_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                reshape_1_node->GetName().c_str(), matmul_node->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, reshape_2_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x1_anchor_peer_node->GetName().c_str(), reshape_2_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(reshape_2_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                reshape_2_node->GetName().c_str(), matmul_node->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), reshape_3_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                matmul_node->GetName().c_str(), reshape_3_node->GetName().c_str()),
        return FAILED);
    for (uint64_t i = 0; i < out_anchor_peer_anchors.size(); ++i) {
      auto out_anchor_peer_anchor = out_anchor_peer_anchors.at(i);
      auto out_anchor_peer_node = out_anchor_peer_anchor->GetOwnerNode();
      FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(reshape_3_node->GetOutDataAnchor(0), out_anchor_peer_anchor),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                reshape_3_node->GetName().c_str(), out_anchor_peer_node->GetName().c_str()),
                        return FAILED);
    }
    // remove node
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove einsum node failed"), return FAILED);
  } else if (equation == "BNFT,BFNH->BTNH") {  // 014: transpose+batchmatmul+transpose
    FUSION_PASS_CHECK((x0_dims.size() != 4) && (x1_dims.size() != 4),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "input dims size must be four and four."), return NOT_CHANGED);
    // add input and output desc
    batchmatmul_desc->AddInputDesc("x1", x0_desc);
    transpose_2_desc->AddInputDesc("x", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x1_dims[0]);
    tmp_dims.push_back(x1_dims[2]);
    tmp_dims.push_back(x1_dims[1]);
    tmp_dims.push_back(x1_dims[3]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    transpose_2_desc->AddOutputDesc("y", x1_desc);
    batchmatmul_desc->AddInputDesc("x2", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x0_dims[3]);
    tmp_dims.push_back(x1_dims[3]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    batchmatmul_desc->AddOutputDesc("y", x1_desc);
    transpose_3_desc->AddInputDesc("x", x1_desc);
    tmp_dims.clear();
    tmp_dims.push_back(x0_dims[0]);
    tmp_dims.push_back(x0_dims[3]);
    tmp_dims.push_back(x0_dims[1]);
    tmp_dims.push_back(x1_dims[3]);
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    transpose_3_desc->AddOutputDesc("y", x1_desc);
    // create transpose and batchmatmul node
    NodePtr transpose_2_node = graph.AddNode(transpose_2_desc);
    NodePtr transpose_3_node = graph.AddNode(transpose_3_desc);
    NodePtr batchmatmul_node = graph.AddNode(batchmatmul_desc);
    // set op attr
    Operator batchmatmul_op = OpDescUtils::CreateOperatorFromNode(batchmatmul_node);
    batchmatmul_op.SetAttr("adj_x1", true);
    batchmatmul_op.SetAttr("adj_x2", false);
    Operator transpose2_op = OpDescUtils::CreateOperatorFromNode(transpose_2_node);
    tmp_dims.clear();
    tmp_dims.push_back(0);
    tmp_dims.push_back(2);
    tmp_dims.push_back(1);
    tmp_dims.push_back(3);
    transpose2_op.SetAttr("perm", tmp_dims);
    Operator transpose3_op = OpDescUtils::CreateOperatorFromNode(transpose_3_node);
    tmp_dims.clear();
    tmp_dims.push_back(0);
    tmp_dims.push_back(2);
    tmp_dims.push_back(1);
    tmp_dims.push_back(3);
    transpose3_op.SetAttr("perm", tmp_dims);
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
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, batchmatmul_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x0_anchor_peer_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, transpose_2_node->GetInDataAnchor(0)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              x1_anchor_peer_node->GetName().c_str(), transpose_2_node->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(transpose_2_node->GetOutDataAnchor(0), batchmatmul_node->GetInDataAnchor(1)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                transpose_2_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(batchmatmul_node->GetOutDataAnchor(0), transpose_3_node->GetInDataAnchor(0)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                batchmatmul_node->GetName().c_str(), transpose_3_node->GetName().c_str()),
        return FAILED);
    for (uint64_t i = 0; i < out_anchor_peer_anchors.size(); ++i) {
      auto out_anchor_peer_anchor = out_anchor_peer_anchors.at(i);
      auto out_anchor_peer_node = out_anchor_peer_anchor->GetOwnerNode();
      FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(transpose_3_node->GetOutDataAnchor(0), out_anchor_peer_anchor),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                transpose_3_node->GetName().c_str(), out_anchor_peer_node->GetName().c_str()),
                        return FAILED);
    }
    // remove node
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove einsum node failed"), return FAILED);
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "equation relu is not match.");
    return NOT_CHANGED;
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "EinsumPass fusion success!");
  return SUCCESS;
}
REGISTER_PASS("EinsumPass", BUILT_IN_GRAPH_PASS, EinsumPass);
}  // namespace fe
