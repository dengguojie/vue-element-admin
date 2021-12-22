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
 * \file einsum_fusion_pass.cc
 * \brief
 */
#include "einsum_fusion_pass.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "error_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe {
static const string kPatternFusedNode = "Einsum";
static const string kEinsum = "Einsum";
static const string kReshape = "Reshape";
static const string kTransposeD = "TransposeD";
static const string kTranspose = "Transpose";
static const string kMatMul = "MatMulV2";
static const string kBatchMatMul = "BatchMatMul";
static const string kFlatten = "FlattenV2";
static const string kGatherShapes = "GatherShapes";

static void EquationNormalization(string &equation) {
  map<char, char> normalize_map;
  size_t indice = 0;
  for (auto &dim_shape : equation) {
    if (isalpha(dim_shape)) {
      if (normalize_map.find(dim_shape) == normalize_map.end()) {
        normalize_map[dim_shape] = 'a' + indice;
        indice++;
      }
      dim_shape = normalize_map[dim_shape];
    }
  }
}

static void AssistIntHelp(const vector<int64_t> &const_vec, int32_t *output) {
  for (size_t i = 0; i < const_vec.size(); ++i) {
    output[i] = const_vec[i];
  }
}

static int64_t GetDimMulValue(int64_t dim_value1, int64_t dim_value2) {
  if (dim_value1 == -1 || dim_value2 == -1) {
    return -1;
  }

  // overflow detected in CheckInputArgs function
  return dim_value1 * dim_value2;
}

std::unordered_map<std::string, EinsumPass::ProcFunc> EinsumPass::staticShapeProcs_ = {
  {"abc,cde->abde", &EinsumPass::HandleStaticABCxCDE2ABDE},
  {"abcd,aecd->aceb", &EinsumPass::HandleABCDxAECD2ACEB},
  {"abcd,adbe->acbe", &EinsumPass::HandleABCDxADBE2ACBE},
  {"abcd,cde->abe", &EinsumPass::HandleABCDxCDE2ABE},
  {"abc,cd->abd", &EinsumPass::HandleABCxCD2ABD},
  {"abc,dc->abd", &EinsumPass::HandleABCxDC2ABD},
  {"abc,abd->dc", &EinsumPass::HandleABCxABD2DC},
  {"abc,dec->abde", &EinsumPass::HandleStaticABCxDEC2ABDE},
  {"abc,abde->dec", &EinsumPass::HandleStaticABCxABDE2DEC},
  {"abcd,aecd->acbe", &EinsumPass::HandleABCDxAECD2ACBE},
  {"abcd,acbe->aecd", &EinsumPass::HandleABCDxACBE2AECD},
  {"abcd,ecd->abe", &EinsumPass::HandleABCDxECD2ABE},
  {"abcd,abe->ecd", &EinsumPass::HandleStaticABCDxABE2ECD},
  {"abcd,acbe->adbe", &EinsumPass::HandleABCDxACBE2ADBE}
};

std::unordered_map<std::string, EinsumPass::ProcFunc> EinsumPass::dynamicShapeProcs_ = {
  {"abc,cde->abde", &EinsumPass::HandleDynamicABCxCDE2ABDE},
  {"abcd,aecd->aceb", &EinsumPass::HandleABCDxAECD2ACEB},
  {"abcd,adbe->acbe", &EinsumPass::HandleABCDxADBE2ACBE},
  {"abcd,cde->abe", &EinsumPass::HandleABCDxCDE2ABE},
  {"abc,cd->abd", &EinsumPass::HandleABCxCD2ABD},
  {"abc,dc->abd", &EinsumPass::HandleABCxDC2ABD},
  {"abc,abd->dc", &EinsumPass::HandleABCxABD2DC},
  {"abc,dec->abde", &EinsumPass::HandleDynamicABCxDEC2ABDE},
  {"abc,abde->dec", &EinsumPass::HandleDynamicABCxABDE2DEC},
  {"abcd,aecd->acbe", &EinsumPass::HandleABCDxAECD2ACBE},
  {"abcd,acbe->aecd", &EinsumPass::HandleABCDxACBE2AECD},
  {"abcd,ecd->abe", &EinsumPass::HandleABCDxECD2ABE},
  {"abcd,abe->ecd", &EinsumPass::HandleDynamicABCDxABE2ECD},
  {"abcd,acbe->adbe", &EinsumPass::HandleABCDxACBE2ADBE}
};

vector<FusionPattern *> EinsumPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = new (nothrow) FusionPattern("EinsumPass");
  FUSION_PASS_CHECK(pattern == nullptr, CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(kPatternFusedNode, {kEinsum}).SetOutput(kPatternFusedNode);
  patterns.push_back(pattern);
  return patterns;
}

std::shared_ptr<ge::OpDesc> EinsumPass::CreateTransposeOpDesc(bool unknown_shape, const NodePtr &node,
                                                              const std::string &op_name) {
  std::shared_ptr<ge::OpDesc> transpose_desc = nullptr;
  if (unknown_shape) {
    FUSION_PASS_MAKE_SHARED(transpose_desc = std::make_shared<ge::OpDesc>(node->GetName() + op_name, kTranspose),
                            return nullptr);
  } else {
    FUSION_PASS_MAKE_SHARED(transpose_desc = std::make_shared<ge::OpDesc>(node->GetName() + op_name, kTransposeD),
                            return nullptr);
  }

  return transpose_desc;
}

bool EinsumPass::SetTransposePerm(bool unknown_shape, const std::vector<int32_t> &perm, ge::ComputeGraph &graph,
                                  std::shared_ptr<ge::OpDesc> &transpose_desc, ge::NodePtr &transpose_node) {
  if (unknown_shape) {
    std::vector<std::string> perm_depends({"perm"});
    transpose_desc->SetOpInferDepends(perm_depends);
    vector<int64_t> perm_shape({static_cast<int64_t>(perm.size())});
    auto perm_input_desc = ge::GeTensorDesc(ge::GeShape(perm_shape), ge::FORMAT_ND, ge::DT_INT32);
    ge::GeTensorPtr out_tensor = nullptr;
    FUSION_PASS_MAKE_SHARED(out_tensor = std::make_shared<ge::GeTensor>(perm_input_desc), return false);
    out_tensor->SetData(reinterpret_cast<const uint8_t *>(perm.data()), perm.size() * sizeof(int32_t));
    ge::OpDescPtr out_op_desc = ge::OpDescUtils::CreateConstOp(out_tensor);
    FUSION_PASS_CHECK(out_op_desc == nullptr, OP_LOGE(kFusedOpType.c_str(), "out_op_desc is null"), return false);
    auto const_node = graph.AddNode(out_op_desc);
    FUSION_PASS_CHECK(const_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "const_node is null"), return false);
    FUSION_PASS_CHECK(transpose_node->AddLinkFrom("perm", const_node) != SUCCESS,
                      OP_LOGE(kFusedOpType.c_str(), "Failed to add perm."), return false);

    transpose_desc->MutableInputDesc(1)->SetOriginShape(transpose_desc->MutableInputDesc(1)->MutableShape());
    return true;
  } else {
    return AttrUtils::SetListInt(transpose_desc, "perm", perm);
  }
}

std::shared_ptr<ge::OpDesc> EinsumPass::CreateReshapeOpDesc(bool unknown_shape, const NodePtr &node, uint32_t seq) {
  std::shared_ptr<ge::OpDesc> reshape_desc = nullptr;
  if (unknown_shape) {
    FUSION_PASS_MAKE_SHARED(
        reshape_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Flatten" + to_string(seq), kFlatten),
        return nullptr);
  } else {
    FUSION_PASS_MAKE_SHARED(
        reshape_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Reshape" + to_string(seq), kReshape),
        return nullptr);
  }

  return reshape_desc;
}

NodePtr EinsumPass::CreateReshapeNode(const std::vector<int64_t> &dims, ge::ComputeGraph &graph,
                                      std::shared_ptr<ge::OpDesc> &reshape_desc, int32_t axis, int32_t end_axis) {
  if (reshape_desc->GetType() == kFlatten) {
    // dynamic shape use FlattenV2
    AttrUtils::SetInt(reshape_desc, "axis", axis);
    AttrUtils::SetInt(reshape_desc, "end_axis", end_axis);
    auto reshape_node = graph.AddNode(reshape_desc);
    FUSION_PASS_CHECK(reshape_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "reshape_node is null"), return nullptr);
    FUSION_PASS_CHECK(reshape_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
                      OP_LOGE(kFusedOpType.c_str(), "FlattenV2 infershape failed."), return nullptr);

    return reshape_node;
  } else {
    // generate assist
    GeTensorDesc assist_desc;
    assist_desc.SetDataType(DT_INT32);
    assist_desc.SetFormat(FORMAT_ND);
    GeTensorPtr assist_ptr = nullptr;

    // geneate is_input_const
    static vector<bool> is_input_const = {false, true};

    // init const
    unique_ptr<int32_t[]> input_assist(new (nothrow) int32_t[dims.size()]());
    FUSION_PASS_CHECK(input_assist == nullptr, CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "input_assist is NULL"),
                      return nullptr);

    AssistIntHelp(dims, input_assist.get());

    // add const
    assist_desc.SetShape(GeShape({static_cast<int64_t>(dims.size())}));
    FUSION_PASS_MAKE_SHARED(
        (assist_ptr = make_shared<GeTensor>(assist_desc, reinterpret_cast<uint8_t *>(input_assist.get()),
                                            dims.size() * sizeof(int32_t))),
        return nullptr);
    vector<GeTensorPtr> weights = {assist_ptr};
    // create reshape node
    NodePtr reshape_node = graph.AddNode(reshape_desc);
    FUSION_PASS_CHECK(reshape_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "reshape_node is null"), return nullptr);
    OpDescUtils::SetWeights(reshape_node, weights);
    auto const_nodes = OpDescUtils::GetConstInputs(reshape_node);
    FUSION_PASS_CHECK(const_nodes.empty(), OP_LOGE(kFusedOpType.c_str(), "const_nodes is empty"), return nullptr);
    NodePtr &const_node = const_nodes[0];
    const_node->GetOpDesc()->SetType("Constant");
    reshape_desc->SetIsInputConst(is_input_const);

    return reshape_node;
  }
}

Status EinsumPass::LinkNode(ge::OutDataAnchor::Vistor<ge::InDataAnchorPtr> &anchors, ge::NodePtr &node) {
  for (size_t i = 0; i < anchors.size(); ++i) {
    auto out_anchor_peer_anchor = anchors.at(i);
    auto out_anchor_peer_node = out_anchor_peer_anchor->GetOwnerNode();
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(node->GetOutDataAnchor(0), out_anchor_peer_anchor),
                      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                            node->GetName().c_str(), out_anchor_peer_node->GetName().c_str()),
                      return FAILED);
  }

  return SUCCESS;
}

void EinsumPass::UnlinkAll(ge::NodePtr &node) {
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
}

Status EinsumPass::RelinkMatmulNode(ge::NodePtr &origin_node, ge::NodePtr &input0, ge::NodePtr &input1,
                                    ge::NodePtr &matmul_node, bool swap_input) {
  auto x0_anchor_peer_anchor = origin_node->GetInDataAnchor(0)->GetPeerOutAnchor();
  auto x0_anchor_peer_node = x0_anchor_peer_anchor->GetOwnerNode();
  auto x1_anchor_peer_anchor = origin_node->GetInDataAnchor(1)->GetPeerOutAnchor();
  auto x1_anchor_peer_node = x1_anchor_peer_anchor->GetOwnerNode();
  auto out_anchor_peer_anchors = origin_node->GetOutDataAnchor(0)->GetPeerInDataAnchors();

  // unlink
  UnlinkAll(origin_node);

  // add edge
  FUSION_PASS_CHECK(GraphUtils::AddEdge(x0_anchor_peer_anchor, input0->GetInDataAnchor(0)) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x0_anchor_peer_node->GetName().c_str(), input0->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(GraphUtils::AddEdge(x1_anchor_peer_anchor, input1->GetInDataAnchor(0)) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x1_anchor_peer_node->GetName().c_str(), input1->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(
      GraphUtils::AddEdge(input0->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(swap_input ? 1 : 0)) != SUCCESS,
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            input0->GetName().c_str(), matmul_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      GraphUtils::AddEdge(input1->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(swap_input ? 0 : 1)) != SUCCESS,
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            input1->GetName().c_str(), matmul_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(LinkNode(out_anchor_peer_anchors, matmul_node) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "link einsum node failed"), return FAILED);

  return SUCCESS;
}

Status EinsumPass::HandleStaticABCxCDE2ABDE(ComputeGraph &graph, NodePtr &node) {
  // 001:reshape+reshape+matmul+reshape
  // get op desc
  OpDescPtr op_desc = node->GetOpDesc();

  // get input
  GeTensorDesc x0_desc = op_desc->GetInputDesc(0);
  std::vector<int64_t> x0_dims = x0_desc.MutableShape().GetDims();
  GeTensorDesc x1_desc = op_desc->GetInputDesc(1);
  std::vector<int64_t> x1_dims = x1_desc.MutableShape().GetDims();

  // common vars
  auto x0_anchor_peer_anchor = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  auto x0_anchor_peer_node = x0_anchor_peer_anchor->GetOwnerNode();
  auto x1_anchor_peer_anchor = node->GetInDataAnchor(1)->GetPeerOutAnchor();
  auto x1_anchor_peer_node = x1_anchor_peer_anchor->GetOwnerNode();
  auto out_anchor_peer_anchors = node->GetOutDataAnchor(0)->GetPeerInDataAnchors();

  // create matmul op desc
  std::shared_ptr<ge::OpDesc> matmul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(matmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/MatMul", kMatMul),
                          return PARAM_INVALID);
  // create reshape op desc
  std::shared_ptr<ge::OpDesc> reshape_1_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_1_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Reshape1", kReshape),
                          return PARAM_INVALID);
  std::shared_ptr<ge::OpDesc> reshape_2_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_2_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Reshape2", kReshape),
                          return PARAM_INVALID);
  std::shared_ptr<ge::OpDesc> reshape_3_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_3_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Reshape3", kReshape),
                          return PARAM_INVALID);

  FUSION_PASS_CHECK((x0_dims.size() != 3) && (x1_dims.size() != 3),
                    OP_LOGI(kFusedOpType.c_str(), "input dims size must be three and three."), return NOT_CHANGED);

  // add input and output desc
  reshape_1_desc->AddInputDesc("x", x0_desc);
  std::vector<int64_t> tmp_dims({x0_dims[0] * x0_dims[1], x0_dims[2]});
  x0_desc.SetShape(GeShape(tmp_dims));
  x0_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_1_desc->AddOutputDesc("y", x0_desc);
  NodePtr reshape_1_node = CreateReshapeNode(tmp_dims, graph, reshape_1_desc);

  matmul_desc->AddInputDesc("x1", x0_desc);
  reshape_2_desc->AddInputDesc("x", x1_desc);
  tmp_dims.assign({x1_dims[0], x1_dims[1] * x1_dims[2]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_2_desc->AddOutputDesc("y", x1_desc);
  NodePtr reshape_2_node = CreateReshapeNode(tmp_dims, graph, reshape_2_desc);

  matmul_desc->AddInputDesc("x2", x1_desc);
  tmp_dims.assign({x0_dims[0] * x0_dims[1], x1_dims[1] * x1_dims[2]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  matmul_desc->AddOutputDesc("y", x1_desc);
  reshape_3_desc->AddInputDesc("x", x1_desc);
  tmp_dims.assign({x0_dims[0], x0_dims[1], x1_dims[1], x1_dims[2]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_3_desc->AddOutputDesc("y", x1_desc);
  NodePtr reshape_3_node = CreateReshapeNode(tmp_dims, graph, reshape_3_desc);

  // create matmul node
  NodePtr matmul_node = graph.AddNode(matmul_desc);
  // set matmul op attr
  AttrUtils::SetBool(matmul_desc, "transpose_x1", false);
  AttrUtils::SetBool(matmul_desc, "transpose_x2", false);

  // unlink
  UnlinkAll(node);

  // add edge
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, reshape_1_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x0_anchor_peer_node->GetName().c_str(), reshape_1_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, reshape_2_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x1_anchor_peer_node->GetName().c_str(), reshape_2_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(reshape_1_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            reshape_1_node->GetName().c_str(), matmul_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(reshape_2_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            reshape_2_node->GetName().c_str(), matmul_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), reshape_3_node->GetInDataAnchor(0)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            matmul_node->GetName().c_str(), reshape_3_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(LinkNode(out_anchor_peer_anchors, reshape_3_node) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "link einsum node failed"), return FAILED);
  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "remove einsum node failed"), return FAILED);
  return SUCCESS;
}

Status EinsumPass::HandleDynamicABCxCDE2ABDE(ComputeGraph &graph, NodePtr &node) {
  // 001:reshape+reshape+matmul+reshape
  // get op desc
  OpDescPtr op_desc = node->GetOpDesc();

  // get input
  GeTensorDesc x0_desc = op_desc->GetInputDesc(0);
  std::vector<int64_t> x0_dims = x0_desc.MutableShape().GetDims();
  GeTensorDesc x1_desc = op_desc->GetInputDesc(1);
  std::vector<int64_t> x1_dims = x1_desc.MutableShape().GetDims();
  bool x0_is_unknown_shape = x0_desc.MutableShape().IsUnknownShape();
  bool x1_is_unknown_shape = x1_desc.MutableShape().IsUnknownShape();

  FUSION_PASS_CHECK((x0_dims.size() != 3) && (x1_dims.size() != 3),
                    OP_LOGI(kFusedOpType.c_str(), "input dims size must be three and three."), return NOT_CHANGED);

  // common vars
  std::vector<int64_t> tmp_dims;
  auto x0_anchor_peer_anchor = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  auto x0_anchor_peer_node = x0_anchor_peer_anchor->GetOwnerNode();
  auto x1_anchor_peer_anchor = node->GetInDataAnchor(1)->GetPeerOutAnchor();
  auto x1_anchor_peer_node = x1_anchor_peer_anchor->GetOwnerNode();
  auto out_anchor_peer_anchors = node->GetOutDataAnchor(0)->GetPeerInDataAnchors();

  // create GatherShapes op desc
  std::shared_ptr<ge::OpDesc> gatherShapes_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      gatherShapes_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/GatherShapes", kGatherShapes),
      return PARAM_INVALID);

  // create reshape op desc
  std::shared_ptr<ge::OpDesc> reshape_1_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_1_desc = CreateReshapeOpDesc(x0_is_unknown_shape, node, 1), return PARAM_INVALID);
  std::shared_ptr<ge::OpDesc> reshape_2_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_2_desc = CreateReshapeOpDesc(x1_is_unknown_shape, node, 2), return PARAM_INVALID);

  // add input and output desc
  gatherShapes_desc->AddInputDesc("x0", x0_desc);
  gatherShapes_desc->AddInputDesc("x1", x1_desc);
  const std::vector<std::vector<int64_t>> axes = {{0, 0}, {0, 1}, {1, 1}, {1, 2}};
  FUSION_PASS_CHECK(AttrUtils::SetListListInt(gatherShapes_desc, "axes", axes) == false,
                    OP_LOGE(kFusedOpType.c_str(), "set gatherShapes axes failed."), return FAILED);

  reshape_1_desc->AddInputDesc("x", x0_desc);
  tmp_dims.assign({GetDimMulValue(x0_dims[0], x0_dims[1]), x0_dims[2]});
  x0_desc.SetShape(GeShape(tmp_dims));
  x0_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_1_desc->AddOutputDesc("y", x0_desc);
  NodePtr reshape_1_node = CreateReshapeNode(tmp_dims, graph, reshape_1_desc, 0, 1);
  FUSION_PASS_CHECK(reshape_1_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "reshape_1_node is null"), return FAILED);

  reshape_2_desc->AddInputDesc("x", x1_desc);
  tmp_dims.assign({x1_dims[0], GetDimMulValue(x1_dims[1], x1_dims[2])});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_2_desc->AddOutputDesc("y", x1_desc);
  NodePtr reshape_2_node = CreateReshapeNode(tmp_dims, graph, reshape_2_desc, 1, 2);
  FUSION_PASS_CHECK(reshape_2_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "reshape_2_node is null"), return FAILED);

  // init GatherShapes op desc
  tmp_dims.assign({x0_dims[0], x0_dims[1], x1_dims[1], x1_dims[2]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  gatherShapes_desc->AddOutputDesc("shape", x1_desc);
  NodePtr gatherShapes_node = graph.AddNode(gatherShapes_desc);
  FUSION_PASS_CHECK(gatherShapes_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "gatherShapes_node is null"),
                    return FAILED);
  if (gatherShapes_node->InferShapeAndType() != ge::GRAPH_SUCCESS) {
    OP_LOGE(kFusedOpType.c_str(), "gatherShapes infershape failed.");
    return FAILED;
  }

  // create matmul op desc
  std::shared_ptr<ge::OpDesc> matmul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(matmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/MatMul", kMatMul),
                          return PARAM_INVALID);
  matmul_desc->AddInputDesc("x1", *(reshape_1_desc->MutableOutputDesc(0)));
  matmul_desc->AddInputDesc("x2", *(reshape_2_desc->MutableOutputDesc(0)));
  tmp_dims.assign({GetDimMulValue(x0_dims[0], x0_dims[1]), GetDimMulValue(x1_dims[1], x1_dims[2])});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  matmul_desc->AddOutputDesc("y", x1_desc);
  AttrUtils::SetBool(matmul_desc, "transpose_x1", false);
  AttrUtils::SetBool(matmul_desc, "transpose_x2", false);
  NodePtr matmul_node = graph.AddNode(matmul_desc);
  FUSION_PASS_CHECK(matmul_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "matmul_node is null"), return FAILED);
  if (matmul_node->InferShapeAndType() != ge::GRAPH_SUCCESS) {
    OP_LOGE(kFusedOpType.c_str(), "matmul infershape failed.");
    return FAILED;
  }

  // create reshape op desc
  std::shared_ptr<ge::OpDesc> reshape_3_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_3_desc = CreateReshapeOpDesc(false, node, 3), return PARAM_INVALID);
  reshape_3_desc->AddInputDesc("x", *(matmul_desc->MutableOutputDesc(0)));
  reshape_3_desc->AddInputDesc("shape", *(gatherShapes_desc->MutableOutputDesc(0)));
  reshape_3_desc->AddOutputDesc("y", *(op_desc->MutableOutputDesc(0)));
  // geneate is_input_const
  static vector<bool> is_input_const = {false, false};
  reshape_3_desc->SetIsInputConst(is_input_const);
  NodePtr reshape_3_node = graph.AddNode(reshape_3_desc);
  FUSION_PASS_CHECK(reshape_3_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "reshape_3_node is null"), return FAILED);
  if (reshape_3_node->InferShapeAndType() != ge::GRAPH_SUCCESS) {
    OP_LOGE(kFusedOpType.c_str(), "reshape infershape failed.");
    return FAILED;
  }

  // unlink
  UnlinkAll(node);

  // add edge
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, gatherShapes_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x0_anchor_peer_node->GetName().c_str(), gatherShapes_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, gatherShapes_node->GetInDataAnchor(1)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x1_anchor_peer_node->GetName().c_str(), gatherShapes_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, reshape_1_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x0_anchor_peer_node->GetName().c_str(), reshape_1_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, reshape_2_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x1_anchor_peer_node->GetName().c_str(), reshape_2_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(reshape_1_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            reshape_1_node->GetName().c_str(), matmul_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(reshape_2_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            reshape_2_node->GetName().c_str(), matmul_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), reshape_3_node->GetInDataAnchor(0)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            matmul_node->GetName().c_str(), reshape_3_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(gatherShapes_node->GetOutDataAnchor(0), reshape_3_node->GetInDataAnchor(1)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            gatherShapes_node->GetName().c_str(), reshape_3_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(LinkNode(out_anchor_peer_anchors, reshape_3_node) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "link einsum node failed"), return FAILED);
  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "remove einsum node failed"), return FAILED);

  return SUCCESS;
}

Status EinsumPass::HandleABCDxAECD2ACEB(ComputeGraph &graph, NodePtr &node) {
  // 002 transpose+transpose+batchmatmul(swap input)
  // get op desc
  OpDescPtr op_desc = node->GetOpDesc();

  // get input
  GeTensorDesc x0_desc = op_desc->GetInputDesc(0);
  std::vector<int64_t> x0_dims = x0_desc.MutableShape().GetDims();
  GeTensorDesc x1_desc = op_desc->GetInputDesc(1);
  std::vector<int64_t> x1_dims = x1_desc.MutableShape().GetDims();
  bool x0_is_unknown_shape = x0_desc.MutableShape().IsUnknownShape();
  bool x1_is_unknown_shape = x1_desc.MutableShape().IsUnknownShape();

  // create batchmatmulv2 op desc
  std::shared_ptr<ge::OpDesc> batchmatmul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      batchmatmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/BatchMatMul", kBatchMatMul),
      return PARAM_INVALID);
  // create transpose op desc
  std::shared_ptr<ge::OpDesc> transpose_1_desc = CreateTransposeOpDesc(x0_is_unknown_shape, node, "/Transpose1");
  std::shared_ptr<ge::OpDesc> transpose_2_desc = CreateTransposeOpDesc(x1_is_unknown_shape, node, "/Transpose2");

  FUSION_PASS_CHECK((x0_dims.size() != 4) && (x1_dims.size() != 4),
                    OP_LOGI(kFusedOpType.c_str(), "input dims size must be four and four."), return NOT_CHANGED);

  // set op attr
  AttrUtils::SetBool(batchmatmul_desc, "adj_x1", false);
  AttrUtils::SetBool(batchmatmul_desc, "adj_x2", true);
  std::vector<int32_t> perm({0, 2, 1, 3});

  // add input and output desc
  transpose_2_desc->AddInputDesc("x", x1_desc);
  std::vector<int64_t> tmp_dims({x1_dims[0], x1_dims[2], x1_dims[1], x1_dims[3]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  transpose_2_desc->AddOutputDesc("y", x1_desc);
  NodePtr transpose_2_node = graph.AddNode(transpose_2_desc);
  SetTransposePerm(x1_is_unknown_shape, perm, graph, transpose_2_desc, transpose_2_node);
  FUSION_PASS_CHECK(x1_is_unknown_shape && transpose_2_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "transpose infershape failed."), return FAILED);

  transpose_1_desc->AddInputDesc("x", x0_desc);
  tmp_dims.assign({x0_dims[0], x0_dims[2], x0_dims[1], x0_dims[3]});
  x0_desc.SetShape(GeShape(tmp_dims));
  x0_desc.SetOriginShape(GeShape(tmp_dims));
  transpose_1_desc->AddOutputDesc("y", x0_desc);
  NodePtr transpose_1_node = graph.AddNode(transpose_1_desc);
  SetTransposePerm(x0_is_unknown_shape, perm, graph, transpose_1_desc, transpose_1_node);
  FUSION_PASS_CHECK(x0_is_unknown_shape && transpose_1_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "transpose infershape failed."), return FAILED);

  batchmatmul_desc->AddInputDesc("x1", *(transpose_2_desc->MutableOutputDesc(0)));
  batchmatmul_desc->AddInputDesc("x2", *(transpose_1_desc->MutableOutputDesc(0)));
  batchmatmul_desc->AddOutputDesc("y", *(op_desc->MutableOutputDesc(0)));

  // create batchmatmul node
  NodePtr batchmatmul_node = graph.AddNode(batchmatmul_desc);
  FUSION_PASS_CHECK(RelinkMatmulNode(node, transpose_1_node, transpose_2_node, batchmatmul_node, true) != SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "failed to relink transpose/batchmatmul"), return FAILED);
  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "remove einsum node failed"), return FAILED);
  return SUCCESS;
}

Status EinsumPass::HandleABCDxADBE2ACBE(ComputeGraph &graph, NodePtr &node) {
  // 003:transpose+batchmatmul+transpose
  // get op desc
  OpDescPtr op_desc = node->GetOpDesc();

  // get input
  GeTensorDesc x0_desc = op_desc->GetInputDesc(0);
  std::vector<int64_t> x0_dims = x0_desc.MutableShape().GetDims();
  GeTensorDesc x1_desc = op_desc->GetInputDesc(1);
  std::vector<int64_t> x1_dims = x1_desc.MutableShape().GetDims();
  bool x0_is_unknown_shape = x0_desc.MutableShape().IsUnknownShape();
  bool x1_is_unknown_shape = x1_desc.MutableShape().IsUnknownShape();

  // common vars
  auto x0_anchor_peer_anchor = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  auto x0_anchor_peer_node = x0_anchor_peer_anchor->GetOwnerNode();
  auto x1_anchor_peer_anchor = node->GetInDataAnchor(1)->GetPeerOutAnchor();
  auto x1_anchor_peer_node = x1_anchor_peer_anchor->GetOwnerNode();
  auto out_anchor_peer_anchors = node->GetOutDataAnchor(0)->GetPeerInDataAnchors();

  // create batchmatmulv2 op desc
  std::shared_ptr<ge::OpDesc> batchmatmul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      batchmatmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/BatchMatMul", kBatchMatMul),
      return PARAM_INVALID);
  // create transpose op desc
  std::shared_ptr<ge::OpDesc> transpose_1_desc = CreateTransposeOpDesc(x1_is_unknown_shape, node, "/Transpose1");

  FUSION_PASS_CHECK((x0_dims.size() != 4) && (x1_dims.size() != 4),
                    OP_LOGI(kFusedOpType.c_str(), "input dims size must be four and four."), return NOT_CHANGED);

  // set op attr
  AttrUtils::SetBool(batchmatmul_desc, "adj_x1", false);
  AttrUtils::SetBool(batchmatmul_desc, "adj_x2", false);
  std::vector<int32_t> perm({0, 2, 1, 3});

  // add input and output desc
  batchmatmul_desc->AddInputDesc("x1", x0_desc);
  transpose_1_desc->AddInputDesc("x", x1_desc);
  std::vector<int64_t> tmp_dims({x1_dims[0], x1_dims[2], x1_dims[1], x1_dims[3]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  transpose_1_desc->AddOutputDesc("y", x1_desc);
  NodePtr transpose_1_node = graph.AddNode(transpose_1_desc);
  SetTransposePerm(x1_is_unknown_shape, perm, graph, transpose_1_desc, transpose_1_node);
  FUSION_PASS_CHECK(x1_is_unknown_shape && transpose_1_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "transpose infershape failed."), return FAILED);

  batchmatmul_desc->AddInputDesc("x2", *(transpose_1_desc->MutableOutputDesc(0)));
  tmp_dims.assign({x0_dims[0], x0_dims[1], x0_dims[2], x1_dims[3]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  batchmatmul_desc->AddOutputDesc("y", x1_desc);
  NodePtr batchmatmul_node = graph.AddNode(batchmatmul_desc);
  FUSION_PASS_CHECK((x0_is_unknown_shape || transpose_1_desc->MutableOutputDesc(0)->MutableShape().IsUnknownShape()) &&
                        batchmatmul_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "batchmatmul infershape failed."), return FAILED);

  bool is_unknown_shape = batchmatmul_desc->MutableOutputDesc(0)->MutableShape().IsUnknownShape();
  std::shared_ptr<ge::OpDesc> transpose_2_desc = CreateTransposeOpDesc(is_unknown_shape, node, "/Transpose2");
  transpose_2_desc->AddInputDesc("x", *(batchmatmul_desc->MutableOutputDesc(0)));
  transpose_2_desc->AddOutputDesc("y", *(op_desc->MutableOutputDesc(0)));
  NodePtr transpose_2_node = graph.AddNode(transpose_2_desc);
  SetTransposePerm(is_unknown_shape, perm, graph, transpose_2_desc, transpose_2_node);

  // unlink
  UnlinkAll(node);

  // add edge
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, batchmatmul_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x0_anchor_peer_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, transpose_1_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x1_anchor_peer_node->GetName().c_str(), transpose_1_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(transpose_1_node->GetOutDataAnchor(0), batchmatmul_node->GetInDataAnchor(1)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            transpose_1_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(batchmatmul_node->GetOutDataAnchor(0), transpose_2_node->GetInDataAnchor(0)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            batchmatmul_node->GetName().c_str(), transpose_2_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(LinkNode(out_anchor_peer_anchors, transpose_2_node) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "link einsum node failed"), return FAILED);
  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "remove einsum node failed"), return FAILED);
  return SUCCESS;
}

Status EinsumPass::HandleABCDxCDE2ABE(ComputeGraph &graph, NodePtr &node) {
  // 004:reshape+reshape+matmul+reshape-->reshape+batchmatmul
  // get op desc
  OpDescPtr op_desc = node->GetOpDesc();

  // get input
  GeTensorDesc x0_desc = op_desc->GetInputDesc(0);
  std::vector<int64_t> x0_dims = x0_desc.MutableShape().GetDims();
  GeTensorDesc x1_desc = op_desc->GetInputDesc(1);
  std::vector<int64_t> x1_dims = x1_desc.MutableShape().GetDims();
  bool x0_is_unknown_shape = x0_desc.MutableShape().IsUnknownShape();
  bool x1_is_unknown_shape = x1_desc.MutableShape().IsUnknownShape();

  // create batchmatmulv2 op desc
  std::shared_ptr<ge::OpDesc> batchmatmul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      batchmatmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/BatchMatMul", kBatchMatMul),
      return PARAM_INVALID);
  // create reshape op desc
  std::shared_ptr<ge::OpDesc> reshape_1_desc = CreateReshapeOpDesc(x0_is_unknown_shape, node, 1);
  std::shared_ptr<ge::OpDesc> reshape_2_desc = CreateReshapeOpDesc(x1_is_unknown_shape, node, 2);

  FUSION_PASS_CHECK((x0_dims.size() != 4) && (x1_dims.size() != 3),
                    OP_LOGI(kFusedOpType.c_str(), "input dims size must be four and three."), return NOT_CHANGED);

  // add input and output desc
  reshape_1_desc->AddInputDesc("x", x0_desc);
  std::vector<int64_t> tmp_dims({x0_dims[0], x0_dims[1], GetDimMulValue(x0_dims[2], x0_dims[3])});
  x0_desc.SetShape(GeShape(tmp_dims));
  x0_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_1_desc->AddOutputDesc("y", x0_desc);
  NodePtr reshape_1_node = CreateReshapeNode(tmp_dims, graph, reshape_1_desc, 2, 3);

  reshape_2_desc->AddInputDesc("x", x1_desc);
  tmp_dims.assign({GetDimMulValue(x1_dims[0], x1_dims[1]), x1_dims[2]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_2_desc->AddOutputDesc("y", x1_desc);
  NodePtr reshape_2_node = CreateReshapeNode(tmp_dims, graph, reshape_2_desc, 0, 1);

  batchmatmul_desc->AddInputDesc("x1", *(reshape_1_desc->MutableOutputDesc(0)));
  batchmatmul_desc->AddInputDesc("x2", *(reshape_2_desc->MutableOutputDesc(0)));
  batchmatmul_desc->AddOutputDesc("y", *(op_desc->MutableOutputDesc(0)));
  // create batchmatmul node
  NodePtr batchmatmul_node = graph.AddNode(batchmatmul_desc);
  // set batchmatmul op attr
  AttrUtils::SetBool(batchmatmul_desc, "adj_x1", false);
  AttrUtils::SetBool(batchmatmul_desc, "adj_x2", false);

  FUSION_PASS_CHECK(RelinkMatmulNode(node, reshape_1_node, reshape_2_node, batchmatmul_node, false) != SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "failed to relink reshape/batchmatmul"), return FAILED);

  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "remove einsum node failed"), return FAILED);
  return SUCCESS;
}

Status EinsumPass::HandleBatchMatmul(bool adj_x2, ComputeGraph &graph, NodePtr &node) {
  // 005:reshape+matmul+reshape-->batchmatmul
  // 006:reshape+matmul+reshape-->batchmatmul

  // get op desc
  OpDescPtr op_desc = node->GetOpDesc();

  // common vars
  auto x0_anchor_peer_anchor = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  auto x0_anchor_peer_node = x0_anchor_peer_anchor->GetOwnerNode();
  auto x1_anchor_peer_anchor = node->GetInDataAnchor(1)->GetPeerOutAnchor();
  auto x1_anchor_peer_node = x1_anchor_peer_anchor->GetOwnerNode();
  auto out_anchor_peer_anchors = node->GetOutDataAnchor(0)->GetPeerInDataAnchors();

  // create batchmatmulv2 op desc
  std::shared_ptr<ge::OpDesc> batchmatmul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      batchmatmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/BatchMatMul", kBatchMatMul),
      return PARAM_INVALID);

  // add input and output desc
  batchmatmul_desc->AddInputDesc("x1", *(op_desc->MutableInputDesc(0)));
  batchmatmul_desc->AddInputDesc("x2", *(op_desc->MutableInputDesc(1)));
  batchmatmul_desc->AddOutputDesc("y", *(op_desc->MutableOutputDesc(0)));

  // create batchmatmul node
  NodePtr batchmatmul_node = graph.AddNode(batchmatmul_desc);
  // set batchmatmul op attr
  AttrUtils::SetBool(batchmatmul_desc, "adj_x1", false);
  AttrUtils::SetBool(batchmatmul_desc, "adj_x2", adj_x2);
  // unlink
  UnlinkAll(node);

  // add edge
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, batchmatmul_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x0_anchor_peer_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, batchmatmul_node->GetInDataAnchor(1)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x1_anchor_peer_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(LinkNode(out_anchor_peer_anchors, batchmatmul_node) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "link einsum node failed"), return FAILED);
  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "remove einsum node failed"), return FAILED);
  return SUCCESS;
}

Status EinsumPass::HandleABCxCD2ABD(ComputeGraph &graph, NodePtr &node) {
  // 005:reshape+matmul+reshape-->batchmatmul
  return HandleBatchMatmul(false, graph, node);
}

Status EinsumPass::HandleABCxDC2ABD(ComputeGraph &graph, NodePtr &node) {
  // 006:reshape+matmul+reshape-->batchmatmul
  return HandleBatchMatmul(true, graph, node);
}

Status EinsumPass::HandleABCxABD2DC(ComputeGraph &graph, NodePtr &node) {
  // 007:reshape+reshape+matmul(swap input)
  // get op desc
  OpDescPtr op_desc = node->GetOpDesc();

  // get input
  GeTensorDesc x0_desc = op_desc->GetInputDesc(0);
  std::vector<int64_t> x0_dims = x0_desc.MutableShape().GetDims();
  GeTensorDesc x1_desc = op_desc->GetInputDesc(1);
  std::vector<int64_t> x1_dims = x1_desc.MutableShape().GetDims();
  bool x0_is_unknown_shape = x0_desc.MutableShape().IsUnknownShape();
  bool x1_is_unknown_shape = x1_desc.MutableShape().IsUnknownShape();

  // create matmul op desc
  std::shared_ptr<ge::OpDesc> matmul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(matmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/MatMul", kMatMul),
                          return PARAM_INVALID);
  // create reshape op desc
  std::shared_ptr<ge::OpDesc> reshape_1_desc = CreateReshapeOpDesc(x0_is_unknown_shape, node, 1);
  std::shared_ptr<ge::OpDesc> reshape_2_desc = CreateReshapeOpDesc(x1_is_unknown_shape, node, 2);

  FUSION_PASS_CHECK((x0_dims.size() != 3) && (x1_dims.size() != 3),
                    OP_LOGI(kFusedOpType.c_str(), "input dims size must be three and three."), return NOT_CHANGED);

  // add input and output desc
  reshape_2_desc->AddInputDesc("x", x1_desc);
  std::vector<int64_t> tmp_dims({GetDimMulValue(x1_dims[0], x1_dims[1]), x1_dims[2]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_2_desc->AddOutputDesc("y", x1_desc);
  NodePtr reshape_2_node = CreateReshapeNode(tmp_dims, graph, reshape_2_desc, 0, 1);

  reshape_1_desc->AddInputDesc("x", x0_desc);
  tmp_dims.assign({GetDimMulValue(x0_dims[0], x0_dims[1]), x0_dims[2]});
  x0_desc.SetShape(GeShape(tmp_dims));
  x0_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_1_desc->AddOutputDesc("y", x0_desc);
  NodePtr reshape_1_node = CreateReshapeNode(tmp_dims, graph, reshape_1_desc, 0, 1);

  matmul_desc->AddInputDesc("x1", *(reshape_2_desc->MutableOutputDesc(0)));
  matmul_desc->AddInputDesc("x2", *(reshape_1_desc->MutableOutputDesc(0)));
  matmul_desc->AddOutputDesc("y", *(op_desc->MutableOutputDesc(0)));
  // create matmul node
  NodePtr matmul_node = graph.AddNode(matmul_desc);
  // set op attr
  AttrUtils::SetBool(matmul_desc, "transpose_x1", true);
  AttrUtils::SetBool(matmul_desc, "transpose_x2", false);

  FUSION_PASS_CHECK(RelinkMatmulNode(node, reshape_1_node, reshape_2_node, matmul_node, true) != SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "failed to relink reshape/matmul"), return FAILED);
  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "remove einsum node failed"), return FAILED);
  return SUCCESS;
}

Status EinsumPass::HandleStaticABCxDEC2ABDE(ComputeGraph &graph, NodePtr &node) {
  // 008:reshape+reshape+matmul+reshape
  // get op desc
  OpDescPtr op_desc = node->GetOpDesc();

  // get input
  GeTensorDesc x0_desc = op_desc->GetInputDesc(0);
  std::vector<int64_t> x0_dims = x0_desc.MutableShape().GetDims();
  GeTensorDesc x1_desc = op_desc->GetInputDesc(1);
  std::vector<int64_t> x1_dims = x1_desc.MutableShape().GetDims();

  // common vars
  auto x0_anchor_peer_anchor = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  auto x0_anchor_peer_node = x0_anchor_peer_anchor->GetOwnerNode();
  auto x1_anchor_peer_anchor = node->GetInDataAnchor(1)->GetPeerOutAnchor();
  auto x1_anchor_peer_node = x1_anchor_peer_anchor->GetOwnerNode();
  auto out_anchor_peer_anchors = node->GetOutDataAnchor(0)->GetPeerInDataAnchors();

  // create matmul op desc
  std::shared_ptr<ge::OpDesc> matmul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(matmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/MatMul", kMatMul),
                          return PARAM_INVALID);
  // create reshape op desc
  std::shared_ptr<ge::OpDesc> reshape_1_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_1_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Reshape1", kReshape),
                          return PARAM_INVALID);
  std::shared_ptr<ge::OpDesc> reshape_2_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_2_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Reshape2", kReshape),
                          return PARAM_INVALID);
  std::shared_ptr<ge::OpDesc> reshape_3_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_3_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Reshape3", kReshape),
                          return PARAM_INVALID);

  FUSION_PASS_CHECK((x0_dims.size() != 3) && (x1_dims.size() != 3),
                    OP_LOGI(kFusedOpType.c_str(), "input dims size must be three and three."), return NOT_CHANGED);

  // add input and output desc
  reshape_1_desc->AddInputDesc("x", x0_desc);
  std::vector<int64_t> tmp_dims({x0_dims[0] * x0_dims[1], x0_dims[2]});
  x0_desc.SetShape(GeShape(tmp_dims));
  x0_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_1_desc->AddOutputDesc("y", x0_desc);
  NodePtr reshape_1_node = CreateReshapeNode(tmp_dims, graph, reshape_1_desc);

  matmul_desc->AddInputDesc("x1", x0_desc);
  reshape_2_desc->AddInputDesc("x", x1_desc);
  tmp_dims.assign({x1_dims[0] * x1_dims[1], x1_dims[2]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_2_desc->AddOutputDesc("y", x1_desc);
  NodePtr reshape_2_node = CreateReshapeNode(tmp_dims, graph, reshape_2_desc);

  matmul_desc->AddInputDesc("x2", x1_desc);
  tmp_dims.assign({x0_dims[0] * x0_dims[1], x1_dims[0] * x1_dims[1]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  matmul_desc->AddOutputDesc("y", x1_desc);

  reshape_3_desc->AddInputDesc("x", x1_desc);
  tmp_dims.assign({x0_dims[0], x0_dims[1], x1_dims[0], x1_dims[1]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_3_desc->AddOutputDesc("y", x1_desc);
  NodePtr reshape_3_node = CreateReshapeNode(tmp_dims, graph, reshape_3_desc);

  // create matmulnode
  NodePtr matmul_node = graph.AddNode(matmul_desc);
  // set op attr
  AttrUtils::SetBool(matmul_desc, "transpose_x1", false);
  AttrUtils::SetBool(matmul_desc, "transpose_x2", true);

  // unlink
  UnlinkAll(node);

  // add edge
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, reshape_1_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x0_anchor_peer_node->GetName().c_str(), reshape_1_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(reshape_1_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            reshape_1_node->GetName().c_str(), matmul_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, reshape_2_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x1_anchor_peer_node->GetName().c_str(), reshape_2_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(reshape_2_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            reshape_2_node->GetName().c_str(), matmul_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), reshape_3_node->GetInDataAnchor(0)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            matmul_node->GetName().c_str(), reshape_3_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(LinkNode(out_anchor_peer_anchors, reshape_3_node) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "link einsum node failed"), return FAILED);
  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "remove einsum node failed"), return FAILED);
  return SUCCESS;
}

Status EinsumPass::HandleDynamicABCxDEC2ABDE(ComputeGraph &graph, NodePtr &node) {
  // 008:reshape+reshape+matmul+reshape
  // get op desc
  OpDescPtr op_desc = node->GetOpDesc();

  // get input
  GeTensorDesc x0_desc = op_desc->GetInputDesc(0);
  std::vector<int64_t> x0_dims = x0_desc.MutableShape().GetDims();
  GeTensorDesc x1_desc = op_desc->GetInputDesc(1);
  std::vector<int64_t> x1_dims = x1_desc.MutableShape().GetDims();
  bool x0_is_unknown_shape = x0_desc.MutableShape().IsUnknownShape();
  bool x1_is_unknown_shape = x1_desc.MutableShape().IsUnknownShape();

  FUSION_PASS_CHECK((x0_dims.size() != 3) && (x1_dims.size() != 3),
                    OP_LOGI(kFusedOpType.c_str(), "input dims size must be three and three."), return NOT_CHANGED);

  // common vars
  std::vector<int64_t> tmp_dims;
  auto x0_anchor_peer_anchor = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  auto x0_anchor_peer_node = x0_anchor_peer_anchor->GetOwnerNode();
  auto x1_anchor_peer_anchor = node->GetInDataAnchor(1)->GetPeerOutAnchor();
  auto x1_anchor_peer_node = x1_anchor_peer_anchor->GetOwnerNode();
  auto out_anchor_peer_anchors = node->GetOutDataAnchor(0)->GetPeerInDataAnchors();

  // create GatherShapes op desc
  std::shared_ptr<ge::OpDesc> gatherShapes_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      gatherShapes_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/GatherShapes", kGatherShapes),
      return PARAM_INVALID);

  // create reshape op desc
  std::shared_ptr<ge::OpDesc> reshape_1_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_1_desc = CreateReshapeOpDesc(x0_is_unknown_shape, node, 1), return PARAM_INVALID);
  std::shared_ptr<ge::OpDesc> reshape_2_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_2_desc = CreateReshapeOpDesc(x1_is_unknown_shape, node, 2), return PARAM_INVALID);

  // add input and output desc
  gatherShapes_desc->AddInputDesc("x0", x0_desc);
  gatherShapes_desc->AddInputDesc("x1", x1_desc);
  const std::vector<std::vector<int64_t>> axes = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  FUSION_PASS_CHECK(AttrUtils::SetListListInt(gatherShapes_desc, "axes", axes) == false,
                    OP_LOGE(kFusedOpType.c_str(), "set gatherShapes axes failed."), return FAILED);

  reshape_1_desc->AddInputDesc("x", x0_desc);
  tmp_dims.assign({GetDimMulValue(x0_dims[0], x0_dims[1]), x0_dims[2]});
  x0_desc.SetShape(GeShape(tmp_dims));
  x0_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_1_desc->AddOutputDesc("y", x0_desc);
  NodePtr reshape_1_node = CreateReshapeNode(tmp_dims, graph, reshape_1_desc, 0, 1);
  FUSION_PASS_CHECK(reshape_1_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "reshape_1_node is null"), return FAILED);

  reshape_2_desc->AddInputDesc("x", x1_desc);
  tmp_dims.assign({GetDimMulValue(x1_dims[0], x1_dims[1]), x1_dims[2]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_2_desc->AddOutputDesc("y", x1_desc);
  NodePtr reshape_2_node = CreateReshapeNode(tmp_dims, graph, reshape_2_desc, 0, 1);
  FUSION_PASS_CHECK(reshape_2_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "reshape_2_node is null"), return FAILED);

  // init GatherShapes op desc
  tmp_dims.assign({x0_dims[0], x0_dims[1], x1_dims[0], x1_dims[1]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  gatherShapes_desc->AddOutputDesc("shape", x1_desc);
  NodePtr gatherShapes_node = graph.AddNode(gatherShapes_desc);
  FUSION_PASS_CHECK(gatherShapes_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "gatherShapes_node is null"),
                    return FAILED);
  if (gatherShapes_node->InferShapeAndType() != ge::GRAPH_SUCCESS) {
    OP_LOGE(kFusedOpType.c_str(), "gatherShapes infershape failed.");
    return FAILED;
  }

  // create matmul op desc
  std::shared_ptr<ge::OpDesc> matmul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(matmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/MatMul", kMatMul),
                          return PARAM_INVALID);
  matmul_desc->AddInputDesc("x1", *(reshape_1_desc->MutableOutputDesc(0)));
  matmul_desc->AddInputDesc("x2", *(reshape_2_desc->MutableOutputDesc(0)));
  tmp_dims.assign({GetDimMulValue(x0_dims[0], x0_dims[1]), GetDimMulValue(x1_dims[0], x1_dims[1])});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  matmul_desc->AddOutputDesc("y", x1_desc);
  AttrUtils::SetBool(matmul_desc, "transpose_x1", false);
  AttrUtils::SetBool(matmul_desc, "transpose_x2", true);
  NodePtr matmul_node = graph.AddNode(matmul_desc);
  FUSION_PASS_CHECK(matmul_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "matmul_node is null"), return FAILED);
  if (matmul_node->InferShapeAndType() != ge::GRAPH_SUCCESS) {
    OP_LOGE(kFusedOpType.c_str(), "matmul infershape failed.");
    return FAILED;
  }

  // create reshape op desc
  std::shared_ptr<ge::OpDesc> reshape_3_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_3_desc = CreateReshapeOpDesc(false, node, 3), return PARAM_INVALID);
  reshape_3_desc->AddInputDesc("x", *(matmul_desc->MutableOutputDesc(0)));
  reshape_3_desc->AddInputDesc("shape", *(gatherShapes_desc->MutableOutputDesc(0)));
  reshape_3_desc->AddOutputDesc("y", *(op_desc->MutableOutputDesc(0)));
  // geneate is_input_const
  static vector<bool> is_input_const = {false, false};
  reshape_3_desc->SetIsInputConst(is_input_const);
  NodePtr reshape_3_node = graph.AddNode(reshape_3_desc);
  FUSION_PASS_CHECK(reshape_3_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "reshape_3_node is null"), return FAILED);
  if (reshape_3_node->InferShapeAndType() != ge::GRAPH_SUCCESS) {
    OP_LOGE(kFusedOpType.c_str(), "reshape infershape failed.");
    return FAILED;
  }

  // unlink
  UnlinkAll(node);

  // add edge
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, gatherShapes_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x0_anchor_peer_node->GetName().c_str(), gatherShapes_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, gatherShapes_node->GetInDataAnchor(1)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x1_anchor_peer_node->GetName().c_str(), gatherShapes_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, reshape_1_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x0_anchor_peer_node->GetName().c_str(), reshape_1_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, reshape_2_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x1_anchor_peer_node->GetName().c_str(), reshape_2_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(reshape_1_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            reshape_1_node->GetName().c_str(), matmul_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(reshape_2_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            reshape_2_node->GetName().c_str(), matmul_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), reshape_3_node->GetInDataAnchor(0)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            matmul_node->GetName().c_str(), reshape_3_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(gatherShapes_node->GetOutDataAnchor(0), reshape_3_node->GetInDataAnchor(1)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            gatherShapes_node->GetName().c_str(), reshape_3_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(LinkNode(out_anchor_peer_anchors, reshape_3_node) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "link einsum node failed"), return FAILED);
  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "remove einsum node failed"), return FAILED);

  return SUCCESS;
}

Status EinsumPass::HandleStaticABCxABDE2DEC(ComputeGraph &graph, NodePtr &node) {
  // 009:reshape+reshape+matmul+reshape(swap input)
  // get op desc
  OpDescPtr op_desc = node->GetOpDesc();

  // get input
  GeTensorDesc x0_desc = op_desc->GetInputDesc(0);
  std::vector<int64_t> x0_dims = x0_desc.MutableShape().GetDims();
  GeTensorDesc x1_desc = op_desc->GetInputDesc(1);
  std::vector<int64_t> x1_dims = x1_desc.MutableShape().GetDims();

  // common vars
  auto x0_anchor_peer_anchor = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  auto x0_anchor_peer_node = x0_anchor_peer_anchor->GetOwnerNode();
  auto x1_anchor_peer_anchor = node->GetInDataAnchor(1)->GetPeerOutAnchor();
  auto x1_anchor_peer_node = x1_anchor_peer_anchor->GetOwnerNode();
  auto out_anchor_peer_anchors = node->GetOutDataAnchor(0)->GetPeerInDataAnchors();

  // create matmul op desc
  std::shared_ptr<ge::OpDesc> matmul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(matmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/MatMul", kMatMul),
                          return PARAM_INVALID);
  // create reshape op desc
  std::shared_ptr<ge::OpDesc> reshape_1_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_1_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Reshape1", kReshape),
                          return PARAM_INVALID);
  std::shared_ptr<ge::OpDesc> reshape_2_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_2_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Reshape2", kReshape),
                          return PARAM_INVALID);
  std::shared_ptr<ge::OpDesc> reshape_3_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_3_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Reshape3", kReshape),
                          return PARAM_INVALID);

  FUSION_PASS_CHECK((x0_dims.size() != 3) && (x1_dims.size() != 4),
                    OP_LOGI(kFusedOpType.c_str(), "input dims size must be three and four."), return NOT_CHANGED);

  // add input and output desc
  reshape_2_desc->AddInputDesc("x", x1_desc);
  std::vector<int64_t> tmp_dims({x1_dims[0] * x1_dims[1], x1_dims[2] * x1_dims[3]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_2_desc->AddOutputDesc("y", x1_desc);
  NodePtr reshape_2_node = CreateReshapeNode(tmp_dims, graph, reshape_2_desc);

  matmul_desc->AddInputDesc("x1", x1_desc);
  reshape_1_desc->AddInputDesc("x", x0_desc);
  tmp_dims.assign({x0_dims[0] * x0_dims[1], x0_dims[2]});
  x0_desc.SetShape(GeShape(tmp_dims));
  x0_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_1_desc->AddOutputDesc("y", x0_desc);
  NodePtr reshape_1_node = CreateReshapeNode(tmp_dims, graph, reshape_1_desc);

  matmul_desc->AddInputDesc("x2", x0_desc);
  tmp_dims.assign({x1_dims[2] * x1_dims[3], x0_dims[2]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  matmul_desc->AddOutputDesc("y", x1_desc);
  reshape_3_desc->AddInputDesc("x", x1_desc);
  tmp_dims.assign({x1_dims[2], x1_dims[3], x0_dims[2]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_3_desc->AddOutputDesc("y", x1_desc);
  NodePtr reshape_3_node = CreateReshapeNode(tmp_dims, graph, reshape_3_desc);

  // create matmul node
  NodePtr matmul_node = graph.AddNode(matmul_desc);
  // set op attr
  AttrUtils::SetBool(matmul_desc, "transpose_x1", true);
  AttrUtils::SetBool(matmul_desc, "transpose_x2", false);

  // unlink
  UnlinkAll(node);

  // add edge
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, reshape_1_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x0_anchor_peer_node->GetName().c_str(), reshape_1_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(reshape_1_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            reshape_1_node->GetName().c_str(), matmul_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, reshape_2_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x1_anchor_peer_node->GetName().c_str(), reshape_2_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(reshape_2_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            reshape_2_node->GetName().c_str(), matmul_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), reshape_3_node->GetInDataAnchor(0)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            matmul_node->GetName().c_str(), reshape_3_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(LinkNode(out_anchor_peer_anchors, reshape_3_node) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "link einsum node failed"), return FAILED);
  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "remove einsum node failed"), return FAILED);
  return SUCCESS;
}

Status EinsumPass::HandleDynamicABCxABDE2DEC(ComputeGraph &graph, NodePtr &node) {
  // 009:reshape+reshape+matmul+reshape(swap input)
  // get op desc
  OpDescPtr op_desc = node->GetOpDesc();

  // get input
  GeTensorDesc x0_desc = op_desc->GetInputDesc(0);
  std::vector<int64_t> x0_dims = x0_desc.MutableShape().GetDims();
  GeTensorDesc x1_desc = op_desc->GetInputDesc(1);
  std::vector<int64_t> x1_dims = x1_desc.MutableShape().GetDims();
  bool x0_is_unknown_shape = x0_desc.MutableShape().IsUnknownShape();
  bool x1_is_unknown_shape = x1_desc.MutableShape().IsUnknownShape();

  FUSION_PASS_CHECK((x0_dims.size() != 3) && (x1_dims.size() != 4),
                    OP_LOGI(kFusedOpType.c_str(), "input dims size must be three and four."), return NOT_CHANGED);

  // common vars
  std::vector<int64_t> tmp_dims;
  auto x0_anchor_peer_anchor = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  auto x0_anchor_peer_node = x0_anchor_peer_anchor->GetOwnerNode();
  auto x1_anchor_peer_anchor = node->GetInDataAnchor(1)->GetPeerOutAnchor();
  auto x1_anchor_peer_node = x1_anchor_peer_anchor->GetOwnerNode();
  auto out_anchor_peer_anchors = node->GetOutDataAnchor(0)->GetPeerInDataAnchors();

  // create GatherShapes op desc
  std::shared_ptr<ge::OpDesc> gatherShapes_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      gatherShapes_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/GatherShapes", kGatherShapes),
      return PARAM_INVALID);

  // create reshape op desc
  std::shared_ptr<ge::OpDesc> reshape_1_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_1_desc = CreateReshapeOpDesc(x0_is_unknown_shape, node, 1), return PARAM_INVALID);
  std::shared_ptr<ge::OpDesc> reshape_2_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_2_desc = CreateReshapeOpDesc(x1_is_unknown_shape, node, 2), return PARAM_INVALID);

  // init GatherShapes op desc
  gatherShapes_desc->AddInputDesc("x0", x0_desc);
  gatherShapes_desc->AddInputDesc("x1", x1_desc);
  const std::vector<std::vector<int64_t>> axes = {{1, 2}, {1, 3}, {0, 2}};
  FUSION_PASS_CHECK(AttrUtils::SetListListInt(gatherShapes_desc, "axes", axes) == false,
                    OP_LOGE(kFusedOpType.c_str(), "set gatherShapes axes failed."), return FAILED);

  gatherShapes_desc->AddOutputDesc("shape", x1_desc);
  NodePtr gatherShapes_node = graph.AddNode(gatherShapes_desc);
  FUSION_PASS_CHECK(gatherShapes_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "gatherShapes_node is null"),
                    return FAILED);
  if (gatherShapes_node->InferShapeAndType() != ge::GRAPH_SUCCESS) {
    OP_LOGE(kFusedOpType.c_str(), "gatherShapes infershape failed.");
    return FAILED;
  }

  reshape_1_desc->AddInputDesc("x", x0_desc);
  tmp_dims.assign({GetDimMulValue(x0_dims[0], x0_dims[1]), x0_dims[2]});
  x0_desc.SetShape(GeShape(tmp_dims));
  x0_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_1_desc->AddOutputDesc("y", x0_desc);
  NodePtr reshape_1_node = CreateReshapeNode(tmp_dims, graph, reshape_1_desc, 0, 1);
  FUSION_PASS_CHECK(reshape_1_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "reshape_1_node is null"), return FAILED);

  reshape_2_desc->AddInputDesc("x", x1_desc);
  tmp_dims.assign({GetDimMulValue(x1_dims[0], x1_dims[1]), GetDimMulValue(x1_dims[2], x1_dims[3])});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_2_desc->AddOutputDesc("y", x1_desc);
  NodePtr reshape_2_node = CreateReshapeNode(tmp_dims, graph, reshape_2_desc, 0, 1);
  FUSION_PASS_CHECK(reshape_2_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "reshape_2_node is null"), return FAILED);

  // create matmul op desc
  std::shared_ptr<ge::OpDesc> matmul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(matmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/MatMul", kMatMul),
                          return PARAM_INVALID);

  std::shared_ptr<ge::OpDesc> reshape_3_desc = nullptr;
  NodePtr reshape_3_node = nullptr;
  if (x1_is_unknown_shape) {
    FUSION_PASS_MAKE_SHARED(reshape_3_desc = CreateReshapeOpDesc(true, node, 3), return PARAM_INVALID);
    reshape_3_desc->AddInputDesc("x", *(reshape_2_desc->MutableOutputDesc(0)));
    reshape_3_desc->AddOutputDesc("y", x1_desc);
    reshape_3_node = CreateReshapeNode(tmp_dims, graph, reshape_3_desc, 1, 2);
    FUSION_PASS_CHECK(reshape_3_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "reshape_3_node is null"), return FAILED);
    matmul_desc->AddInputDesc("x1", *(reshape_3_desc->MutableOutputDesc(0)));
  } else {
    matmul_desc->AddInputDesc("x1", *(reshape_2_desc->MutableOutputDesc(0)));
  }

  matmul_desc->AddInputDesc("x2", *(reshape_1_desc->MutableOutputDesc(0)));
  matmul_desc->AddOutputDesc("y", x1_desc);
  AttrUtils::SetBool(matmul_desc, "transpose_x1", true);
  AttrUtils::SetBool(matmul_desc, "transpose_x2", false);
  NodePtr matmul_node = graph.AddNode(matmul_desc);
  FUSION_PASS_CHECK(matmul_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "matmul_node is null"), return FAILED);
  if (matmul_node->InferShapeAndType() != ge::GRAPH_SUCCESS) {
    OP_LOGE(kFusedOpType.c_str(), "matmul infershape failed.");
    return FAILED;
  }

  // create reshape op desc
  std::shared_ptr<ge::OpDesc> reshape_4_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_4_desc = CreateReshapeOpDesc(false, node, 4), return PARAM_INVALID);
  reshape_4_desc->AddInputDesc("x", *(matmul_desc->MutableOutputDesc(0)));
  reshape_4_desc->AddInputDesc("shape", *(gatherShapes_desc->MutableOutputDesc(0)));
  reshape_4_desc->AddOutputDesc("y", *(op_desc->MutableOutputDesc(0)));
  // geneate is_input_const
  static vector<bool> is_input_const = {false, false};
  reshape_4_desc->SetIsInputConst(is_input_const);
  NodePtr reshape_4_node = graph.AddNode(reshape_4_desc);
  FUSION_PASS_CHECK(reshape_4_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "reshape_4_node is null"), return FAILED);
  if (reshape_4_node->InferShapeAndType() != ge::GRAPH_SUCCESS) {
    OP_LOGE(kFusedOpType.c_str(), "reshape infershape failed.");
    return FAILED;
  }

  // unlink
  UnlinkAll(node);

  // add edge
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, gatherShapes_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x0_anchor_peer_node->GetName().c_str(), gatherShapes_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, gatherShapes_node->GetInDataAnchor(1)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x1_anchor_peer_node->GetName().c_str(), gatherShapes_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, reshape_1_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x0_anchor_peer_node->GetName().c_str(), reshape_1_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, reshape_2_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x1_anchor_peer_node->GetName().c_str(), reshape_2_node->GetName().c_str()),
                    return FAILED);
  if (x1_is_unknown_shape) {
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(reshape_2_node->GetOutDataAnchor(0), reshape_3_node->GetInDataAnchor(0)),
        CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              reshape_2_node->GetName().c_str(), reshape_3_node->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(reshape_3_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0)),
        CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              reshape_3_node->GetName().c_str(), matmul_node->GetName().c_str()),
        return FAILED);
  } else {
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(reshape_2_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0)),
        CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              reshape_2_node->GetName().c_str(), matmul_node->GetName().c_str()),
        return FAILED);
  }
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(reshape_1_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            reshape_1_node->GetName().c_str(), matmul_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), reshape_4_node->GetInDataAnchor(0)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            matmul_node->GetName().c_str(), reshape_4_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(gatherShapes_node->GetOutDataAnchor(0), reshape_4_node->GetInDataAnchor(1)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            gatherShapes_node->GetName().c_str(), reshape_4_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(LinkNode(out_anchor_peer_anchors, reshape_4_node) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "link einsum node failed"), return FAILED);
  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "remove einsum node failed"), return FAILED);

  return SUCCESS;
}

Status EinsumPass::HandleABCDxAECD2ACBE(ComputeGraph &graph, NodePtr &node) {
  // 010:transpose+batchmatmul+transpose
  // get op desc
  OpDescPtr op_desc = node->GetOpDesc();

  // get input
  GeTensorDesc x0_desc = op_desc->GetInputDesc(0);
  std::vector<int64_t> x0_dims = x0_desc.MutableShape().GetDims();
  GeTensorDesc x1_desc = op_desc->GetInputDesc(1);
  std::vector<int64_t> x1_dims = x1_desc.MutableShape().GetDims();
  bool x0_is_unknown_shape = x0_desc.MutableShape().IsUnknownShape();
  bool x1_is_unknown_shape = x1_desc.MutableShape().IsUnknownShape();

  // create batchmatmulv2 op desc
  std::shared_ptr<ge::OpDesc> batchmatmul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      batchmatmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/BatchMatMul", kBatchMatMul),
      return PARAM_INVALID);
  // create transpose op desc
  std::shared_ptr<ge::OpDesc> transpose_1_desc = CreateTransposeOpDesc(x0_is_unknown_shape, node, "/Transpose1");
  std::shared_ptr<ge::OpDesc> transpose_2_desc = CreateTransposeOpDesc(x1_is_unknown_shape, node, "/Transpose2");

  FUSION_PASS_CHECK((x0_dims.size() != 4) && (x1_dims.size() != 4),
                    OP_LOGI(kFusedOpType.c_str(), "input dims size must be four and four."), return NOT_CHANGED);

  // set op attr
  AttrUtils::SetBool(batchmatmul_desc, "adj_x1", false);
  AttrUtils::SetBool(batchmatmul_desc, "adj_x2", true);
  std::vector<int32_t> perm({0, 2, 1, 3});

  // add input and output desc
  transpose_1_desc->AddInputDesc("x", x0_desc);
  std::vector<int64_t> tmp_dims({x0_dims[0], x0_dims[2], x0_dims[1], x0_dims[3]});
  x0_desc.SetShape(GeShape(tmp_dims));
  x0_desc.SetOriginShape(GeShape(tmp_dims));
  transpose_1_desc->AddOutputDesc("y", x0_desc);
  NodePtr transpose_1_node = graph.AddNode(transpose_1_desc);
  SetTransposePerm(x0_is_unknown_shape, perm, graph, transpose_1_desc, transpose_1_node);
  FUSION_PASS_CHECK(x0_is_unknown_shape && transpose_1_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "transpose infershape failed."), return FAILED);

  transpose_2_desc->AddInputDesc("x", x1_desc);
  tmp_dims.assign({x1_dims[0], x1_dims[2], x1_dims[1], x1_dims[3]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  transpose_2_desc->AddOutputDesc("y", x1_desc);
  NodePtr transpose_2_node = graph.AddNode(transpose_2_desc);
  SetTransposePerm(x1_is_unknown_shape, perm, graph, transpose_2_desc, transpose_2_node);
  FUSION_PASS_CHECK(x1_is_unknown_shape && transpose_2_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "transpose infershape failed."), return FAILED);

  batchmatmul_desc->AddInputDesc("x1", *(transpose_1_desc->MutableOutputDesc(0)));
  batchmatmul_desc->AddInputDesc("x2", *(transpose_2_desc->MutableOutputDesc(0)));
  batchmatmul_desc->AddOutputDesc("y", *(op_desc->MutableOutputDesc(0)));
  // create transpose and batchmatmul node
  NodePtr batchmatmul_node = graph.AddNode(batchmatmul_desc);
  FUSION_PASS_CHECK(RelinkMatmulNode(node, transpose_1_node, transpose_2_node, batchmatmul_node, false) != SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "failed to relink transpose/batchmatmul"), return FAILED);

  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "remove einsum node failed"), return FAILED);
  return SUCCESS;
}

Status EinsumPass::HandleABCDxACBE2AECD(ComputeGraph &graph, NodePtr &node) {
  // 011:transpose+batchmatmul+transpose(swap input)
  // get op desc
  OpDescPtr op_desc = node->GetOpDesc();

  // get input
  GeTensorDesc x0_desc = op_desc->GetInputDesc(0);
  std::vector<int64_t> x0_dims = x0_desc.MutableShape().GetDims();
  GeTensorDesc x1_desc = op_desc->GetInputDesc(1);
  std::vector<int64_t> x1_dims = x1_desc.MutableShape().GetDims();
  bool x0_is_unknown_shape = x0_desc.MutableShape().IsUnknownShape();
  bool x1_is_unknown_shape = x1_desc.MutableShape().IsUnknownShape();

  // common vars
  auto x0_anchor_peer_anchor = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  auto x0_anchor_peer_node = x0_anchor_peer_anchor->GetOwnerNode();
  auto x1_anchor_peer_anchor = node->GetInDataAnchor(1)->GetPeerOutAnchor();
  auto x1_anchor_peer_node = x1_anchor_peer_anchor->GetOwnerNode();
  auto out_anchor_peer_anchors = node->GetOutDataAnchor(0)->GetPeerInDataAnchors();

  // create batchmatmulv2 op desc
  std::shared_ptr<ge::OpDesc> batchmatmul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      batchmatmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/BatchMatMul", kBatchMatMul),
      return PARAM_INVALID);
  // create transpose op desc
  std::shared_ptr<ge::OpDesc> transpose_1_desc = CreateTransposeOpDesc(x0_is_unknown_shape, node, "/Transpose1");

  FUSION_PASS_CHECK((x0_dims.size() != 4) && (x1_dims.size() != 4),
                    OP_LOGI(kFusedOpType.c_str(), "input dims size must be four and four."), return NOT_CHANGED);

  // set op attr
  AttrUtils::SetBool(batchmatmul_desc, "adj_x1", true);
  AttrUtils::SetBool(batchmatmul_desc, "adj_x2", false);
  std::vector<int32_t> perm({0, 2, 1, 3});

  // add input and output desc
  batchmatmul_desc->AddInputDesc("x1", x1_desc);
  transpose_1_desc->AddInputDesc("x", x0_desc);
  std::vector<int64_t> tmp_dims({x0_dims[0], x0_dims[2], x0_dims[1], x0_dims[3]});
  x0_desc.SetShape(GeShape(tmp_dims));
  x0_desc.SetOriginShape(GeShape(tmp_dims));
  transpose_1_desc->AddOutputDesc("y", x0_desc);
  NodePtr transpose_1_node = graph.AddNode(transpose_1_desc);
  SetTransposePerm(x0_is_unknown_shape, perm, graph, transpose_1_desc, transpose_1_node);
  FUSION_PASS_CHECK(x0_is_unknown_shape && transpose_1_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "transpose infershape failed."), return FAILED);

  batchmatmul_desc->AddInputDesc("x2", *(transpose_1_desc->MutableOutputDesc(0)));
  tmp_dims.assign({x1_dims[0], x1_dims[1], x1_dims[3], x0_dims[3]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  batchmatmul_desc->AddOutputDesc("y", x1_desc);
  NodePtr batchmatmul_node = graph.AddNode(batchmatmul_desc);
  FUSION_PASS_CHECK((x1_is_unknown_shape || transpose_1_desc->MutableOutputDesc(0)->MutableShape().IsUnknownShape()) &&
                        batchmatmul_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "batchmatmul infershape failed."), return FAILED);

  bool is_unknown_shape = batchmatmul_desc->MutableOutputDesc(0)->MutableShape().IsUnknownShape();
  std::shared_ptr<ge::OpDesc> transpose_2_desc = CreateTransposeOpDesc(is_unknown_shape, node, "/Transpose2");
  transpose_2_desc->AddInputDesc("x", *(batchmatmul_desc->MutableOutputDesc(0)));
  transpose_2_desc->AddOutputDesc("y", *(op_desc->MutableOutputDesc(0)));
  NodePtr transpose_2_node = graph.AddNode(transpose_2_desc);
  SetTransposePerm(is_unknown_shape, perm, graph, transpose_2_desc, transpose_2_node);

  // unlink
  UnlinkAll(node);

  // add edge
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, transpose_1_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x0_anchor_peer_node->GetName().c_str(), transpose_1_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(transpose_1_node->GetOutDataAnchor(0), batchmatmul_node->GetInDataAnchor(1)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            transpose_1_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, batchmatmul_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x1_anchor_peer_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(batchmatmul_node->GetOutDataAnchor(0), transpose_2_node->GetInDataAnchor(0)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            batchmatmul_node->GetName().c_str(), transpose_2_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(LinkNode(out_anchor_peer_anchors, transpose_2_node) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "link einsum node failed"), return FAILED);
  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "remove einsum node failed"), return FAILED);
  return SUCCESS;
}

Status EinsumPass::HandleABCDxECD2ABE(ComputeGraph &graph, NodePtr &node) {
  // 012:reshape+reshape+matmul+reshape-->reshape+batchmatmul
  // get op desc
  OpDescPtr op_desc = node->GetOpDesc();

  // get input
  GeTensorDesc x0_desc = op_desc->GetInputDesc(0);
  std::vector<int64_t> x0_dims = x0_desc.MutableShape().GetDims();
  GeTensorDesc x1_desc = op_desc->GetInputDesc(1);
  std::vector<int64_t> x1_dims = x1_desc.MutableShape().GetDims();
  bool x0_is_unknown_shape = x0_desc.MutableShape().IsUnknownShape();
  bool x1_is_unknown_shape = x1_desc.MutableShape().IsUnknownShape();

  // create batchmatmulv2 op desc
  std::shared_ptr<ge::OpDesc> batchmatmul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      batchmatmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/BatchMatMul", kBatchMatMul),
      return PARAM_INVALID);
  // create reshape op desc
  std::shared_ptr<ge::OpDesc> reshape_1_desc = CreateReshapeOpDesc(x0_is_unknown_shape, node, 1);
  std::shared_ptr<ge::OpDesc> reshape_2_desc = CreateReshapeOpDesc(x1_is_unknown_shape, node, 2);

  FUSION_PASS_CHECK((x0_dims.size() != 4) && (x1_dims.size() != 3),
                    OP_LOGI(kFusedOpType.c_str(), "input dims size must be four and three."), return NOT_CHANGED);

  // add input and output desc
  reshape_1_desc->AddInputDesc("x", x0_desc);
  std::vector<int64_t> tmp_dims({x0_dims[0], x0_dims[1], GetDimMulValue(x0_dims[2], x0_dims[3])});
  x0_desc.SetShape(GeShape(tmp_dims));
  x0_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_1_desc->AddOutputDesc("y", x0_desc);
  NodePtr reshape_1_node = CreateReshapeNode(tmp_dims, graph, reshape_1_desc, 2, 3);

  reshape_2_desc->AddInputDesc("x", x1_desc);
  tmp_dims.assign({x1_dims[0], GetDimMulValue(x1_dims[1], x1_dims[2])});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_2_desc->AddOutputDesc("y", x1_desc);
  NodePtr reshape_2_node = CreateReshapeNode(tmp_dims, graph, reshape_2_desc, 1, 2);

  batchmatmul_desc->AddInputDesc("x1", *(reshape_1_desc->MutableOutputDesc(0)));
  batchmatmul_desc->AddInputDesc("x2", *(reshape_2_desc->MutableOutputDesc(0)));
  batchmatmul_desc->AddOutputDesc("y", *(op_desc->MutableOutputDesc(0)));
  // create batchmatmul and reshape node
  NodePtr batchmatmul_node = graph.AddNode(batchmatmul_desc);
  // set op attr
  AttrUtils::SetBool(batchmatmul_desc, "adj_x1", false);
  AttrUtils::SetBool(batchmatmul_desc, "adj_x2", true);
  FUSION_PASS_CHECK(RelinkMatmulNode(node, reshape_1_node, reshape_2_node, batchmatmul_node, false) != SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "failed to relink reshape/batchmatmul"), return FAILED);

  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "remove einsum node failed"), return FAILED);
  return SUCCESS;
}

Status EinsumPass::HandleStaticABCDxABE2ECD(ComputeGraph &graph, NodePtr &node) {
  // 013:reshape+reshape+matmul+reshape(swap input)
  // get op desc
  OpDescPtr op_desc = node->GetOpDesc();

  // get input
  GeTensorDesc x0_desc = op_desc->GetInputDesc(0);
  std::vector<int64_t> x0_dims = x0_desc.MutableShape().GetDims();
  GeTensorDesc x1_desc = op_desc->GetInputDesc(1);
  std::vector<int64_t> x1_dims = x1_desc.MutableShape().GetDims();

  // common vars
  auto x0_anchor_peer_anchor = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  auto x0_anchor_peer_node = x0_anchor_peer_anchor->GetOwnerNode();
  auto x1_anchor_peer_anchor = node->GetInDataAnchor(1)->GetPeerOutAnchor();
  auto x1_anchor_peer_node = x1_anchor_peer_anchor->GetOwnerNode();
  auto out_anchor_peer_anchors = node->GetOutDataAnchor(0)->GetPeerInDataAnchors();

  // create matmul op desc
  std::shared_ptr<ge::OpDesc> matmul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(matmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/MatMul", kMatMul),
                          return PARAM_INVALID);
  // create reshape op desc
  std::shared_ptr<ge::OpDesc> reshape_1_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_1_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Reshape1", kReshape),
                          return PARAM_INVALID);
  std::shared_ptr<ge::OpDesc> reshape_2_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_2_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Reshape2", kReshape),
                          return PARAM_INVALID);
  std::shared_ptr<ge::OpDesc> reshape_3_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_3_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/Reshape3", kReshape),
                          return PARAM_INVALID);

  FUSION_PASS_CHECK((x0_dims.size() != 4) && (x1_dims.size() != 3),
                    OP_LOGI(kFusedOpType.c_str(), "input dims size must be four and three."), return NOT_CHANGED);

  // add input and output desc
  reshape_2_desc->AddInputDesc("x", x1_desc);
  std::vector<int64_t> tmp_dims({x1_dims[0] * x1_dims[1], x1_dims[2]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_2_desc->AddOutputDesc("y", x1_desc);
  NodePtr reshape_2_node = CreateReshapeNode(tmp_dims, graph, reshape_2_desc);

  matmul_desc->AddInputDesc("x1", x1_desc);
  reshape_1_desc->AddInputDesc("x", x0_desc);
  tmp_dims.assign({x0_dims[0] * x0_dims[1], x0_dims[2] * x0_dims[3]});
  x0_desc.SetShape(GeShape(tmp_dims));
  x0_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_1_desc->AddOutputDesc("y", x0_desc);
  NodePtr reshape_1_node = CreateReshapeNode(tmp_dims, graph, reshape_1_desc);

  matmul_desc->AddInputDesc("x2", x0_desc);
  tmp_dims.assign({x1_dims[2], x0_dims[2] * x0_dims[3]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  matmul_desc->AddOutputDesc("y", x1_desc);
  reshape_3_desc->AddInputDesc("x", x1_desc);
  tmp_dims.assign({x1_dims[2], x0_dims[2], x0_dims[3]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_3_desc->AddOutputDesc("y", x1_desc);
  NodePtr reshape_3_node = CreateReshapeNode(tmp_dims, graph, reshape_3_desc);

  // create matmul node
  NodePtr matmul_node = graph.AddNode(matmul_desc);
  // set op attr
  AttrUtils::SetBool(matmul_desc, "transpose_x1", true);
  AttrUtils::SetBool(matmul_desc, "transpose_x2", false);

  // unlink
  UnlinkAll(node);

  // add edge
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, reshape_1_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x0_anchor_peer_node->GetName().c_str(), reshape_1_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(reshape_1_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            reshape_1_node->GetName().c_str(), matmul_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, reshape_2_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x1_anchor_peer_node->GetName().c_str(), reshape_2_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(reshape_2_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            reshape_2_node->GetName().c_str(), matmul_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), reshape_3_node->GetInDataAnchor(0)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            matmul_node->GetName().c_str(), reshape_3_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(LinkNode(out_anchor_peer_anchors, reshape_3_node) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "link einsum node failed"), return FAILED);
  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "remove einsum node failed"), return FAILED);
  return SUCCESS;
}

Status EinsumPass::HandleDynamicABCDxABE2ECD(ComputeGraph &graph, NodePtr &node) {
  // 013:reshape+reshape+matmul+reshape(swap input)
  // get op desc
  OpDescPtr op_desc = node->GetOpDesc();

  // get input
  GeTensorDesc x0_desc = op_desc->GetInputDesc(0);
  std::vector<int64_t> x0_dims = x0_desc.MutableShape().GetDims();
  GeTensorDesc x1_desc = op_desc->GetInputDesc(1);
  std::vector<int64_t> x1_dims = x1_desc.MutableShape().GetDims();
  bool x0_is_unknown_shape = x0_desc.MutableShape().IsUnknownShape();
  bool x1_is_unknown_shape = x1_desc.MutableShape().IsUnknownShape();

  FUSION_PASS_CHECK((x0_dims.size() != 4) && (x1_dims.size() != 3),
                    OP_LOGI(kFusedOpType.c_str(), "input dims size must be four and three."), return NOT_CHANGED);

  // common vars
  std::vector<int64_t> tmp_dims;
  auto x0_anchor_peer_anchor = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  auto x0_anchor_peer_node = x0_anchor_peer_anchor->GetOwnerNode();
  auto x1_anchor_peer_anchor = node->GetInDataAnchor(1)->GetPeerOutAnchor();
  auto x1_anchor_peer_node = x1_anchor_peer_anchor->GetOwnerNode();
  auto out_anchor_peer_anchors = node->GetOutDataAnchor(0)->GetPeerInDataAnchors();

  // create GatherShapes op desc
  std::shared_ptr<ge::OpDesc> gatherShapes_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      gatherShapes_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/GatherShapes", kGatherShapes),
      return PARAM_INVALID);

  // create reshape op desc
  std::shared_ptr<ge::OpDesc> reshape_1_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_1_desc = CreateReshapeOpDesc(x0_is_unknown_shape, node, 1), return PARAM_INVALID);
  std::shared_ptr<ge::OpDesc> reshape_2_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_2_desc = CreateReshapeOpDesc(x1_is_unknown_shape, node, 2), return PARAM_INVALID);

  // init input and output desc
  gatherShapes_desc->AddInputDesc("x0", x0_desc);
  gatherShapes_desc->AddInputDesc("x1", x1_desc);
  const std::vector<std::vector<int64_t>> axes = {{1, 2}, {0, 2}, {0, 3}};
  FUSION_PASS_CHECK(AttrUtils::SetListListInt(gatherShapes_desc, "axes", axes) == false,
                    OP_LOGE(kFusedOpType.c_str(), "set gatherShapes axes failed."), return FAILED);

  reshape_1_desc->AddInputDesc("x", x0_desc);
  tmp_dims.assign({GetDimMulValue(x0_dims[0], x0_dims[1]), GetDimMulValue(x0_dims[2], x0_dims[3])});
  x0_desc.SetShape(GeShape(tmp_dims));
  x0_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_1_desc->AddOutputDesc("y", x0_desc);
  NodePtr reshape_1_node = CreateReshapeNode(tmp_dims, graph, reshape_1_desc, 0, 1);
  FUSION_PASS_CHECK(reshape_1_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "reshape_1_node is null"), return FAILED);

  reshape_2_desc->AddInputDesc("x", x1_desc);
  tmp_dims.assign({GetDimMulValue(x1_dims[0], x1_dims[1]), x1_dims[2]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  reshape_2_desc->AddOutputDesc("y", x1_desc);
  NodePtr reshape_2_node = CreateReshapeNode(tmp_dims, graph, reshape_2_desc, 0, 1);
  FUSION_PASS_CHECK(reshape_2_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "reshape_2_node is null"), return FAILED);

  tmp_dims.assign({x1_dims[2], x0_dims[2], x0_dims[3]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  gatherShapes_desc->AddOutputDesc("shape", x1_desc);
  NodePtr gatherShapes_node = graph.AddNode(gatherShapes_desc);
  FUSION_PASS_CHECK(gatherShapes_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "gatherShapes_node is null"),
                    return FAILED);
  if (gatherShapes_node->InferShapeAndType() != ge::GRAPH_SUCCESS) {
    OP_LOGE(kFusedOpType.c_str(), "gatherShapes infershape failed.");
    return FAILED;
  }

  // create matmul op desc
  std::shared_ptr<ge::OpDesc> matmul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(matmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/MatMul", kMatMul),
                          return PARAM_INVALID);
  matmul_desc->AddInputDesc("x1", *(reshape_2_desc->MutableOutputDesc(0)));

  std::shared_ptr<ge::OpDesc> reshape_3_desc = nullptr;
  NodePtr reshape_3_node = nullptr;
  if (x0_is_unknown_shape) {
    FUSION_PASS_MAKE_SHARED(reshape_3_desc = CreateReshapeOpDesc(true, node, 3), return PARAM_INVALID);
    reshape_3_desc->AddInputDesc("x", *(reshape_1_desc->MutableOutputDesc(0)));
    tmp_dims.assign({GetDimMulValue(x0_dims[0], x0_dims[1]), GetDimMulValue(x0_dims[2], x0_dims[3])});
    x1_desc.SetShape(GeShape(tmp_dims));
    x1_desc.SetOriginShape(GeShape(tmp_dims));
    reshape_3_desc->AddOutputDesc("y", x1_desc);
    reshape_3_node = CreateReshapeNode(tmp_dims, graph, reshape_3_desc, 1, 2);
    FUSION_PASS_CHECK(reshape_3_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "reshape_3_node is null"),
                      return FAILED);
    matmul_desc->AddInputDesc("x2", *(reshape_3_desc->MutableOutputDesc(0)));
  } else {
    matmul_desc->AddInputDesc("x2", *(reshape_1_desc->MutableOutputDesc(0)));
  }
  tmp_dims.assign({x1_dims[2], GetDimMulValue(x0_dims[2], x0_dims[3])});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  matmul_desc->AddOutputDesc("y", x1_desc);
  AttrUtils::SetBool(matmul_desc, "transpose_x1", true);
  AttrUtils::SetBool(matmul_desc, "transpose_x2", false);
  NodePtr matmul_node = graph.AddNode(matmul_desc);
  FUSION_PASS_CHECK(matmul_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "matmul_node is null"), return FAILED);
  if (matmul_node->InferShapeAndType() != ge::GRAPH_SUCCESS) {
    OP_LOGE(kFusedOpType.c_str(), "matmul infershape failed.");
    return FAILED;
  }

  // create reshape op desc
  std::shared_ptr<ge::OpDesc> reshape_4_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(reshape_4_desc = CreateReshapeOpDesc(false, node, 4), return PARAM_INVALID);
  reshape_4_desc->AddInputDesc("x", *(matmul_desc->MutableOutputDesc(0)));
  reshape_4_desc->AddInputDesc("shape", *(gatherShapes_desc->MutableOutputDesc(0)));
  reshape_4_desc->AddOutputDesc("y", *(op_desc->MutableOutputDesc(0)));
  // geneate is_input_const
  static vector<bool> is_input_const = {false, false};
  reshape_4_desc->SetIsInputConst(is_input_const);
  NodePtr reshape_4_node = graph.AddNode(reshape_4_desc);
  FUSION_PASS_CHECK(reshape_4_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "reshape_4_node is null"), return FAILED);
  if (reshape_4_node->InferShapeAndType() != ge::GRAPH_SUCCESS) {
    OP_LOGE(kFusedOpType.c_str(), "reshape infershape failed.");
    return FAILED;
  }

  // unlink
  UnlinkAll(node);

  // add edge
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, gatherShapes_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x0_anchor_peer_node->GetName().c_str(), gatherShapes_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, gatherShapes_node->GetInDataAnchor(1)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x1_anchor_peer_node->GetName().c_str(), gatherShapes_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, reshape_1_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x0_anchor_peer_node->GetName().c_str(), reshape_1_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, reshape_2_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x1_anchor_peer_node->GetName().c_str(), reshape_2_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(reshape_2_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(0)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            reshape_2_node->GetName().c_str(), matmul_node->GetName().c_str()),
      return FAILED);
  if (x0_is_unknown_shape) {
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(reshape_1_node->GetOutDataAnchor(0), reshape_3_node->GetInDataAnchor(0)),
        CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              reshape_1_node->GetName().c_str(), reshape_3_node->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(reshape_3_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1)),
        CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              reshape_3_node->GetName().c_str(), matmul_node->GetName().c_str()),
        return FAILED);
  } else {
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(reshape_1_node->GetOutDataAnchor(0), matmul_node->GetInDataAnchor(1)),
        CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              reshape_1_node->GetName().c_str(), matmul_node->GetName().c_str()),
        return FAILED);
  }
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), reshape_4_node->GetInDataAnchor(0)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            matmul_node->GetName().c_str(), reshape_4_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(gatherShapes_node->GetOutDataAnchor(0), reshape_4_node->GetInDataAnchor(1)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            gatherShapes_node->GetName().c_str(), reshape_4_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(LinkNode(out_anchor_peer_anchors, reshape_4_node) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "link einsum node failed"), return FAILED);
  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "remove einsum node failed"), return FAILED);

  return SUCCESS;
}

Status EinsumPass::HandleABCDxACBE2ADBE(ComputeGraph &graph, NodePtr &node) {
  // 014: transpose+batchmatmul+transpose
  // get op desc
  OpDescPtr op_desc = node->GetOpDesc();

  // get input
  GeTensorDesc x0_desc = op_desc->GetInputDesc(0);
  std::vector<int64_t> x0_dims = x0_desc.MutableShape().GetDims();
  GeTensorDesc x1_desc = op_desc->GetInputDesc(1);
  std::vector<int64_t> x1_dims = x1_desc.MutableShape().GetDims();
  bool x0_is_unknown_shape = x0_desc.MutableShape().IsUnknownShape();
  bool x1_is_unknown_shape = x1_desc.MutableShape().IsUnknownShape();

  // common vars
  auto x0_anchor_peer_anchor = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  auto x0_anchor_peer_node = x0_anchor_peer_anchor->GetOwnerNode();
  auto x1_anchor_peer_anchor = node->GetInDataAnchor(1)->GetPeerOutAnchor();
  auto x1_anchor_peer_node = x1_anchor_peer_anchor->GetOwnerNode();
  auto out_anchor_peer_anchors = node->GetOutDataAnchor(0)->GetPeerInDataAnchors();

  // create batchmatmulv2 op desc
  std::shared_ptr<ge::OpDesc> batchmatmul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      batchmatmul_desc = std::make_shared<ge::OpDesc>(node->GetName() + "/BatchMatMul", kBatchMatMul),
      return PARAM_INVALID);
  // create transpose op desc
  std::shared_ptr<ge::OpDesc> transpose_2_desc = CreateTransposeOpDesc(x1_is_unknown_shape, node, "/Transpose2");

  FUSION_PASS_CHECK((x0_dims.size() != 4) && (x1_dims.size() != 4),
                    OP_LOGI(kFusedOpType.c_str(), "input dims size must be four and four."), return NOT_CHANGED);

  // set op attr
  AttrUtils::SetBool(batchmatmul_desc, "adj_x1", true);
  AttrUtils::SetBool(batchmatmul_desc, "adj_x2", false);
  std::vector<int32_t> perm({0, 2, 1, 3});

  // add input and output desc
  batchmatmul_desc->AddInputDesc("x1", x0_desc);
  transpose_2_desc->AddInputDesc("x", x1_desc);
  std::vector<int64_t> tmp_dims({x1_dims[0], x1_dims[2], x1_dims[1], x1_dims[3]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  transpose_2_desc->AddOutputDesc("y", x1_desc);
  NodePtr transpose_2_node = graph.AddNode(transpose_2_desc);
  SetTransposePerm(x1_is_unknown_shape, perm, graph, transpose_2_desc, transpose_2_node);
  FUSION_PASS_CHECK(x1_is_unknown_shape && transpose_2_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "transpose infershape failed."), return FAILED);

  batchmatmul_desc->AddInputDesc("x2", *(transpose_2_desc->MutableOutputDesc(0)));
  tmp_dims.assign({x0_dims[0], x0_dims[1], x0_dims[3], x1_dims[3]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  batchmatmul_desc->AddOutputDesc("y", x1_desc);
  NodePtr batchmatmul_node = graph.AddNode(batchmatmul_desc);
  FUSION_PASS_CHECK((x0_is_unknown_shape || transpose_2_desc->MutableOutputDesc(0)->MutableShape().IsUnknownShape()) &&
                        batchmatmul_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "batchmatmul infershape failed."), return FAILED);

  bool is_unknown_shape = batchmatmul_desc->MutableOutputDesc(0)->MutableShape().IsUnknownShape();
  std::shared_ptr<ge::OpDesc> transpose_3_desc = CreateTransposeOpDesc(is_unknown_shape, node, "/Transpose3");
  transpose_3_desc->AddInputDesc("x", *(batchmatmul_desc->MutableOutputDesc(0)));
  transpose_3_desc->AddOutputDesc("y", *(op_desc->MutableOutputDesc(0)));
  NodePtr transpose_3_node = graph.AddNode(transpose_3_desc);
  SetTransposePerm(is_unknown_shape, perm, graph, transpose_3_desc, transpose_3_node);

  // unlink
  UnlinkAll(node);

  // add edge
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, batchmatmul_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x0_anchor_peer_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, transpose_2_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x1_anchor_peer_node->GetName().c_str(), transpose_2_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(transpose_2_node->GetOutDataAnchor(0), batchmatmul_node->GetInDataAnchor(1)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            transpose_2_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(batchmatmul_node->GetOutDataAnchor(0), transpose_3_node->GetInDataAnchor(0)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            batchmatmul_node->GetName().c_str(), transpose_3_node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(LinkNode(out_anchor_peer_anchors, transpose_3_node) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "link einsum node failed"), return FAILED);
  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "remove einsum node failed"), return FAILED);
  return SUCCESS;
}

Status EinsumPass::CheckProduct(const std::vector<int64_t> &shape) {
  int64_t product = 1;
  for (auto var : shape) {
    if (var > 0) {
      if (product > (INT64_MAX / var)) {
        return PARAM_INVALID;
      } else {
        product *= var;
      }
    }
  }

  return SUCCESS;
}

Status EinsumPass::CheckInputArgs(const Mapping &mapping) {
  // get node
  NodePtr node = GetNodeFromMapping(kPatternFusedNode, mapping);
  FUSION_PASS_CHECK(node == nullptr, CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "einsum node is null, fusion failed."),
                    return PARAM_INVALID);
  // get op desc
  OpDescPtr op_desc = node->GetOpDesc();
  FUSION_PASS_CHECK(op_desc == nullptr,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "einsum desc is null, fusion failed."),
                    return PARAM_INVALID);

  // check input link relation
  FUSION_PASS_CHECK(node->GetInDataNodes().size() != 2,
                    OP_LOGI(kFusedOpType.c_str(), "Input node of einsum node size is [%lu], which not equal to 2.",
                            node->GetInDataNodes().size()),
                    return NOT_CHANGED);
  // get input
  auto x0_desc = op_desc->MutableInputDesc(0);
  auto x1_desc = op_desc->MutableInputDesc(1);

  FUSION_PASS_CHECK(x0_desc == nullptr || x1_desc == nullptr,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "einsum input desc is null, fusion failed."),
                    return PARAM_INVALID);

  // check whether the tensor size overflows
  if ((CheckProduct(x0_desc->MutableShape().GetDims()) != SUCCESS) ||
      (CheckProduct(x1_desc->MutableShape().GetDims()) != SUCCESS)) {
    return PARAM_INVALID;
  }

  auto x0_anchor = node->GetInDataAnchor(0);
  FUSION_PASS_CHECK(x0_anchor == nullptr,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "einsum x0_anchor is null, fusion failed."),
                    return PARAM_INVALID);
  auto x0_anchor_peer_anchor = node->GetInDataAnchor(0)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(x0_anchor_peer_anchor == nullptr,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "einsum x0_anchor_peer_anchor is null, fusion failed."),
                    return PARAM_INVALID);
  auto x0_anchor_peer_node = x0_anchor_peer_anchor->GetOwnerNode();
  FUSION_PASS_CHECK(x0_anchor_peer_node == nullptr,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "einsum x0_anchor_peer_node is null, fusion failed."),
                    return PARAM_INVALID);
  auto x1_anchor = node->GetInDataAnchor(1);
  FUSION_PASS_CHECK(x1_anchor == nullptr,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "einsum x1_anchor is null, fusion failed."),
                    return PARAM_INVALID);
  auto x1_anchor_peer_anchor = node->GetInDataAnchor(1)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(x1_anchor_peer_anchor == nullptr,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "einsum x1_anchor_peer_anchor is null, fusion failed."),
                    return PARAM_INVALID);
  auto x1_anchor_peer_node = x1_anchor_peer_anchor->GetOwnerNode();
  FUSION_PASS_CHECK(x1_anchor_peer_node == nullptr,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "einsum x1_anchor_peer_node is null, fusion failed."),
                    return PARAM_INVALID);
  auto out_anchor = node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(out_anchor == nullptr,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "einsum out_anchor is null, fusion failed."),
                    return PARAM_INVALID);
  return SUCCESS;
}

// vector<NodePtr> &fusion_nodes: Store fusion nodes,
//       including newly added nodes and fused but not deleted nodes
Status EinsumPass::Fusion(ComputeGraph &graph, Mapping &mapping, vector<NodePtr> & /* fusion_nodes */) {
  OP_LOGI(kFusedOpType.c_str(), "EinsumPass fusion in!");
  Status ret = CheckInputArgs(mapping);
  if (ret != SUCCESS) {
    return ret;
  }

  // get attr equation
  std::string equation;
  NodePtr node = GetNodeFromMapping(kPatternFusedNode, mapping);
  OpDescPtr op_desc = node->GetOpDesc();
  FUSION_PASS_CHECK(AttrUtils::GetStr(op_desc, "equation", equation) == false,
                    OP_LOGI(kFusedOpType.c_str(), "Get attr equation failed."), return NOT_CHANGED);

  EquationNormalization(equation);
  auto x0_desc = op_desc->MutableInputDesc(0);
  auto x1_desc = op_desc->MutableInputDesc(1);
  std::unordered_map<std::string, ProcFunc> *procs = &staticShapeProcs_;
  if (x0_desc->MutableShape().IsUnknownShape() || x1_desc->MutableShape().IsUnknownShape()) {
    procs = &dynamicShapeProcs_;
  }

  auto it = procs->find(equation);
  if (it != procs->end()) {
    ret = (this->*(it->second))(graph, node);
    if (ret == SUCCESS) {
      OP_LOGI(kFusedOpType.c_str(), "EinsumPass fusion success!");
    }

    return ret;
  } else {
    OP_LOGI(kFusedOpType.c_str(), "equation[%s] is not match.", equation.c_str());
    return NOT_CHANGED;
  }
}

REGISTER_PASS("EinsumPass", BUILT_IN_GRAPH_PASS, EinsumPass);
}  // namespace fe
