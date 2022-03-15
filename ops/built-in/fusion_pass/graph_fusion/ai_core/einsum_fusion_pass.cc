/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
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
#include <numeric>
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
static const string kBatchMatMulV2 = "BatchMatMulV2";
static const string kFlatten = "FlattenV2";
static const string kGatherShapes = "GatherShapes";
static const string kReduceSumD = "ReduceSumD";
static const string kFusedOpType = "Einsum";
static const string kBroadCastLabel = "...";
static const size_t kBroadCastLabelLen = 3;

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

static std::string Vector2Str(const std::vector<int32_t> &dims) {
  if (dims.empty()) {
    return "";
  }

  std::stringstream ss;
  ss << dims[0];
  for (size_t i = 1; i < dims.size(); ++i) {
    ss << ", " << dims[i];
  }

  return ss.str();
}

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

std::shared_ptr<ge::OpDesc> EinsumPass::CreateTransposeOpDesc(const NodePtr &node, const std::string &op_name) {
  std::shared_ptr<ge::OpDesc> transpose_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(transpose_desc = std::make_shared<ge::OpDesc>(node->GetName() + op_name, kTranspose),
                          return nullptr);
  return transpose_desc;
}

bool EinsumPass::SetTransposePerm(const std::vector<int32_t> &perm, ge::ComputeGraph &graph,
                                  std::shared_ptr<ge::OpDesc> &transpose_desc, ge::NodePtr &transpose_node) {
  OP_LOGD(kFusedOpType.c_str(), "set perm list[%s] for node[%s]", Vector2Str(perm).c_str(),
          transpose_node->GetName().c_str());
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

Status EinsumPass::LinkEinsumOutputNode(const ge::OutDataAnchor::Vistor<ge::InDataAnchorPtr> &anchors,
                                        const ge::NodePtr &node) const {
  FUSION_PASS_CHECK(node == nullptr, OP_LOGE(kFusedOpType.c_str(), "last node for einsum is null"), return FAILED);
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

void EinsumPass::UnlinkAllDataAnchors(const ge::NodePtr &node) const {
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
  UnlinkAllDataAnchors(origin_node);

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
  FUSION_PASS_CHECK(LinkEinsumOutputNode(out_anchor_peer_anchors, matmul_node) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "link einsum node failed"), return FAILED);

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
  FUSION_PASS_CHECK(reshape_3_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "reshape infershape failed."), return FAILED);
  FUSION_PASS_CHECK(reshape_3_desc->UpdateOutputDesc("y", *(op_desc->MutableOutputDesc(0))) != ge::GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "failed to update reshape output."), return FAILED);
  // unlink
  UnlinkAllDataAnchors(node);

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
  FUSION_PASS_CHECK(LinkEinsumOutputNode(out_anchor_peer_anchors, reshape_3_node) != SUCCESS,
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
  std::shared_ptr<ge::OpDesc> transpose_1_desc = CreateTransposeOpDesc(node, "/Transpose1");
  std::shared_ptr<ge::OpDesc> transpose_2_desc = CreateTransposeOpDesc(node, "/Transpose2");

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
  SetTransposePerm(perm, graph, transpose_2_desc, transpose_2_node);
  FUSION_PASS_CHECK(x1_is_unknown_shape && transpose_2_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "transpose infershape failed."), return FAILED);

  transpose_1_desc->AddInputDesc("x", x0_desc);
  tmp_dims.assign({x0_dims[0], x0_dims[2], x0_dims[1], x0_dims[3]});
  x0_desc.SetShape(GeShape(tmp_dims));
  x0_desc.SetOriginShape(GeShape(tmp_dims));
  transpose_1_desc->AddOutputDesc("y", x0_desc);
  NodePtr transpose_1_node = graph.AddNode(transpose_1_desc);
  SetTransposePerm(perm, graph, transpose_1_desc, transpose_1_node);
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
  std::shared_ptr<ge::OpDesc> transpose_1_desc = CreateTransposeOpDesc(node, "/Transpose1");

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
  SetTransposePerm(perm, graph, transpose_1_desc, transpose_1_node);
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

  std::shared_ptr<ge::OpDesc> transpose_2_desc = CreateTransposeOpDesc(node, "/Transpose2");
  transpose_2_desc->AddInputDesc("x", *(batchmatmul_desc->MutableOutputDesc(0)));
  transpose_2_desc->AddOutputDesc("y", *(op_desc->MutableOutputDesc(0)));
  NodePtr transpose_2_node = graph.AddNode(transpose_2_desc);
  SetTransposePerm(perm, graph, transpose_2_desc, transpose_2_node);

  // unlink
  UnlinkAllDataAnchors(node);

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
  FUSION_PASS_CHECK(LinkEinsumOutputNode(out_anchor_peer_anchors, transpose_2_node) != SUCCESS,
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
  UnlinkAllDataAnchors(node);

  // add edge
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x0_anchor_peer_anchor, batchmatmul_node->GetInDataAnchor(0)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x0_anchor_peer_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, batchmatmul_node->GetInDataAnchor(1)),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                          x1_anchor_peer_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(LinkEinsumOutputNode(out_anchor_peer_anchors, batchmatmul_node) != SUCCESS,
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
  FUSION_PASS_CHECK(reshape_3_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "reshape infershape failed."), return FAILED);
  FUSION_PASS_CHECK(reshape_3_desc->UpdateOutputDesc("y", *(op_desc->MutableOutputDesc(0))) != ge::GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "failed to update reshape output."), return FAILED);

  // unlink
  UnlinkAllDataAnchors(node);

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
  FUSION_PASS_CHECK(LinkEinsumOutputNode(out_anchor_peer_anchors, reshape_3_node) != SUCCESS,
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
    FUSION_PASS_CHECK(reshape_3_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "reshape_3_node is null"),
                      return FAILED);
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
  FUSION_PASS_CHECK(reshape_4_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "reshape infershape failed."), return FAILED);
  FUSION_PASS_CHECK(reshape_4_desc->UpdateOutputDesc("y", *(op_desc->MutableOutputDesc(0))) != ge::GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "failed to update reshape output."), return FAILED);
  // unlink
  UnlinkAllDataAnchors(node);

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
  FUSION_PASS_CHECK(LinkEinsumOutputNode(out_anchor_peer_anchors, reshape_4_node) != SUCCESS,
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
  std::shared_ptr<ge::OpDesc> transpose_1_desc = CreateTransposeOpDesc(node, "/Transpose1");
  std::shared_ptr<ge::OpDesc> transpose_2_desc = CreateTransposeOpDesc(node, "/Transpose2");

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
  SetTransposePerm(perm, graph, transpose_1_desc, transpose_1_node);
  FUSION_PASS_CHECK(x0_is_unknown_shape && transpose_1_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "transpose infershape failed."), return FAILED);

  transpose_2_desc->AddInputDesc("x", x1_desc);
  tmp_dims.assign({x1_dims[0], x1_dims[2], x1_dims[1], x1_dims[3]});
  x1_desc.SetShape(GeShape(tmp_dims));
  x1_desc.SetOriginShape(GeShape(tmp_dims));
  transpose_2_desc->AddOutputDesc("y", x1_desc);
  NodePtr transpose_2_node = graph.AddNode(transpose_2_desc);
  SetTransposePerm(perm, graph, transpose_2_desc, transpose_2_node);
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
  std::shared_ptr<ge::OpDesc> transpose_1_desc = CreateTransposeOpDesc(node, "/Transpose1");

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
  SetTransposePerm(perm, graph, transpose_1_desc, transpose_1_node);
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

  std::shared_ptr<ge::OpDesc> transpose_2_desc = CreateTransposeOpDesc(node, "/Transpose2");
  transpose_2_desc->AddInputDesc("x", *(batchmatmul_desc->MutableOutputDesc(0)));
  transpose_2_desc->AddOutputDesc("y", *(op_desc->MutableOutputDesc(0)));
  NodePtr transpose_2_node = graph.AddNode(transpose_2_desc);
  SetTransposePerm(perm, graph, transpose_2_desc, transpose_2_node);

  // unlink
  UnlinkAllDataAnchors(node);

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
  FUSION_PASS_CHECK(LinkEinsumOutputNode(out_anchor_peer_anchors, transpose_2_node) != SUCCESS,
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
  FUSION_PASS_CHECK(reshape_4_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "reshape infershape failed."), return FAILED);
  FUSION_PASS_CHECK(reshape_4_desc->UpdateOutputDesc("y", *(op_desc->MutableOutputDesc(0))) != ge::GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "failed to update reshape output."), return FAILED);

  // unlink
  UnlinkAllDataAnchors(node);

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
  FUSION_PASS_CHECK(LinkEinsumOutputNode(out_anchor_peer_anchors, reshape_4_node) != SUCCESS,
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
  std::shared_ptr<ge::OpDesc> transpose_2_desc = CreateTransposeOpDesc(node, "/Transpose2");

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
  SetTransposePerm(perm, graph, transpose_2_desc, transpose_2_node);
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

  std::shared_ptr<ge::OpDesc> transpose_3_desc = CreateTransposeOpDesc(node, "/Transpose3");
  transpose_3_desc->AddInputDesc("x", *(batchmatmul_desc->MutableOutputDesc(0)));
  transpose_3_desc->AddOutputDesc("y", *(op_desc->MutableOutputDesc(0)));
  NodePtr transpose_3_node = graph.AddNode(transpose_3_desc);
  SetTransposePerm(perm, graph, transpose_3_desc, transpose_3_node);

  // unlink
  UnlinkAllDataAnchors(node);

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
  FUSION_PASS_CHECK(LinkEinsumOutputNode(out_anchor_peer_anchors, transpose_3_node) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "link einsum node failed"), return FAILED);
  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "remove einsum node failed"), return FAILED);
  return SUCCESS;
}

Status EinsumPass::CheckProduct(const std::vector<int64_t> &shape) const {
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

Status EinsumPass::CheckInputArgs(const Mapping &mapping, bool &is_dynamic_shape) const {
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
  FUSION_PASS_CHECK(node->GetInDataNodes().size() > 2,
                    OP_LOGI(kFusedOpType.c_str(), "Input node of einsum node size is [%lu], which not less than 2.",
                            node->GetInDataNodes().size()),
                    return NOT_CHANGED);

  for (size_t idx = 0; idx < node->GetInDataNodes().size(); ++idx) {
    // get input
    auto x_desc = op_desc->MutableInputDesc(idx);
    FUSION_PASS_CHECK(x_desc == nullptr,
                      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "einsum input desc is null, fusion failed."),
                      return PARAM_INVALID);

    // check whether the tensor size overflows
    if (CheckProduct(x_desc->MutableShape().GetDims()) != SUCCESS) {
      return PARAM_INVALID;
    }

    is_dynamic_shape = is_dynamic_shape || x_desc->MutableShape().IsUnknownShape();
    auto x_anchor = node->GetInDataAnchor(idx);
    FUSION_PASS_CHECK(x_anchor == nullptr,
                      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "einsum x%zu_anchor is null, fusion failed.", idx),
                      return PARAM_INVALID);
    auto x_anchor_peer_anchor = node->GetInDataAnchor(idx)->GetPeerOutAnchor();
    FUSION_PASS_CHECK(
        x_anchor_peer_anchor == nullptr,
        CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "einsum x%zu_anchor_peer_anchor is null, fusion failed.", idx),
        return PARAM_INVALID);
    auto x_anchor_peer_node = x_anchor_peer_anchor->GetOwnerNode();
    FUSION_PASS_CHECK(
        x_anchor_peer_node == nullptr,
        CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "einsum x%zu_anchor_peer_node is null, fusion failed.", idx),
        return PARAM_INVALID);
  }

  auto out_anchor = node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(out_anchor == nullptr,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "einsum out_anchor is null, fusion failed."),
                    return PARAM_INVALID);

  FUSION_PASS_CHECK(is_dynamic_shape && node->GetInDataNodes().size() == 1,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "not support einsum dynamic shape with single input"),
                    return PARAM_INVALID);

  return SUCCESS;
}

// vector<NodePtr> &fusion_nodes: Store fusion nodes,
//       including newly added nodes and fused but not deleted nodes
Status EinsumPass::Fusion(ComputeGraph &graph, Mapping &mapping, vector<NodePtr> & /* fusion_nodes */) {
  OP_LOGI(kFusedOpType.c_str(), "EinsumPass fusion in!");
  bool is_dynamic_shape = false;
  Status ret = CheckInputArgs(mapping, is_dynamic_shape);
  if (ret != SUCCESS) {
    return ret;
  }

  // get attr equation
  std::string equation;
  NodePtr node = GetNodeFromMapping(kPatternFusedNode, mapping);
  OpDescPtr op_desc = node->GetOpDesc();
  FUSION_PASS_CHECK(AttrUtils::GetStr(op_desc, "equation", equation) == false,
                    OP_LOGI(kFusedOpType.c_str(), "Get attr equation failed."), return NOT_CHANGED);

  if (is_dynamic_shape) {
    EquationNormalization(equation);
    std::unordered_map<std::string, ProcFunc> *procs = &dynamicShapeProcs_;
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
  } else {
    // no einsum op in cann, return FAILED is ok
    return SplitOpInFuzzScene(equation, graph, node);
  }
}

static int64_t AccumulateDims(std::vector<int64_t> &dims, const std::vector<int32_t> &indices) {
  int64_t dim_value = 1;
  for (size_t idx = 0; idx < indices.size(); ++idx) {
    if (static_cast<size_t>(indices[idx]) >= dims.size()) {
      break;
    }

    dim_value = GetDimMulValue(dim_value, dims[indices[idx]]);
  }

  return dim_value;
}

void EinsumPass::SplitStr2Vector(const std::string &input, const std::string &delimiter,
                                 std::vector<std::string> &output) const {
  auto delimiter_len = delimiter.size();
  std::string::size_type curr_pos = 0;
  std::string::size_type next_pos = input.find(delimiter, curr_pos);
  while (next_pos != std::string::npos) {
    output.emplace_back(std::move(input.substr(curr_pos, next_pos - curr_pos)));
    curr_pos = next_pos + delimiter_len;
    next_pos = input.find(delimiter, curr_pos);
  }

  if (curr_pos < input.size()) {
    output.emplace_back(std::move(input.substr(curr_pos)));
  }
}

void EinsumPass::CountLabels(const std::string &equation, LabelCount &label_count, std::set<char> &labels) const {
  for (size_t pos = 0; pos < equation.size(); ++pos) {
    char label = equation[pos];
    (void)labels.insert(label);  // to make unique, ignore return value
    if (label == '.') {
      pos += (kBroadCastLabelLen - 1);  // skip next two .
    }

    auto it = label_count.find(label);
    if (it == label_count.end()) {
      (void)label_count.insert({label, 1});  // always succeed, ignore return value
    } else {
      it->second += 1;
    }
  }
}

void EinsumPass::MapDimensionType(const std::set<char> &labels, const LabelCount &input0_label_count,
                                  const LabelCount &input1_label_count, const LabelCount &output_label_count) {
  for (auto label : labels) {
    EinsumDimensionType dim_type = EinsumDimensionType::BROAD_CAST;
    if (label == '.') {
      (void)dim_types_map.insert({label, dim_type});
      continue;
    }

    bool exist_in_input0 = (input0_label_count.find(label) != input0_label_count.end());
    bool exist_in_input1 = (input1_label_count.find(label) != input1_label_count.end());
    bool exist_in_output = (output_label_count.find(label) != output_label_count.end());
    if (exist_in_input0 && exist_in_input1) {
      if (exist_in_output) {
        dim_type = EinsumDimensionType::BATCH;
      } else {
        dim_type = EinsumDimensionType::CONTRACT;
      }
    } else {
      // only exists in one input
      if (exist_in_output) {
        dim_type = EinsumDimensionType::FREE;
      } else {
        dim_type = EinsumDimensionType::REDUCE;
      }
    }

    OP_LOGD(kFusedOpType.c_str(), "label %c dimension type is %s.", label, kDimensionType2Str[dim_type].c_str());
    (void)dim_types_map.insert({label, dim_type});
  }
}

Status EinsumPass::ParseEquation(const std::string &equation, std::vector<std::string> &in_equations,
                                 std::string &out_equation) {
  std::vector<std::string> inputs_and_output;
  SplitStr2Vector(equation, "->", inputs_and_output);
  if (inputs_and_output.size() != 2) {  // 2 means input and output
    OP_LOGE(kFusedOpType.c_str(), "equation[%s] is not support.", equation.c_str());
    return FAILED;
  }

  out_equation = inputs_and_output[1];
  SplitStr2Vector(inputs_and_output[0], ",", in_equations);
  if (in_equations.size() != 1 && in_equations.size() != 2) {  // only support 2 inputs now
    OP_LOGE(kFusedOpType.c_str(), "equation[%s] is not support.", equation.c_str());
    return FAILED;
  }

  std::set<char> labels;
  LabelCount input0_label_count;
  LabelCount input1_label_count;
  LabelCount output_label_count;
  CountLabels(in_equations[0], input0_label_count, labels);
  if (in_equations.size() > 1) {
    CountLabels(in_equations[1], input1_label_count, labels);
  }

  CountLabels(out_equation, output_label_count, labels);
  MapDimensionType(labels, input0_label_count, input1_label_count, output_label_count);
  return SUCCESS;
}

void EinsumPass::CollectDimensionType(size_t dim_num, const std::string &equation,
                                      DimensionType2LabelInfo &label_info) const {
  label_info.resize(EinsumDimensionType::DIM_TYPE_NUM);
  std::string equation_copy(equation);  // replace operation will change str, so need a copy
  auto pos = equation_copy.find(kBroadCastLabel);
  if (pos != std::string::npos) {
    // already check in Verify, check here to fullfil security
    if (dim_num < (equation_copy.size() - kBroadCastLabelLen)) {
      OP_LOGE(kFusedOpType.c_str(), "equation[%s] is valid, dim size[%zu].", equation.c_str(), dim_num);
      return;
    }

    size_t broad_cast_len = dim_num - (equation_copy.size() - kBroadCastLabelLen);
    std::string target_broad_cast_str(broad_cast_len, '.');  // 0 is valid as length here
    // change ... to .... or .. according to acutal broad cast dim length
    equation_copy.replace(equation_copy.begin() + pos, equation_copy.begin() + pos + kBroadCastLabelLen,
                          target_broad_cast_str);
    OP_LOGD(kFusedOpType.c_str(), "equation[%s], new_equation[%s], broadcast length[%zu].", equation.c_str(),
            equation_copy.c_str(), broad_cast_len);
  }

  // now label pos is same as dim index
  for (size_t idx = 0; idx < equation_copy.size(); ++idx) {
    char label = equation_copy[idx];
    // always exists in dim_types_map
    auto it = dim_types_map.find(label);
    if (it != dim_types_map.end()) {
      label_info[it->second].labels.append({label});
      label_info[it->second].indices.push_back(static_cast<int>(idx));  // transpose op perm need int32_t
    }
  }
}

void EinsumPass::ReorderAxes(DimensionType2LabelInfo &label_infos) const {
  for (size_t dim_type = 0; dim_type < label_infos.size(); ++dim_type) {
    auto &labels = label_infos[dim_type].labels;
    auto &indices = label_infos[dim_type].indices;

    // actually, size always equal
    if (labels.size() != indices.size()) {
      continue;
    }

    size_t pos = 0;
    while (pos < labels.size()) {
      size_t curr_pos = pos + 1;
      size_t next_pos = labels.find(labels[pos], curr_pos);
      // to make same label in neighbour
      while (next_pos != std::string::npos) {
        labels.erase(next_pos, 1);
        labels.insert(curr_pos, 1, labels[pos]);
        int32_t index = indices[next_pos];
        indices.erase(indices.begin() + next_pos, indices.begin() + next_pos + 1);
        indices.insert(indices.begin() + curr_pos, index);
        ++curr_pos;
        next_pos = labels.find(labels[pos], curr_pos);
      }

      pos = curr_pos;
    }
  }
}

void EinsumPass::CompareAxes(EinsumDimensionType dim_type, const DimensionType2LabelInfo &target_label_info,
                             DimensionType2LabelInfo &input_label_info) const {
  auto &input_labels = input_label_info[dim_type].labels;
  auto &input_indices = input_label_info[dim_type].indices;
  const auto &output_labels = target_label_info[dim_type].labels;
  if (input_labels.empty() || output_labels.empty()) {
    return;
  }

  std::string new_labels;
  std::vector<int32_t> new_indices;
  new_labels.reserve(input_labels.size());
  new_indices.reserve(input_indices.size());
  size_t pos = 0;
  while (pos < output_labels.size()) {
    char label = output_labels[pos];
    size_t first_pos = input_labels.find_first_of(label);
    if (first_pos != std::string::npos) {
      size_t end_pos = input_labels.find_first_not_of(label, first_pos);
      size_t repeat_num = (end_pos != std::string::npos) ? end_pos - first_pos : input_labels.size() - first_pos;
      new_labels.insert(new_labels.size(), repeat_num, label);
      new_indices.insert(new_indices.end(), input_indices.begin() + first_pos,
                         input_indices.begin() + first_pos + repeat_num);
    }

    do {
      ++pos;
    } while (pos < new_labels.size() && output_labels[pos] == label);
  }

  if (input_labels.size() != new_labels.size()) {
    size_t pos = 0;
    while (pos < input_labels.size()) {
      char label = input_labels[pos];
      size_t new_labels_pos = new_labels.find_first_of(label);
      if (new_labels_pos == std::string::npos) {
        size_t end_pos = input_labels.find_first_not_of(label, pos);
        size_t repeat_num = (end_pos != std::string::npos) ? end_pos - pos : input_labels.size() - pos;
        size_t prev_label_pos = (pos > 0) ? new_labels.rfind(input_labels[pos - 1]) + 1 : 0;
        new_labels.insert(prev_label_pos, repeat_num, label);
        new_indices.insert(new_indices.begin() + prev_label_pos, input_indices.begin() + pos,
                           input_indices.begin() + pos + repeat_num);
        pos += repeat_num;
        continue;
      }

      do {
        ++pos;
      } while (pos < input_labels.size() && input_labels[pos] == label);
    }
  }

  input_labels.swap(new_labels);
  new_indices.swap(input_indices);
  OP_LOGD(kFusedOpType.c_str(), "dimension type[%s] labels[%s]", kDimensionType2Str[dim_type].c_str(),
          input_labels.c_str());
}

bool EinsumPass::GetTransposeDstEquation(const std::string &ori_equation, const DimensionType2LabelInfo &label_infos,
                                         std::vector<int32_t> &perm_list, bool &input_free_contract_order,
                                         std::string &dst_equation) const {
  // performance optimize logic: no batchmatmul output transpose op is better
  for (auto label_iter = ori_equation.rbegin(); label_iter != ori_equation.rend(); ++label_iter) {
    auto it = dim_types_map.find(*label_iter);
    if (it != dim_types_map.end()) {
      if (it->second == EinsumDimensionType::FREE) {
        input_free_contract_order = false;
        break;
      } else if (it->second == EinsumDimensionType::CONTRACT) {
        input_free_contract_order = true;
        break;
      }
    }
  }

  // broad cast lables in label_infos maybe is . or .. or ..... or empty, so need a copied string
  std::string broad_cast_labels;
  std::vector<int32_t> broad_cast_indices;
  std::string ori_equation_copy(ori_equation);
  if (!label_infos[EinsumDimensionType::BROAD_CAST].indices.empty()) {
    broad_cast_labels.assign(kBroadCastLabel);
    broad_cast_indices = label_infos[EinsumDimensionType::BROAD_CAST].indices;
  } else {
    auto pos = ori_equation_copy.find(kBroadCastLabel);
    if (pos != std::string::npos) {
      ori_equation_copy.erase(pos, kBroadCastLabelLen);
    }
  }

  auto &batch_labels = label_infos[EinsumDimensionType::BATCH].labels;
  auto &batch_indices = label_infos[EinsumDimensionType::BATCH].indices;
  auto &free_labels = label_infos[EinsumDimensionType::FREE].labels;
  auto &free_indices = label_infos[EinsumDimensionType::FREE].indices;
  auto &contract_labels = label_infos[EinsumDimensionType::CONTRACT].labels;
  auto &contract_indices = label_infos[EinsumDimensionType::CONTRACT].indices;
  auto &reduce_labels = label_infos[EinsumDimensionType::REDUCE].labels;
  auto &reduce_indices = label_infos[EinsumDimensionType::REDUCE].indices;

  dst_equation.reserve(ori_equation.size());
  dst_equation.append(broad_cast_labels).append(batch_labels);
  perm_list.insert(perm_list.end(), broad_cast_indices.begin(), broad_cast_indices.end());
  perm_list.insert(perm_list.end(), batch_indices.begin(), batch_indices.end());
  if (input_free_contract_order) {
    dst_equation.append(free_labels).append(contract_labels).append(reduce_labels);
    perm_list.insert(perm_list.end(), free_indices.begin(), free_indices.end());
    perm_list.insert(perm_list.end(), contract_indices.begin(), contract_indices.end());
  } else {
    dst_equation.append(contract_labels).append(free_labels).append(reduce_labels);
    perm_list.insert(perm_list.end(), contract_indices.begin(), contract_indices.end());
    perm_list.insert(perm_list.end(), free_indices.begin(), free_indices.end());
  }

  perm_list.insert(perm_list.end(), reduce_indices.begin(), reduce_indices.end());
  return dst_equation != ori_equation_copy;
}

GeTensorDescPtr EinsumPass::GetPrevOutputDesc(const NodePtr &node, size_t idx) const {
  if (bmm_input_nodes[idx].empty()) {
    auto op_desc = node->GetOpDesc();
    auto x_desc = op_desc->MutableInputDesc(idx);
    return x_desc;
  } else {
    auto op_desc = bmm_input_nodes[idx].back()->GetOpDesc();
    return op_desc->MutableOutputDesc(0);
  }
}

GeTensorDescPtr EinsumPass::GetPrevOutputDescAfterBmm(const NodePtr &node) const {
  if (bmm_output_nodes.empty()) {
    if (batchmatmul_node == nullptr) {
      return GetPrevOutputDesc(node, 0);
    }
    auto op_desc = batchmatmul_node->GetOpDesc();
    auto x_desc = op_desc->MutableOutputDesc(0);
    return x_desc;
  } else {
    auto op_desc = bmm_output_nodes.back()->GetOpDesc();
    return op_desc->MutableOutputDesc(0);
  }
}

Status EinsumPass::TransposeInput(const std::vector<std::string> &in_equations, const NodePtr &node,
                                  ge::ComputeGraph &graph) {
  input_free_contract_orders.resize(in_equations.size());
  input_label_infos.resize(in_equations.size());
  bmm_input_nodes.resize(in_equations.size());
  for (size_t idx = 0; idx < in_equations.size(); ++idx) {
    auto x_desc = GetPrevOutputDesc(node, idx);
    CollectDimensionType(x_desc->MutableShape().GetDimNum(), in_equations[idx], input_label_infos[idx]);
    ReorderAxes(input_label_infos[idx]);
    CompareAxes(EinsumDimensionType::BATCH, output_label_info, input_label_infos[idx]);
    CompareAxes(EinsumDimensionType::FREE, output_label_info, input_label_infos[idx]);
    if (idx > 0) {
      CompareAxes(EinsumDimensionType::CONTRACT, input_label_infos[0], input_label_infos[idx]);
    }

    std::vector<int32_t> perm_list;
    perm_list.reserve(x_desc->MutableShape().GetDimNum());
    bool input_free_contract_order = false;
    std::string dst_equation;
    if (GetTransposeDstEquation(in_equations[idx], input_label_infos[idx], perm_list, input_free_contract_order,
                                dst_equation)) {
      input_free_contract_orders[idx] = input_free_contract_order;
      auto transpose_desc = CreateTransposeOpDesc(node, "/Transpose" + std::to_string(transpose_seq++));
      FUSION_PASS_CHECK(transpose_desc == nullptr, OP_LOGE(kFusedOpType.c_str(), "tranpose_desc is null"),
                        return FAILED);
      transpose_desc->AddInputDesc("x", *x_desc);
      transpose_desc->AddOutputDesc("y", *x_desc);
      auto transpose_node = graph.AddNode(transpose_desc);
      FUSION_PASS_CHECK(transpose_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "transpose_node is null"),
                        return FAILED);
      SetTransposePerm(perm_list, graph, transpose_desc, transpose_node);
      FUSION_PASS_CHECK(
          transpose_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
          OP_LOGE(kFusedOpType.c_str(), "einsum[%s] transpose infershape failed", node->GetName().c_str()),
          return FAILED);
      bmm_input_nodes[idx].emplace_back(transpose_node);
      OP_LOGD(kFusedOpType.c_str(), "einsum[%s] input%zu transpose output[%s], equation[%s], free contract order[%s]",
              node->GetName().c_str(), idx, transpose_desc->MutableOutputDesc(0)->MutableShape().ToString().c_str(),
              dst_equation.c_str(), input_free_contract_order ? "true" : "false");
    } else {
      input_free_contract_orders[idx] = input_free_contract_order;
      OP_LOGD(kFusedOpType.c_str(), "no need to transpose input%zu for equation[%s], free contract order[%s]", idx,
              in_equations[idx].c_str(), input_free_contract_order ? "true" : "false");
    }
  }

  return SUCCESS;
}

Status EinsumPass::TransposeOutput(const std::string &out_equation, const NodePtr &node, ge::ComputeGraph &graph,
                                   std::string &cur_out_equation) {
  FUSION_PASS_CHECK(cur_out_equation == out_equation,
                    OP_LOGD(kFusedOpType.c_str(), "No need output transpose for einsum[%s].", node->GetName().c_str()),
                    return SUCCESS);

  // get ellipsis dim size
  auto x_desc = GetPrevOutputDescAfterBmm(node);
  auto x_dim_size = x_desc->MutableShape().GetDimNum();
  bool ell_dim_flag = out_equation.find(kBroadCastLabel) != std::string::npos;
  auto out_label_dim = ell_dim_flag ? out_equation.size() - 3 : out_equation.size();
  auto ell_dim_size = x_dim_size - out_label_dim;
  auto ell_dim_offset = ell_dim_flag ? 3 : 0;

  // get perm
  std::vector<int32_t> perm_list;
  perm_list.reserve(x_dim_size);
  for (size_t i = 0; i < out_equation.size(); i++) {
    if (out_equation[i] == '.') {
      // broadcast dim must appear in head of cur_out_equation
      std::vector<int32_t> ell_indice(ell_dim_size);
      std::iota(ell_indice.begin(), ell_indice.end(), 0);
      perm_list.insert(perm_list.end(), ell_indice.begin(), ell_indice.end());
      // skip next ..
      i += (kBroadCastLabelLen - 1);
    } else {
      auto cur_idx =
          static_cast<int32_t>(cur_out_equation.find_first_of(out_equation[i]) - ell_dim_offset + ell_dim_size);
      perm_list.insert(perm_list.end(), cur_idx);
    }
  }

  std::vector<int32_t> no_transpose_list(x_dim_size);
  std::iota(no_transpose_list.begin(), no_transpose_list.end(), 0);
  if (perm_list == no_transpose_list) {
    OP_LOGD(kFusedOpType.c_str(), "No need output transpose for node [%s], which perm is [%s].",
            node->GetName().c_str(), Vector2Str(perm_list).c_str());
    return SUCCESS;
  }

  auto transpose_desc = CreateTransposeOpDesc(node, "/Transpose" + std::to_string(transpose_seq++));
  FUSION_PASS_CHECK(transpose_desc == nullptr, OP_LOGE(kFusedOpType.c_str(), "tranpose_desc is null"), return FAILED);
  transpose_desc->AddInputDesc("x", *x_desc);
  transpose_desc->AddOutputDesc("y", *x_desc);
  auto transpose_node = graph.AddNode(transpose_desc);
  FUSION_PASS_CHECK(transpose_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "transpose_node is null"), return FAILED);
  SetTransposePerm(perm_list, graph, transpose_desc, transpose_node);
  FUSION_PASS_CHECK(transpose_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "einsum[%s] transpose infershape failed", node->GetName().c_str()),
                    return FAILED);
  bmm_output_nodes.emplace_back(transpose_node);
  OP_LOGD(kFusedOpType.c_str(), "einsum[%s] transpose output shape: [%s]", node->GetName().c_str(),
          transpose_desc->MutableOutputDesc(0)->MutableShape().ToString().c_str());
  return SUCCESS;
}

Status EinsumPass::StrideInput(const std::vector<std::string> &in_equations) const {
  for (size_t idx = 0; idx < in_equations.size(); ++idx) {
    const std::string &in_equation = in_equations[idx];
    for (size_t pos = 0; pos < in_equation.size(); ++pos) {
      char label = in_equation[pos];
      if (label == '.') {
        pos += (kBroadCastLabelLen - 1);  // skip next two .
        continue;
      }

      if (in_equation.find(label, pos + 1) != std::string::npos) {
        OP_LOGE(kFusedOpType.c_str(), "not support stride input now, equation: %s", in_equation.c_str());
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

Status EinsumPass::InflatedOutput(const std::string &out_equation) const {
  std::string out_equation_bak(out_equation);
  auto start_idx = out_equation_bak.find(kBroadCastLabel);
  int ellipsis_len = 3;
  if (start_idx != std::string::npos) {
    out_equation_bak.erase(start_idx, ellipsis_len);
  }
  // temporarily not support inflated
  std::set<char> out_label_set(out_equation_bak.begin(), out_equation_bak.end());
  if (out_equation_bak.size() != out_label_set.size()) {
    OP_LOGE(kFusedOpType.c_str(), "not support inflated now, equation: %s", out_equation.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status EinsumPass::ReduceInput(const std::vector<std::string> &in_equations, const NodePtr &node,
                               ge::ComputeGraph &graph) {
  for (size_t idx = 0; idx < in_equations.size(); ++idx) {
    auto &indices = input_label_infos[idx][EinsumDimensionType::REDUCE].indices;
    if (!indices.empty()) {
      std::shared_ptr<ge::OpDesc> reduce_desc;
      FUSION_PASS_MAKE_SHARED(reduce_desc = std::make_shared<ge::OpDesc>(
                                  node->GetName() + "/ReduceSum" + std::to_string(reduce_seq++), kReduceSumD),
                              return FAILED);
      auto x_desc = GetPrevOutputDesc(node, idx);
      size_t dim_num = x_desc->MutableShape().GetDimNum();
      FUSION_PASS_CHECK(
          dim_num < indices.size(),
          OP_LOGE(kFusedOpType.c_str(), "einsum[%s] prev output dim number[%zu] less than than reduce dim num[%zu]",
                  node->GetName().c_str(), dim_num, indices.size()),
          return FAILED);

      std::vector<int32_t> axes;
      axes.reserve(indices.size());
      for (size_t axis = dim_num - indices.size(); axis < dim_num; ++axis) {
        axes.push_back(static_cast<int32_t>(axis));
      }

      reduce_desc->AddInputDesc("x", *x_desc);
      reduce_desc->AddOutputDesc("y", *x_desc);
      auto reduce_node = graph.AddNode(reduce_desc);
      FUSION_PASS_CHECK(reduce_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "reduce_node is null"), return FAILED);
      FUSION_PASS_CHECK(
          !(AttrUtils::SetListInt(reduce_desc, "axes", axes) && AttrUtils::SetBool(reduce_desc, "keep_dims", false)),
          OP_LOGE(kFusedOpType.c_str(), "failed to set attr for reduce."), return FAILED);
      FUSION_PASS_CHECK(
          reduce_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
          OP_LOGE(kFusedOpType.c_str(), "einsum[%s] reducesum infershape failed.", node->GetName().c_str()),
          return FAILED);
      bmm_input_nodes[idx].emplace_back(reduce_node);
      OP_LOGD(kFusedOpType.c_str(), "einsum[%s] input%zu reduce output shape: [%s]", node->GetName().c_str(), idx,
              reduce_desc->MutableOutputDesc(0)->MutableShape().ToString().c_str());
    }
  }

  return SUCCESS;
}

void EinsumPass::CheckMergeFreeLabels(const NodePtr &node, const std::string &out_equation) {
  size_t input0_batch_num = input_label_infos[0][EinsumDimensionType::BATCH].indices.size();
  size_t input1_batch_num = input_label_infos[1][EinsumDimensionType::BATCH].indices.size();
  if (input0_batch_num > 0 || input1_batch_num > 0) {
    return;
  }

  std::string out_equation_copy(out_equation);
  auto pos = out_equation_copy.find(kBroadCastLabel);
  if (pos != std::string::npos) {
    out_equation_copy.erase(pos, kBroadCastLabelLen);
  }

  size_t input0_free_num = input_label_infos[0][EinsumDimensionType::FREE].indices.size();
  size_t input1_free_num = input_label_infos[1][EinsumDimensionType::FREE].indices.size();
  size_t input0_broad_cast_num = input_label_infos[0][EinsumDimensionType::BROAD_CAST].indices.size();
  size_t input1_broad_cast_num = input_label_infos[1][EinsumDimensionType::BROAD_CAST].indices.size();
  if (input0_free_num == 1 && input1_free_num > 1 && input_free_contract_orders[1]) {
    std::string free_labels(input_label_infos[1][EinsumDimensionType::FREE].labels);
    free_labels.append(input_label_infos[0][EinsumDimensionType::FREE].labels);
    merge_free_labels = !(input0_broad_cast_num == 0 && free_labels == out_equation_copy);
  } else if (input0_free_num > 1 && input1_free_num == 1 && input_free_contract_orders[0]) {
    std::string free_labels(input_label_infos[0][EinsumDimensionType::FREE].labels);
    free_labels.append(input_label_infos[1][EinsumDimensionType::FREE].labels);
    merge_free_labels = !(input1_broad_cast_num == 0 && free_labels == out_equation_copy);
  }

  OP_LOGD(kFusedOpType.c_str(), "%smerge free labels in front of batchmatmul for einsum[%s]",
          merge_free_labels ? "" : "not ", node->GetName().c_str());
  return;
}

Status EinsumPass::ReshapeInput(const std::vector<std::string> &in_equations, const std::string &out_equation,
                                const NodePtr &node, ComputeGraph &graph) {
  if (in_equations.size() == 1) {
    OP_LOGD(kFusedOpType.c_str(), "single input, no need reshape in front of batchmatmul for einsum[%s]",
            node->GetName().c_str());
    return SUCCESS;
  }

  CheckMergeFreeLabels(node, out_equation);
  auto op_desc = node->GetOpDesc();
  for (size_t idx = 0; idx < in_equations.size(); ++idx) {
    auto x_desc = GetPrevOutputDesc(node, idx);
    auto dims = x_desc->MutableShape().GetDims();
    auto ori_dims = op_desc->MutableInputDesc(idx)->MutableShape().GetDims();
    auto &input_label_info = input_label_infos[idx];
    std::vector<EinsumDimensionType> check_orders({EinsumDimensionType::FREE, EinsumDimensionType::CONTRACT});
    if (!input_free_contract_orders[idx]) {
      check_orders.assign({EinsumDimensionType::CONTRACT, EinsumDimensionType::FREE});
    }

    std::vector<int64_t> new_dims;
    new_dims.reserve(dims.size());
    size_t broadcast_batch_num = input_label_info[EinsumDimensionType::BROAD_CAST].indices.size() +
                                 input_label_info[EinsumDimensionType::BATCH].indices.size();
    new_dims.assign(dims.begin(), dims.begin() + broadcast_batch_num);
    for (EinsumDimensionType dim_type : check_orders) {
      auto &indices = input_label_info[dim_type].indices;
      if (merge_free_labels || dim_type != EinsumDimensionType::FREE) {
        new_dims.push_back(AccumulateDims(ori_dims, indices));
        continue;
      }

      for (auto index : indices) {
        new_dims.push_back(ori_dims[index]);
      }
    }

    if (new_dims == dims) {
      OP_LOGD(kFusedOpType.c_str(), "no need to reshape input[%zu] for einsum[%s]", idx, node->GetName().c_str());
      continue;
    }

    auto reshape_desc = CreateReshapeOpDesc(x_desc->MutableShape().IsUnknownShape(), node, reshape_seq++);
    FUSION_PASS_CHECK(reshape_desc == nullptr, OP_LOGE(kFusedOpType.c_str(), "reshape_desc is null"), return FAILED);
    reshape_desc->AddInputDesc("x", *x_desc);
    reshape_desc->AddOutputDesc("y", *x_desc);
    reshape_desc->MutableOutputDesc(0)->SetShape(GeShape(new_dims));
    reshape_desc->MutableOutputDesc(0)->SetOriginShape(GeShape(new_dims));
    auto reshape_node = CreateReshapeNode(new_dims, graph, reshape_desc);
    FUSION_PASS_CHECK(reshape_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "reshape_node is null"), return FAILED);
    bmm_input_nodes[idx].emplace_back(reshape_node);
    OP_LOGD(kFusedOpType.c_str(), "input%zu reshape output shape: [%s]", idx,
            reshape_desc->MutableOutputDesc(0)->MutableShape().ToString().c_str());
  }

  return SUCCESS;
}

void EinsumPass::CalcBatchMatmulOutput(size_t input_num, const NodePtr &node, bool &need_reshape,
                                       std::string &bmm_out_equation, std::vector<int64_t> &bmm_dims) const {
  std::string out_free_labels;
  std::vector<int64_t> out_free_dim;

  // get ori free dimensions and labels in bmm out shape
  auto op_desc = node->GetOpDesc();
  for (size_t idx = 0; idx < input_num; ++idx) {
    auto &free_indices = input_label_infos[idx][EinsumDimensionType::FREE].indices;
    auto &free_labels = input_label_infos[idx][EinsumDimensionType::FREE].labels;
    if (free_indices.empty()) {
      continue;
    }

    need_reshape = need_reshape || free_indices.size() > 1;
    auto ori_dims = op_desc->MutableInputDesc(idx)->MutableShape().GetDims();
    if (swap_bmm_inputs) {
      out_free_labels.insert(out_free_labels.begin(), free_labels.begin(), free_labels.end());
      for (size_t offset = 0; offset < free_indices.size(); ++offset) {
        out_free_dim.insert(out_free_dim.begin() + offset, ori_dims[free_indices[offset]]);
      }
    } else {
      out_free_labels.insert(out_free_labels.end(), free_labels.begin(), free_labels.end());
      for (auto i : free_indices) {
        out_free_dim.push_back(ori_dims[i]);
      }
    }
  }

  // ori batch dimensions in bmm out shape, batch labels appear in each input
  std::vector<int64_t> out_batch_dim;
  auto &batch_indices = input_label_infos[0][EinsumDimensionType::BATCH].indices;
  size_t batch_num = (batch_indices.empty()) ? 0 : batch_indices.size();
  std::string batch_labels(input_label_infos[0][EinsumDimensionType::BATCH].labels);
  auto ori_dims_batch = op_desc->MutableInputDesc(0)->MutableShape().GetDims();
  for (auto i : batch_indices) {
    out_batch_dim.push_back(ori_dims_batch[i]);
  }

  // get reshape out shape
  size_t free_num = merge_free_labels ? 2
                                      : input_label_infos[0][EinsumDimensionType::FREE].indices.size() +
                                            input_label_infos[1][EinsumDimensionType::FREE].indices.size();
  bmm_dims.erase(bmm_dims.end() - batch_num - free_num, bmm_dims.end());
  bmm_dims.insert(bmm_dims.end(), out_batch_dim.begin(), out_batch_dim.end());
  bmm_dims.insert(bmm_dims.end(), out_free_dim.begin(), out_free_dim.end());

  // get out_equation after reshape
  if (bmm_out_equation.find(kBroadCastLabel) != std::string::npos) {
    bmm_out_equation.assign(kBroadCastLabel);
  } else {
    bmm_out_equation.clear();
  }
  bmm_out_equation.append(batch_labels).append(out_free_labels);
  OP_LOGD(kFusedOpType.c_str(), "batchmatmul output equation is: [%s] for eimsum[%s].", bmm_out_equation.c_str(),
          node->GetName().c_str());
}

Status EinsumPass::ReshapeOutput(const std::string &ori_out_equation, const NodePtr &node, ge::ComputeGraph &graph,
                                 std::string &cur_out_equation, const std::vector<std::string> &in_equations) {
  FUSION_PASS_CHECK(in_equations.size() == 1,
                    OP_LOGD(kFusedOpType.c_str(), "single input, no need reshape output for einsum[%s].",
                            node->GetName().c_str()),
                    return SUCCESS);

  bool need_reshape = false;
  auto bmm_desc = GetPrevOutputDescAfterBmm(node);
  auto bmm_dims = bmm_desc->MutableShape().GetDims();
  auto bmm_ori_len = bmm_dims.size();
  CalcBatchMatmulOutput(in_equations.size(), node, need_reshape, cur_out_equation, bmm_dims);
  if (!need_reshape && bmm_ori_len == bmm_dims.size()) {
    OP_LOGD(kFusedOpType.c_str(), "This output, no need reshape for einsum[%s].", node->GetName().c_str());
    return SUCCESS;
  }

  if (!merge_free_labels) {
    OP_LOGD(kFusedOpType.c_str(), "not merge free labels, no need reshape output for einsum[%s].",
            node->GetName().c_str());
    return SUCCESS;
  }

  // insert reshape node
  auto reshape_desc = CreateReshapeOpDesc(bmm_desc->MutableShape().IsUnknownShape(), node, reshape_seq++);
  FUSION_PASS_CHECK(reshape_desc == nullptr, OP_LOGE(kFusedOpType.c_str(), "reshape_desc is null"), return FAILED);

  reshape_desc->AddInputDesc("x", *bmm_desc);
  reshape_desc->AddOutputDesc("y", *bmm_desc);

  reshape_desc->MutableOutputDesc(0)->SetShape(GeShape(bmm_dims));
  reshape_desc->MutableOutputDesc(0)->SetOriginShape(GeShape(bmm_dims));

  auto reshape_node = CreateReshapeNode(bmm_dims, graph, reshape_desc);
  FUSION_PASS_CHECK(reshape_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "reshape_node is null"), return FAILED);
  bmm_output_nodes.emplace_back(reshape_node);
  OP_LOGD(kFusedOpType.c_str(), "einsum[%s] reshape output shape: [%s].", node->GetName().c_str(),
          reshape_desc->MutableOutputDesc(0)->MutableShape().ToString().c_str());
  return SUCCESS;
}

void EinsumPass::CheckBatchMatmulSwapInputs() {
  // if all input1 free labels in front of input0 free labels
  auto &input1_free_lable_infos = input_label_infos[1][EinsumDimensionType::FREE];
  if (input1_free_lable_infos.labels.empty()) {
    return;
  }

  auto &output_free_lable_infos = output_label_info[EinsumDimensionType::FREE];
  bool free_label_from_input1 = true;
  for (char label : output_free_lable_infos.labels) {
    if (free_label_from_input1) {
      free_label_from_input1 = input1_free_lable_infos.labels.find(label) != std::string::npos;
    } else {
      if (input1_free_lable_infos.labels.find(label) != std::string::npos) {
        return;
      }
    }
  }

  swap_bmm_inputs = true;
}

Status EinsumPass::DoBatchMatmul(const std::vector<std::string> &in_equations, const ge::NodePtr &node,
                                 ge::ComputeGraph &graph) {
  if (in_equations.size() == 1) {
    OP_LOGD(kFusedOpType.c_str(), "single input, no need batchmatmul for einsum[%s]", node->GetName().c_str());
    return SUCCESS;
  }

  std::shared_ptr<ge::OpDesc> batchmatmul_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(batchmatmul_desc = std::make_shared<ge::OpDesc>(
                              node->GetName() + "/BatchMatMul" + std::to_string(batchmatmul_seq), kBatchMatMulV2),
                          return FAILED);

  // free contract, free contract: adj_x1 = false, adj_x2 = true
  // free contract, contract free: adj_x1 = false, adj_x2 = false
  // contract free, free contract: adj_x1 = true, adj_x2 = true
  // contract free, contract free: adj_x1 = true, adj_x2 = false
  bool adj_x1 = !(input_free_contract_orders[0]);
  bool adj_x2 = input_free_contract_orders[1];

  auto x1_desc = GetPrevOutputDesc(node, 0);
  auto x2_desc = GetPrevOutputDesc(node, 1);
  CheckBatchMatmulSwapInputs();
  if (swap_bmm_inputs) {
    OP_LOGD(kFusedOpType.c_str(), "swap einsum[%s] batchmatmul inputs", node->GetName().c_str());
    batchmatmul_desc->AddInputDesc("x1", *x2_desc);
    batchmatmul_desc->AddInputDesc("x2", *x1_desc);
    adj_x1 = !(input_free_contract_orders[1]);
    adj_x2 = input_free_contract_orders[0];
  } else {
    batchmatmul_desc->AddInputDesc("x1", *x1_desc);
    batchmatmul_desc->AddInputDesc("x2", *x2_desc);
  }
  batchmatmul_desc->AddOutputDesc("y", *x1_desc);
  FUSION_PASS_CHECK(!(AttrUtils::SetBool(batchmatmul_desc, "adj_x1", adj_x1) &&
                      AttrUtils::SetBool(batchmatmul_desc, "adj_x2", adj_x2)),
                    OP_LOGE(kFusedOpType.c_str(), "failed to set attr for reduce."), return FAILED);

  batchmatmul_node = graph.AddNode(batchmatmul_desc);
  FUSION_PASS_CHECK(batchmatmul_node == nullptr, OP_LOGE(kFusedOpType.c_str(), "batchmatmul_node is null"),
                    return FAILED);
  FUSION_PASS_CHECK(batchmatmul_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "batchmatmul infershape failed."), return FAILED);
  OP_LOGD(kFusedOpType.c_str(), "einsum[%s] batchmatmul output shape: [%s], adj_x1: %d, adj_x2: %d",
          node->GetName().c_str(), batchmatmul_desc->MutableOutputDesc(0)->MutableShape().ToString().c_str(), adj_x1,
          adj_x2);
  return SUCCESS;
}

Status EinsumPass::LinkEinsumInputNode(const NodePtr &node, const NodePtr &first_node, int32_t anchor_idx,
                                       int32_t first_node_anchor_idx) const {
  auto x_anchor_peer_anchor = node->GetInDataAnchor(anchor_idx)->GetPeerOutAnchor();
  auto x_anchor_peer_node = x_anchor_peer_anchor->GetOwnerNode();
  node->GetInDataAnchor(anchor_idx)->UnlinkAll();

  FUSION_PASS_CHECK(
      SUCCESS != GraphUtils::AddEdge(x_anchor_peer_anchor, first_node->GetInDataAnchor(first_node_anchor_idx)),
      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                            x_anchor_peer_node->GetName().c_str(), first_node->GetName().c_str()),
      return FAILED);
  return SUCCESS;
}

Status EinsumPass::LinkContinuousNodes(const std::vector<ge::NodePtr> &nodes) const {
  // ReLinkNodes make sure nodes non empty
  for (size_t seq = 0; seq < nodes.size() - 1; ++seq) {
    const NodePtr &curr_node = nodes[seq];
    const NodePtr &next_node = nodes[seq + 1];
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(curr_node->GetOutDataAnchor(0), next_node->GetInDataAnchor(0)),
                      CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                            curr_node->GetName().c_str(), next_node->GetName().c_str()),
                      return FAILED);
  }

  return SUCCESS;
}

Status EinsumPass::ReLinkCtrlEdges(const NodePtr &node, const NodePtr &last_node) const {
  // input control anchor
  if (node->GetInControlAnchor() != nullptr && !node->GetInControlAnchor()->GetPeerOutControlAnchors().empty()) {
    for (size_t idx = 0; idx < bmm_input_nodes.size(); ++idx) {
      NodePtr first_node = nullptr;
      if (!bmm_input_nodes[idx].empty()) {
        first_node = bmm_input_nodes[idx][0];
      } else if (batchmatmul_node != nullptr) {
        first_node = batchmatmul_node;
      } else if (!bmm_output_nodes.empty()) {
        first_node = bmm_output_nodes[0];
      }

      if (first_node == nullptr || first_node->GetInControlAnchor() == nullptr) {
        continue;
      }

      OP_LOGI(kFusedOpType.c_str(), "relink einsum[%s] input control anchors.", node->GetName().c_str());
      for (auto &peer_anchor : node->GetInControlAnchor()->GetPeerOutControlAnchors()) {
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(peer_anchor, first_node->GetInControlAnchor()),
            OP_LOGE(kFusedOpType.c_str(), "Fail to add input control edge for node:%s.", first_node->GetName().c_str()),
            return FAILED);
      }
    }
    node->GetInControlAnchor()->UnlinkAll();
  }

  // out control anchor
  if (node->GetOutControlAnchor() != nullptr) {
    if (!node->GetOutControlAnchor()->GetPeerInControlAnchors().empty() &&
        last_node->GetOutControlAnchor() != nullptr) {
      OP_LOGI(kFusedOpType.c_str(), "relink einsum[%s] input control anchors.", node->GetName().c_str());
      for (auto &peer_anchor : node->GetOutControlAnchor()->GetPeerInControlAnchors()) {
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(last_node->GetOutControlAnchor(), peer_anchor),
            OP_LOGE(kFusedOpType.c_str(), "Fail to add output control edge for node:%s.", last_node->GetName().c_str()),
            return FAILED);
      }
    }
    node->GetOutControlAnchor()->UnlinkAll();
  }

  return SUCCESS;
}

Status EinsumPass::ReLinkNodes(const NodePtr &node) {
  NodePtr orig_x0_node = node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  // add edge
  for (size_t idx = 0; idx < bmm_input_nodes.size(); ++idx) {
    if (bmm_input_nodes[idx].empty()) {
      continue;
    }

    FUSION_PASS_CHECK(LinkEinsumInputNode(node, bmm_input_nodes[idx][0], static_cast<int>(idx), 0) != SUCCESS,
                      OP_LOGE(kFusedOpType.c_str(), "failed to link einsum input to bmm_input_nodes"), return FAILED);
    FUSION_PASS_CHECK(LinkContinuousNodes(bmm_input_nodes[idx]) != SUCCESS,
                      OP_LOGE(kFusedOpType.c_str(), "failed to link bmm_input_nodes"), return FAILED);
  }

  if (batchmatmul_node != nullptr) {
    int32_t in_anchor = swap_bmm_inputs ? 1 : 0;
    if (bmm_input_nodes[0].empty()) {
      FUSION_PASS_CHECK(LinkEinsumInputNode(node, batchmatmul_node, 0, in_anchor) != SUCCESS,
                        OP_LOGE(kFusedOpType.c_str(), "failed to link einsum input to batchmatmul"), return FAILED);
    } else {
      NodePtr &x1_node = bmm_input_nodes[0].back();
      FUSION_PASS_CHECK(
          SUCCESS != GraphUtils::AddEdge(x1_node->GetOutDataAnchor(0), batchmatmul_node->GetInDataAnchor(in_anchor)),
          CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                x1_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
          return FAILED);
    }

    in_anchor = swap_bmm_inputs ? 0 : 1;
    if (bmm_input_nodes[1].empty()) {
      FUSION_PASS_CHECK(LinkEinsumInputNode(node, batchmatmul_node, 1, in_anchor) != SUCCESS,
                        OP_LOGE(kFusedOpType.c_str(), "failed to link einsum input to batchmatmul"), return FAILED);
    } else {
      NodePtr &x2_node = bmm_input_nodes[1].back();
      FUSION_PASS_CHECK(
          SUCCESS != GraphUtils::AddEdge(x2_node->GetOutDataAnchor(0), batchmatmul_node->GetInDataAnchor(in_anchor)),
          CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                x2_node->GetName().c_str(), batchmatmul_node->GetName().c_str()),
          return FAILED);
    }
  }

  NodePtr prev_node = (batchmatmul_node != nullptr)
                          ? batchmatmul_node
                          : (bmm_input_nodes[0].empty() ? orig_x0_node : bmm_input_nodes[0].back());
  bmm_output_nodes.insert(bmm_output_nodes.begin(), prev_node);
  FUSION_PASS_CHECK(LinkContinuousNodes(bmm_output_nodes) != SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "failed to link bmm_output_nodes"), return FAILED);

  auto out_anchor_peer_anchors = node->GetOutDataAnchor(0)->GetPeerInDataAnchors();
  node->GetOutDataAnchor(0)->UnlinkAll();
  NodePtr last_node = !bmm_output_nodes.empty() ? bmm_output_nodes.back()
                                                : (batchmatmul_node != nullptr ? batchmatmul_node : orig_x0_node);
  FUSION_PASS_CHECK(LinkEinsumOutputNode(out_anchor_peer_anchors, last_node) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "link einsum node output failed"), return FAILED);
  return ReLinkCtrlEdges(node, last_node);
}

void EinsumPass::ResetFusionPass() {
  transpose_seq = 1;
  reduce_seq = 1;
  reshape_seq = 1;
  batchmatmul_seq = 1;
  swap_bmm_inputs = false;
  merge_free_labels = true;
  dim_types_map.clear();
  input_free_contract_orders.clear();
  bmm_input_nodes.clear();
  input_label_infos.clear();
  output_label_info.clear();
  ori_output_label_info.clear();
  batchmatmul_node = nullptr;
  bmm_output_nodes.clear();
}

Status EinsumPass::SplitOpInFuzzScene(const std::string &equation, ComputeGraph &graph, NodePtr &node) {
  ResetFusionPass();
  std::vector<std::string> in_equations;
  std::string out_equation;
  FUSION_PASS_CHECK(ParseEquation(equation, in_equations, out_equation) != SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "failed to parse equation"), return FAILED);
  std::string cur_out_equation(out_equation);
  auto op_desc = node->GetOpDesc();
  auto y_desc = op_desc->MutableOutputDesc(0);
  CollectDimensionType(y_desc->MutableShape().GetDimNum(), out_equation, output_label_info);
  ori_output_label_info = output_label_info;
  ReorderAxes(output_label_info);

  bmm_input_nodes.resize(in_equations.size());
  FUSION_PASS_CHECK(TransposeInput(in_equations, node, graph) != SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "failed to process input transpose"), return FAILED);

  FUSION_PASS_CHECK(StrideInput(in_equations) != SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "failed to process input stirde"), return FAILED);

  FUSION_PASS_CHECK(ReduceInput(in_equations, node, graph) != SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "failed to process input reduce"), return FAILED);

  FUSION_PASS_CHECK(ReshapeInput(in_equations, out_equation, node, graph) != SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "failed to process input reshape"), return FAILED);

  FUSION_PASS_CHECK(DoBatchMatmul(in_equations, node, graph) != SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "failed to process batchmatmul"), return FAILED);

  FUSION_PASS_CHECK(ReshapeOutput(out_equation, node, graph, cur_out_equation, in_equations) != SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "failed to process output reshape"), return FAILED);

  FUSION_PASS_CHECK(InflatedOutput(out_equation) != SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "failed to process output inflated"), return FAILED);

  FUSION_PASS_CHECK(TransposeOutput(out_equation, node, graph, cur_out_equation) != SUCCESS,
                    OP_LOGE(kFusedOpType.c_str(), "failed to process output transpose"), return FAILED);

  auto last_y_desc = GetPrevOutputDescAfterBmm(node);
  FUSION_PASS_CHECK(
      !(last_y_desc->MutableShape() == y_desc->MutableShape()),
      OP_LOGE(kFusedOpType.c_str(), "last node output not equal with origin einsum[%s]", node->GetName().c_str()),
      return FAILED);

  FUSION_PASS_CHECK(ReLinkNodes(node) != SUCCESS, OP_LOGE(kFusedOpType.c_str(), "failed to relink nodes"),
                    return FAILED);

  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(node),
                    CUBE_INNER_ERR_REPORT(kFusedOpType.c_str(), "remove einsum node failed"), return FAILED);
  return SUCCESS;
}

REGISTER_PASS("EinsumPass", BUILT_IN_GRAPH_PASS, EinsumPass);
}  // namespace fe
