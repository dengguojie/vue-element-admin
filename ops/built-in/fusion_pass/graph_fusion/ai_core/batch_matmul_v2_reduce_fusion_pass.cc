/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
 * \file batch_matmul_v2_reduce_fusion_pass.cc
 * \brief batch_matmul_v2_reduce_fusion_pass
 */

#include "batch_matmul_v2_reduce_fusion_pass.h"

#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "anchor_util.h"
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

namespace fe {
static const string PATTERN_BATCHMATMULV2 = "BatchMatMulV2";
static const string BATCHMATMULV2 = "BatchMatMulV2";
static const string REDUCESUMD = "ReduceSumD";
static const string CAST = "Cast";
static const string TRANSPOSED = "TransposeD";
static const string RESHAPE = "Reshape";
static const uint32_t kLeftShapeDim = 3;
static const uint32_t kRightShapeDim = 3;

vector<FusionPattern *> BatchMatMulV2ReduceFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = new (std::nothrow) FusionPattern("BatchMatMulV2ReduceFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object fail."), return patterns);
  pattern->AddOpDesc(PATTERN_BATCHMATMULV2, {BATCHMATMULV2}).SetOutput(PATTERN_BATCHMATMULV2);
  patterns.push_back(pattern);
  return patterns;
}

// BatchMatMulV2 --> ReduceSumD --> Output
bool BatchMatMulV2ReduceFusionPass::IsMatchScenario1(const ge::NodePtr &fused_node) const {
  // check input link relation
  FUSION_PASS_CHECK(fused_node->GetInDataNodes().size() != 2,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Input node of fused_node size is [%lu], which not equal to 2.",
                            fused_node->GetInDataNodes().size()),
                    return false);

  auto out_anchor = fused_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(out_anchor == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "out_anchor is null."),
                    return false);
  auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  if (peer_in_anchors.size() != 1) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "fused_node peer_in_anchors.size() is not 1.");
    return false;
  }

  // check if next_node is ReduceSumD
  auto next_node = peer_in_anchors.at(0)->GetOwnerNode();
  FUSION_PASS_CHECK(next_node == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "next_node is null."),
                    return false);
  return (next_node->GetType() == REDUCESUMD);
}

// BatchMatMulV2 --> Cast32 --> ReduceSumD --> Output
bool BatchMatMulV2ReduceFusionPass::IsMatchScenario2(const ge::NodePtr &fused_node) const {
  // check input link relation
  FUSION_PASS_CHECK(fused_node->GetInDataNodes().size() != 2,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Input node of fused_node size is [%lu], which not equal to 2.",
                            fused_node->GetInDataNodes().size()),
                    return false);

  auto out_anchor = fused_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(out_anchor == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "out_anchor is null."),
                    return false);
  auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  if (peer_in_anchors.size() != 1) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "fused_node peer_in_anchors.size() is not 1.");
    return false;
  }

  // check if next_node is Cast32
  auto next_node = peer_in_anchors.at(0)->GetOwnerNode();
  FUSION_PASS_CHECK(next_node == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "next_node is null."),
                    return false);
  if (next_node->GetType() != CAST) {
    return false;
  }
  ge::DataType batchmatmul_output_dtype = fused_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
  ge::DataType cast_output_dtype = next_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
  if (batchmatmul_output_dtype != ge::DT_FLOAT16 || cast_output_dtype != ge::DT_FLOAT) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "BatchMatMul output dtype is %u, Cast output dtype is %u", batchmatmul_output_dtype,
            cast_output_dtype);
    return false;
  }

  // check if next_next_node is ReduceSumD
  out_anchor = next_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(out_anchor == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "next_node out_anchor is null."), return false);
  peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  if (peer_in_anchors.size() != 1) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "next_node peer_in_anchors.size() is not 1.");
    return false;
  }
  auto next_next_node = peer_in_anchors.at(0)->GetOwnerNode();
  FUSION_PASS_CHECK(next_next_node == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "next_next_node is null."),
                    return false);
  return (next_next_node->GetType() == REDUCESUMD);
}

bool BatchMatMulV2ReduceFusionPass::CheckProduct(const std::vector<int64_t> &shape, std::size_t len) const {
  if (len > shape.size()) {
    return false;
  }
  int64_t product = 1;
  for (std::size_t i = 0; i < len; i++) {
    if (shape[i] > 0) {
      if (product > (INT64_MAX / shape[i])) {
        return false;
      } else {
        product *= shape[i];
      }
    }
  }
  return true;
}

bool BatchMatMulV2ReduceFusionPass::CheckNeedChange(const ge::NodePtr &fused_node, const vector<int64_t> &shape_x,
                                                    const vector<int64_t> &shape_y,
                                                    const vector<int64_t> &product_shape_x,
                                                    const vector<int64_t> &product_shape_y) const {
  auto x_dims = shape_x.size();
  auto y_dims = shape_y.size();
  auto product_x_dims = product_shape_x.size();
  auto product_y_dims = product_shape_y.size();
  if (x_dims == 0 || y_dims == 0 || product_x_dims == 0 || product_y_dims == 0) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "shape_x or shape_y or product_x_dims or product_y_dims is empty.");
    return false;
  }

  if (x_dims == kLeftShapeDim && y_dims == kRightShapeDim && shape_x[0] > 1) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "input shape x1=[%ld, %ld, %ld], x2=[%ld, %ld, %ld].", shape_x[0], shape_x[1],
            shape_x[2], shape_y[0], shape_y[1], shape_y[2]);
    auto op_desc = fused_node->GetOpDesc();
    FUSION_PASS_CHECK(op_desc == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "op_desc is null."),
                      return false);
    // check if dynamic shape
    auto x0_desc = op_desc->MutableInputDesc(0);
    auto x1_desc = op_desc->MutableInputDesc(1);
    if (x0_desc->MutableShape().IsUnknownShape() || x1_desc->MutableShape().IsUnknownShape()) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "not support dynamic shape.");
      return false;
    }
    // check if can fusion
    bool need_change = CheckProduct(product_shape_x, 2) && CheckProduct(product_shape_y, 2) &&
                       (IsMatchScenario2(fused_node) || IsMatchScenario1(fused_node));
    if (need_change) {
      return true;
    }
  }
  return false;
}

Status BatchMatMulV2ReduceFusionPass::CreateReshapeNode(ge::ComputeGraph &graph, ge::NodePtr &fused_node,
                                                        const ge::OutDataAnchorPtr &out_anchor,
                                                        const vector<int64_t> &shape, ge::NodePtr &shape_node) const {
  auto previous_node = out_anchor->GetOwnerNode();
  int idx = out_anchor->GetIdx();
  auto previous_node_desc = previous_node->GetOpDesc()->GetOutputDesc(idx);
  ge::GeTensorDesc next_in_desc = previous_node_desc.Clone();
  next_in_desc.SetShape(ge::GeShape(shape));
  next_in_desc.SetOriginShape(ge::GeShape(shape));

  ge::OpDescPtr reshape_desc;
  FUSION_PASS_MAKE_SHARED((reshape_desc = std::make_shared<ge::OpDesc>(
                               previous_node->GetName() + "_" + fused_node->GetName() + "_cann" + "/Reshape", RESHAPE)),
                          return FAILED);
  FUSION_PASS_CHECK(reshape_desc->AddInputDesc("x", previous_node_desc) != GRAPH_SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add input desc x to reshape."), return FAILED);
  FUSION_PASS_CHECK(reshape_desc->AddOutputDesc("y", next_in_desc) != GRAPH_SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add input desc y to reshape."), return FAILED);
  ge::AttrUtils::SetListInt(reshape_desc, "shape", shape);

  shape_node = graph.AddNode(reshape_desc);
  FUSION_PASS_CHECK(shape_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add reshape to graph."),
                    return FAILED);
  return SUCCESS;
}

Status BatchMatMulV2ReduceFusionPass::InsertReshapeNode(ge::ComputeGraph &graph, ge::NodePtr &fused_node, int32_t index,
                                                        const vector<int64_t> &new_shape) const {
  ge::NodePtr reshape_node = nullptr;
  auto in_anchor = fused_node->GetInDataAnchor(index);
  FUSION_PASS_CHECK(in_anchor == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "in_anchor is null."),
                    return FAILED);
  auto out_anchor = in_anchor->GetPeerOutAnchor();
  FUSION_PASS_CHECK(out_anchor == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "out_anchor is null."),
                    return FAILED);
  FUSION_PASS_CHECK(CreateReshapeNode(graph, fused_node, out_anchor, new_shape, reshape_node) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "failed to create reshape node."), return FAILED);
  auto input_desc_ptr = GetCurrNodeMutableInputDesc(fused_node, index);
  FUSION_PASS_CHECK(input_desc_ptr == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_desc_ptr is null."),
                    return FAILED);
  input_desc_ptr->SetShape(ge::GeShape(new_shape));
  input_desc_ptr->SetOriginShape(ge::GeShape(new_shape));

  auto ret = ge::GraphUtils::InsertNodeBetweenDataAnchors(out_anchor, in_anchor, reshape_node);
  FUSION_PASS_CHECK(ret != ge::GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "InsertNodeBetweenDataAnchors failed."),
                    return FAILED);
  return SUCCESS;
}

Status BatchMatMulV2ReduceFusionPass::LinkEdge(ge::NodePtr &fused_node, ge::NodePtr &tgt_node) const {
  auto batchmatmul_in_control_anchor = fused_node->GetInControlAnchor();
  FUSION_PASS_CHECK(batchmatmul_in_control_anchor == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "batchmatmul_in_control_anchor is null."),
                    return FAILED);
  auto batchmatmul_out_data_anchor = fused_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(batchmatmul_out_data_anchor == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "batchmatmul_out_data_anchor is null."),
                    return FAILED);

  // deal with in control anchor
  ge::InControlAnchorPtr inControlAnchor = tgt_node->GetInControlAnchor();
  if (inControlAnchor != nullptr) {
    for (ge::OutControlAnchorPtr srcAnchor : inControlAnchor->GetPeerOutControlAnchors()) {
      if (ge::GraphUtils::RemoveEdge(srcAnchor, inControlAnchor) != ge::GRAPH_SUCCESS ||
          ge::GraphUtils::AddEdge(srcAnchor, batchmatmul_in_control_anchor) != ge::GRAPH_SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "replace in control anchor failed.");
        return FAILED;
      }
    }
  }
  // deal with out data anchor
  for (ge::OutDataAnchorPtr &anchor : tgt_node->GetAllOutDataAnchors()) {
    FUSION_PASS_CHECK(anchor == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "cast_out_anchor is null."), return FAILED);
    for (ge::InDataAnchorPtr &dst_anchor : anchor->GetPeerInDataAnchors()) {
      if (ge::GraphUtils::RemoveEdge(anchor, dst_anchor) != ge::GRAPH_SUCCESS ||
          ge::GraphUtils::AddEdge(batchmatmul_out_data_anchor, dst_anchor) != ge::GRAPH_SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "replace out data anchor failed.");
        return FAILED;
      }
    }
  }
  // deal with out control anchor
  auto tgt_out_control_anchor = tgt_node->GetOutControlAnchor();
  if (tgt_out_control_anchor != nullptr) {
    for (ge::InControlAnchorPtr &dst_anchor : tgt_out_control_anchor->GetPeerInControlAnchors()) {
      if (ge::GraphUtils::RemoveEdge(tgt_out_control_anchor, dst_anchor) != ge::GRAPH_SUCCESS ||
          ge::GraphUtils::AddEdge(fused_node->GetOutControlAnchor(), dst_anchor) != ge::GRAPH_SUCCESS) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "replace out control anchor failed.");
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status BatchMatMulV2ReduceFusionPass::InsertTransposeDNode(
    ge::ComputeGraph &graph, ge::NodePtr &fused_node,
    std::tuple<int, std::vector<int64_t>, std::vector<int32_t>> &param, ge::NodePtr &transposedNode) const {
  int index = std::get<0>(param);
  std::vector<int64_t> new_shape = std::get<1>(param);
  std::vector<int32_t> perm = std::get<2>(param);

  GeTensorDesc inputdesc = fused_node->GetOpDesc()->GetInputDesc(index);
  auto in_anchor = fused_node->GetInDataAnchor(index);
  FUSION_PASS_CHECK(in_anchor == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "in_anchor is null."),
                    return FAILED);
  auto out_anchor = in_anchor->GetPeerOutAnchor();
  FUSION_PASS_CHECK(out_anchor == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "out_anchor is null."),
                    return FAILED);
  auto previous_node = out_anchor->GetOwnerNode();
  ge::OpDescPtr transpose_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (transpose_desc = std::make_shared<ge::OpDesc>(
           previous_node->GetName() + "_" + fused_node->GetName() + "_cann" + "/Transpose", TRANSPOSED)),
      return FAILED);
  transpose_desc->AddInputDesc("x", inputdesc);
  inputdesc.SetShape(GeShape(new_shape));
  inputdesc.SetOriginShape(GeShape(new_shape));
  transpose_desc->AddOutputDesc("y", inputdesc);
  AttrUtils::SetListInt(transpose_desc, "perm", perm);
  transposedNode = graph.AddNode(transpose_desc);
  FUSION_PASS_CHECK(transposedNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add transposed node."),
                    return FAILED);
  return SUCCESS;
}

Status BatchMatMulV2ReduceFusionPass::DealWithInputWithKOne(
    ge::ComputeGraph &graph, ge::NodePtr &fused_node,
    std::tuple<int, std::vector<int64_t>, int, std::vector<int64_t>, std::vector<bool>> &param) const {
  int index_x1 = std::get<0>(param);
  std::vector<int64_t> new_x1_out_shape = std::get<1>(param);
  int index_x2 = std::get<2>(param);
  std::vector<int64_t> new_x2_out_shape = std::get<3>(param);

  // inser reshape node for x1, x2
  FUSION_PASS_CHECK(SUCCESS != InsertReshapeNode(graph, fused_node, index_x1, new_x1_out_shape),
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "InsertReshapeNode x1 failed!"), return FAILED);
  FUSION_PASS_CHECK(SUCCESS != InsertReshapeNode(graph, fused_node, index_x2, new_x2_out_shape),
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "InsertReshapeNode x2 failed!"), return FAILED);
  return SUCCESS;
}

Status BatchMatMulV2ReduceFusionPass::DoFusionWithKOne(ge::ComputeGraph &graph, ge::NodePtr &fused_node,
                                                       const vector<int64_t> &new_x1_out_shape,
                                                       const vector<int64_t> &new_x2_out_shape,
                                                       const vector<bool> &trans) const {
  auto param_tuple = std::make_tuple(0, new_x1_out_shape, 1, new_x2_out_shape, trans);
  FUSION_PASS_CHECK(SUCCESS != DealWithInputWithKOne(graph, fused_node, param_tuple),
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "DealWithInputWithKOne failed!"), return FAILED);

  auto op_desc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(op_desc == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "op_desc is null."),
                    return FAILED);
  if (!trans[0]) {
    AttrUtils::SetBool(op_desc, "adj_x1", true);
  }
  if (trans[1]) {
    AttrUtils::SetBool(op_desc, "adj_x2", false);
  }

  auto out_anchor = fused_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(out_anchor == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "out_anchor is null."),
                    return FAILED);
  auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  if (peer_in_anchors.size() != 1) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "fused_node peer_in_anchors.size() is not 1.");
    return FAILED;
  }

  // check if next_node is Cast32
  auto next_node = peer_in_anchors.at(0)->GetOwnerNode();
  FUSION_PASS_CHECK(next_node == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "next_node is null."),
                    return FAILED);
  if (next_node->GetType() == CAST) {
    ge::DataType batchmatmul_output_dtype = fused_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
    ge::DataType cast_output_dtype = next_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
    if (batchmatmul_output_dtype != ge::DT_FLOAT16 || cast_output_dtype != ge::DT_FLOAT) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "BatchMatMul output dtype is %u, Cast output dtype is %u",
              batchmatmul_output_dtype, cast_output_dtype);
      return FAILED;
    }

    // check if next_next_node is ReduceSumD
    out_anchor = next_node->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(out_anchor == nullptr,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "next_node out_anchor is null."), return FAILED);
    peer_in_anchors = out_anchor->GetPeerInDataAnchors();
    if (peer_in_anchors.size() != 1) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "next_node peer_in_anchors.size() is not 1.");
      return FAILED;
    }
    auto next_next_node = peer_in_anchors.at(0)->GetOwnerNode();
    FUSION_PASS_CHECK(next_next_node == nullptr,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "next_next_node is null."), return FAILED);
    // bmm + cast32 + reducesumd : fused_node -> next_node -> next_next_node
    if (next_next_node->GetType() == REDUCESUMD) {
      FUSION_PASS_CHECK(LinkEdge(fused_node, next_node) == FAILED,
                        OP_LOGD(FUSED_OP_TYPE.c_str(), "link next_node edge Failed."), return FAILED);
      FUSION_PASS_CHECK(LinkEdge(fused_node, next_next_node) == FAILED,
                        OP_LOGD(FUSED_OP_TYPE.c_str(), "link next_next_node edge Failed."), return FAILED);
      FUSION_PASS_CHECK(graph.RemoveNode(next_node) == ge::GRAPH_FAILED,
                        OP_LOGD(FUSED_OP_TYPE.c_str(), "remove cast32 node failed"), return FAILED);
      FUSION_PASS_CHECK(graph.RemoveNode(next_next_node) == ge::GRAPH_FAILED,
                        OP_LOGD(FUSED_OP_TYPE.c_str(), "remove reducesumd node failed"), return FAILED);

      // set batchmatmul output dtype fp32
      auto batchmatmul_output_desc = fused_node->GetOpDesc()->MutableOutputDesc(0);
      batchmatmul_output_desc->SetDataType(ge::DT_FLOAT);
      batchmatmul_output_desc->SetOriginDataType(ge::DT_FLOAT);
    }
  } else if (next_node->GetType() == REDUCESUMD) {
    // bmm + reducesumd : fused_node -> next_node
    FUSION_PASS_CHECK(LinkEdge(fused_node, next_node) == FAILED,
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "link next_node edge Failed."), return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(next_node) == ge::GRAPH_FAILED,
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "remove reducesumd node failed"), return FAILED);
  }
  return SUCCESS;
}

Status BatchMatMulV2ReduceFusionPass::DealWithInputWithKNotOne(
    ge::ComputeGraph &graph, ge::NodePtr &fused_node,
    std::tuple<int, std::vector<int64_t>, int, std::vector<int64_t>, std::vector<bool>> &param) const {
  int index_x1 = std::get<0>(param);
  std::vector<int64_t> new_x1_out_shape = std::get<1>(param);
  int index_x2 = std::get<2>(param);
  std::vector<int64_t> new_x2_out_shape = std::get<3>(param);
  std::vector<bool> trans = std::get<4>(param);

  // inser TransposeD or reshape node for x1, x2
  auto input1desc = GetCurrNodeInputDesc(fused_node, index_x1);
  auto input2desc = GetCurrNodeInputDesc(fused_node, index_x2);
  FUSION_PASS_CHECK(input1desc == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputDesc0 is null"),
                    return FAILED);
  FUSION_PASS_CHECK(input2desc == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputDesc1 is null"),
                    return FAILED);
  auto x1_shape = input1desc->GetOriginShape().GetDims();
  auto x2_shape = input2desc->GetOriginShape().GetDims();
  ge::NodePtr transposedNode = nullptr;
  std::shared_ptr<ge::OpDesc> reshape_desc = nullptr;
  ge::NodePtr reshape_node = nullptr;
  ge::GeTensorDescPtr output0desc = nullptr;
  std::vector<int64_t> transposed_output_shape;
  std::vector<int64_t> output0_shape;
  std::vector<int64_t> new_out_shape;
  std::vector<int32_t> perm({1, 0, 2});
  auto x1_anchor_peer_anchor = fused_node->GetInDataAnchor(index_x1)->GetPeerOutAnchor();
  auto x1_anchor_peer_node = x1_anchor_peer_anchor->GetOwnerNode();
  auto x2_anchor_peer_anchor = fused_node->GetInDataAnchor(index_x2)->GetPeerOutAnchor();
  auto x2_anchor_peer_node = x2_anchor_peer_anchor->GetOwnerNode();
  bool trans_a = trans[0];
  bool trans_b = trans[1];

  if (trans_a) {
    // Reshape
    FUSION_PASS_CHECK(SUCCESS != InsertReshapeNode(graph, fused_node, index_x1, new_x1_out_shape),
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "InsertReshapeNode x1 failed!"), return FAILED);
  } else {
    // TransposeD + Reshape; b,m,k --> m,b,k --> m,b*k
    transposed_output_shape.assign({x1_shape[1], x1_shape[0], x1_shape[2]});
    auto param_tuple = std::make_tuple(index_x1, transposed_output_shape, perm);
    FUSION_PASS_CHECK(SUCCESS != InsertTransposeDNode(graph, fused_node, param_tuple, transposedNode),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to insert Transposed node."), return FAILED);
    // create reshape op desc
    output0desc = transposedNode->GetOpDesc()->MutableOutputDesc(0);
    FUSION_PASS_CHECK(output0desc == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "output0desc is null"),
                      return FAILED);
    output0_shape = output0desc->GetOriginShape().GetDims();
    new_out_shape.assign({output0_shape[0], output0_shape[1] * output0_shape[2]});
    FUSION_PASS_MAKE_SHARED(reshape_desc = std::make_shared<ge::OpDesc>(
                                fused_node->GetName() + "_" + to_string(index_x1) + "_cann" + "/Reshape", RESHAPE),
                            return FAILED);
    GeTensorDesc x1_desc = fused_node->GetOpDesc()->GetInputDesc(index_x1);
    reshape_desc->AddInputDesc("x", *(output0desc));
    x1_desc.SetShape(GeShape(new_out_shape));
    x1_desc.SetOriginShape(GeShape(new_out_shape));
    reshape_desc->AddOutputDesc("y", x1_desc);
    reshape_node = graph.AddNode(reshape_desc);
    FUSION_PASS_CHECK(reshape_node == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reshape_node is null"),
                      return FAILED);

    // add edge
    FUSION_PASS_CHECK(GraphUtils::RemoveEdge(x1_anchor_peer_anchor, fused_node->GetInDataAnchor(index_x1)) == FAILED,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove fused_node input edge Failed."), return FAILED);
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x1_anchor_peer_anchor, transposedNode->GetInDataAnchor(0)),
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                            x1_anchor_peer_node->GetName().c_str(), transposedNode->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(transposedNode->GetOutDataAnchor(0), reshape_node->GetInDataAnchor(0)),
        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              transposedNode->GetName().c_str(), reshape_node->GetName().c_str()),
        return FAILED);

    fused_node->GetOpDesc()->AddInputDesc("x1", *(reshape_node->GetOpDesc()->MutableOutputDesc(0)));

    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(reshape_node->GetOutDataAnchor(0), fused_node->GetInDataAnchor(index_x1)),
        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              reshape_node->GetName().c_str(), fused_node->GetName().c_str()),
        return FAILED);
  }
  if (trans_b) {
    // TransposeD + Reshape; b,n,k --> n,b,k --> n,b*k
    transposed_output_shape.assign({x2_shape[1], x2_shape[0], x2_shape[2]});
    auto param_tuple = std::make_tuple(index_x2, transposed_output_shape, perm);
    FUSION_PASS_CHECK(SUCCESS != InsertTransposeDNode(graph, fused_node, param_tuple, transposedNode),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to insert Transposed node."), return FAILED);
    // create reshape op desc
    output0desc = transposedNode->GetOpDesc()->MutableOutputDesc(0);
    FUSION_PASS_CHECK(output0desc == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "output0desc is null"),
                      return FAILED);
    output0_shape = output0desc->GetOriginShape().GetDims();
    new_out_shape.assign({output0_shape[0], output0_shape[1] * output0_shape[2]});

    FUSION_PASS_MAKE_SHARED(reshape_desc = std::make_shared<ge::OpDesc>(
                                fused_node->GetName() + "_" + to_string(index_x2) + "_cann" + "/Reshape", RESHAPE),
                            return FAILED);
    GeTensorDesc x2_desc = fused_node->GetOpDesc()->GetInputDesc(index_x2);
    reshape_desc->AddInputDesc("x", *(output0desc));
    x2_desc.SetShape(GeShape(new_out_shape));
    x2_desc.SetOriginShape(GeShape(new_out_shape));
    reshape_desc->AddOutputDesc("y", x2_desc);
    reshape_node = graph.AddNode(reshape_desc);
    FUSION_PASS_CHECK(reshape_node == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reshape_node is null"),
                      return FAILED);

    // add edge
    FUSION_PASS_CHECK(GraphUtils::RemoveEdge(x2_anchor_peer_anchor, fused_node->GetInDataAnchor(index_x2)) == FAILED,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove fused_node input edge Failed."), return FAILED);
    FUSION_PASS_CHECK(SUCCESS != GraphUtils::AddEdge(x2_anchor_peer_anchor, transposedNode->GetInDataAnchor(0)),
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                                            x2_anchor_peer_node->GetName().c_str(), transposedNode->GetName().c_str()),
                      return FAILED);
    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(transposedNode->GetOutDataAnchor(0), reshape_node->GetInDataAnchor(0)),
        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              transposedNode->GetName().c_str(), reshape_node->GetName().c_str()),
        return FAILED);

    fused_node->GetOpDesc()->AddInputDesc("x2", *(reshape_node->GetOpDesc()->MutableOutputDesc(0)));

    FUSION_PASS_CHECK(
        SUCCESS != GraphUtils::AddEdge(reshape_node->GetOutDataAnchor(0), fused_node->GetInDataAnchor(index_x2)),
        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              reshape_node->GetName().c_str(), fused_node->GetName().c_str()),
        return FAILED);
  } else {
    // Reshape
    FUSION_PASS_CHECK(SUCCESS != InsertReshapeNode(graph, fused_node, index_x2, new_x2_out_shape),
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "InsertReshapeNode x2 failed!"), return FAILED);
  }
  return SUCCESS;
}

Status BatchMatMulV2ReduceFusionPass::DoFusionWithKNotOne(ge::ComputeGraph &graph, ge::NodePtr &fused_node,
                                                          const vector<int64_t> &new_x1_out_shape,
                                                          const vector<int64_t> &new_x2_out_shape,
                                                          const vector<bool> &trans) const {
  auto param_tuple = std::make_tuple(0, new_x1_out_shape, 1, new_x2_out_shape, trans);
  FUSION_PASS_CHECK(SUCCESS != DealWithInputWithKNotOne(graph, fused_node, param_tuple),
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "DealWithInputWithKNotOne failed!"), return FAILED);

  // check if next_node is Cast32
  auto out_anchor = fused_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(out_anchor == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "out_anchor is null."),
                    return FAILED);
  auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  if (peer_in_anchors.size() != 1) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "fused_node peer_in_anchors.size() is not 1.");
    return FAILED;
  }
  auto next_node = peer_in_anchors.at(0)->GetOwnerNode();
  FUSION_PASS_CHECK(next_node == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "next_node is null."),
                    return FAILED);
  if (next_node->GetType() == CAST) {
    ge::DataType batchmatmul_output_dtype = fused_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
    ge::DataType cast_output_dtype = next_node->GetOpDesc()->GetOutputDesc(0).GetDataType();
    if (batchmatmul_output_dtype != ge::DT_FLOAT16 || cast_output_dtype != ge::DT_FLOAT) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "BatchMatMul output dtype is %u, Cast output dtype is %u",
              batchmatmul_output_dtype, cast_output_dtype);
      return FAILED;
    }

    // check if next_next_node is ReduceSumD
    out_anchor = next_node->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(out_anchor == nullptr,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "next_node out_anchor is null."), return FAILED);
    peer_in_anchors = out_anchor->GetPeerInDataAnchors();
    if (peer_in_anchors.size() != 1) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "next_node peer_in_anchors.size() is not 1.");
      return FAILED;
    }
    auto next_next_node = peer_in_anchors.at(0)->GetOwnerNode();
    FUSION_PASS_CHECK(next_next_node == nullptr,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "next_next_node is null."), return FAILED);
    // bmm + cast32 + reducesumd : fused_node -> next_node -> next_next_node
    if (next_next_node->GetType() == REDUCESUMD) {
      FUSION_PASS_CHECK(LinkEdge(fused_node, next_node) == FAILED,
                        OP_LOGD(FUSED_OP_TYPE.c_str(), "link next_node edge Failed."), return FAILED);
      FUSION_PASS_CHECK(LinkEdge(fused_node, next_next_node) == FAILED,
                        OP_LOGD(FUSED_OP_TYPE.c_str(), "link edge Failed."), return FAILED);
      FUSION_PASS_CHECK(graph.RemoveNode(next_node) == ge::GRAPH_FAILED,
                        OP_LOGD(FUSED_OP_TYPE.c_str(), "cast32 node remove failed"), return FAILED);
      FUSION_PASS_CHECK(graph.RemoveNode(next_next_node) == ge::GRAPH_FAILED,
                        OP_LOGD(FUSED_OP_TYPE.c_str(), "reducesumd node remove failed"), return FAILED);

      // set batchmatmul output dtype fp32
      auto batchmatmul_output_desc = fused_node->GetOpDesc()->MutableOutputDesc(0);
      batchmatmul_output_desc->SetDataType(ge::DT_FLOAT);
      batchmatmul_output_desc->SetOriginDataType(ge::DT_FLOAT);
    }
  } else if (next_node->GetType() == REDUCESUMD) {  // bmm + reducesumd : fused_node -> next_node
    FUSION_PASS_CHECK(LinkEdge(fused_node, next_node) == FAILED,
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "link next_node edge Failed."), return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(next_node) == ge::GRAPH_FAILED,
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "reducesumd node remove failed"), return FAILED);
  }
  return SUCCESS;
}

Status BatchMatMulV2ReduceFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                             vector<ge::NodePtr> & /* fuion_nodes */) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter BatchMatMulV2ReduceFusionPass.");

  // common vars
  ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_BATCHMATMULV2, mapping);
  FUSION_PASS_CHECK(fused_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Fuse node is null, fusion failed."),
                    return FAILED);

  auto input0desc = GetCurrNodeInputDesc(fused_node, 0);
  auto input1desc = GetCurrNodeInputDesc(fused_node, 1);
  FUSION_PASS_CHECK(input0desc == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputDesc0 is null"),
                    return FAILED);
  FUSION_PASS_CHECK(input1desc == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputDesc1 is null"),
                    return FAILED);
  auto x1_shape = input0desc->GetOriginShape().GetDims();
  auto x2_shape = input1desc->GetOriginShape().GetDims();
  bool trans_a = false;
  bool trans_b = false;
  auto op_desc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(op_desc == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "op_desc is null."),
                    return FAILED);
  FUSION_PASS_CHECK(!AttrUtils::GetBool(op_desc, "adj_x1", trans_a),
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GetBool adj_x1 failed!"), return FAILED);
  FUSION_PASS_CHECK(!AttrUtils::GetBool(op_desc, "adj_x2", trans_b),
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GetBool adj_x2 failed!"), return FAILED);

  std::vector<int64_t> product_x1_shape;
  std::vector<int64_t> product_x2_shape;
  std::vector<int64_t> new_x1_out_shape;
  std::vector<int64_t> new_x2_out_shape;
  std::vector<int64_t> new_out_shape;
  if (trans_a) {
    // b,k,m trans_a=true --> b*k,m
    product_x1_shape.assign({x1_shape[0], x1_shape[1]});
    new_x1_out_shape.assign({x1_shape[0] * x1_shape[1], x1_shape[2]});
    new_out_shape.push_back(x1_shape[2]);
  } else {
    // b,m,k trans_a=false --> b*k,m
    product_x1_shape.assign({x1_shape[0], x1_shape[2]});
    new_x1_out_shape.assign({x1_shape[0], x1_shape[2] * x1_shape[1]});
    new_out_shape.push_back(x1_shape[1]);
  }
  if (trans_b) {
    // b,n,k trans_b=true --> b*k,n
    product_x2_shape.assign({x2_shape[0], x2_shape[2]});
    new_x2_out_shape.assign({x2_shape[0], x2_shape[2] * x2_shape[1]});
    new_out_shape.push_back(x2_shape[1]);
  } else {
    // b,k,n trans_b=false --> b*k,n
    product_x2_shape.assign({x2_shape[0], x2_shape[1]});
    new_x2_out_shape.assign({x2_shape[0] * x2_shape[1], x2_shape[2]});
    new_out_shape.push_back(x2_shape[2]);
  }

  if (!CheckNeedChange(fused_node, x1_shape, x2_shape, product_x1_shape, product_x2_shape)) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "The graph doesn't need to be changed.");
    return NOT_CHANGED;
  }

  if (x1_shape.size() == kLeftShapeDim && x2_shape.size() == kRightShapeDim && x1_shape[0] > 1) {
    std::vector<bool> trans = {trans_a, trans_b};
    auto k_dim = (trans_a == true) ? x1_shape[1] : x1_shape[2];
    if (k_dim == 1) {
      FUSION_PASS_CHECK(SUCCESS != DoFusionWithKOne(graph, fused_node, new_x1_out_shape, new_x2_out_shape, trans),
                        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "DoFusionWithKOne failed!"), return FAILED);
    } else {
      FUSION_PASS_CHECK(SUCCESS != DoFusionWithKNotOne(graph, fused_node, new_x1_out_shape, new_x2_out_shape, trans),
                        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "DoFusionWithKNotOne failed!"), return FAILED);
    }

    // flush the batmatmul output shape
    auto out_desc_ptr = fused_node->GetOpDesc()->MutableOutputDesc(0);
    out_desc_ptr->SetShape(ge::GeShape(new_out_shape));
    out_desc_ptr->SetOriginShape(ge::GeShape(new_out_shape));

    OP_LOGD(FUSED_OP_TYPE.c_str(), "BatchMatMulV2ReduceFusionPass %u*%u scenario success.", kLeftShapeDim,
            kRightShapeDim);
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Leave BatchMatMulV2ReduceFusionPass.");
  return SUCCESS;
}

REGISTER_PASS("BatchMatMulV2ReduceFusionPass", BUILT_IN_GRAPH_PASS, BatchMatMulV2ReduceFusionPass);
} // namespace fe
