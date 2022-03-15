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
 * \file batch_matmul_v2_reshape_fusion_pass.cc
 * \brief the pass will effect if the input shape of batch_matmul_v2 is 1D
 */

#include "batch_matmul_v2_reshape_fusion_pass.h"

#include "anchor_util.h"
#include "graph/utils/graph_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
static const string PATTERN_BATCHMATMULV2 = "BatchMatMulV2";
static const string BATCHMATMULV2 = "BatchMatMulV2";
static const uint32_t kLeftShapeDim = 3;
static const uint32_t kRightShapeDim = 2;
static const uint32_t kBatchMinValue = 50;
static const uint32_t kMaxValue = 32;
static const int32_t kDynamicFlagUnrank = -2;
static const uint32_t kBigBatchBatchDim = 4096;
static const uint32_t kBigBatchMDim = 64;
vector<FusionPattern *> BatchMatMulV2ReshapeFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = new (std::nothrow) FusionPattern("BatchMatMulV2ReshapeFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object fail."), return patterns);
  pattern->AddOpDesc(PATTERN_BATCHMATMULV2, {BATCHMATMULV2}).SetOutput(PATTERN_BATCHMATMULV2);
  patterns.push_back(pattern);
  return patterns;
}

bool BatchMatMulV2ReshapeFusionPass::CheckProduct(const std::vector<int64_t> &shape, std::size_t len) {
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

// check next node exist or not
bool BatchMatMulV2ReshapeFusionPass::CheckNextNode(const ge::NodePtr &fused_node, ge::NodePtr &next_node) const {
  auto out_anchor = fused_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(out_anchor == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "out_anchor is null."),
                    return false);
  auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  if (peer_in_anchors.size() != 1) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "fused_node peer_in_anchors.size() is not 1.");
    return false;
  }
  next_node = peer_in_anchors.at(0)->GetOwnerNode();
  FUSION_PASS_CHECK(next_node == nullptr, OP_LOGD(FUSED_OP_TYPE.c_str(), "next_node is null."), return false);
  return true;
}

// BatchMatMulV2 --> Add --> Output
bool BatchMatMulV2ReshapeFusionPass::IsMatchScenario1(const ge::NodePtr &fused_node) const {
  ge::NodePtr next_node = nullptr;
  FUSION_PASS_CHECK(!CheckNextNode(fused_node, next_node),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "do not match elemwise fusion scenario1!"), return false);
  auto out_node_num = next_node->GetOutDataNodesSize();
  if (out_node_num != 1) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "next_node out_node_num is not 1.");
    return false;
  }
  if (next_node->GetType() == "Add") {
    // the add operator specifies two inputs
    auto input1_dims = next_node->GetOpDesc()->GetInputDesc(0).GetOriginShape().GetDimNum();
    auto input2_dims = next_node->GetOpDesc()->GetInputDesc(1).GetOriginShape().GetDimNum();
    if (input1_dims == 1 || input2_dims == 1) {
      return true;
    }
  }

  return false;
}

// BatchMatMulV2 --> Add --> Add --> Output
bool BatchMatMulV2ReshapeFusionPass::IsMatchScenario2(const ge::NodePtr &fused_node) const {
  if (IsMatchScenario1(fused_node)) {
    ge::NodePtr next_node = nullptr;
    FUSION_PASS_CHECK(!CheckNextNode(fused_node, next_node),
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "do not match elemwise fusion scenario2!"), return false);
    auto peer_in_anchors = next_node->GetOutDataAnchor(0)->GetPeerInDataAnchors();
    if (peer_in_anchors.size() != 1) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "next_node peer_in_anchors.size() is not 1.");
      return false;
    }
    auto next_next_node = peer_in_anchors.at(0)->GetOwnerNode();
    FUSION_PASS_CHECK(next_next_node == nullptr,
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "next_next_node is null."), return FAILED);
    auto out_node_num = next_next_node->GetOutDataNodesSize();
    if (out_node_num != 1) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "next_next_node out_node_num is not 1.");
      return false;
    }
    if (next_next_node->GetType() == "Add") {
      auto idx = peer_in_anchors.at(0)->GetIdx();
      auto input_dims = next_next_node->GetOpDesc()->GetInputDesc(1 - idx).GetOriginShape().GetDimNum();
      if (input_dims > 1) {
        return true;
      }
    }
  }

  return false;
}

/*
 * BatchMatMulV2 --> Add --> Mul --> Sigmoid --> Mul --> Output
 *                    \__________________________/
 */
bool BatchMatMulV2ReshapeFusionPass::IsMatchScenario3(const ge::NodePtr &fused_node) const {
  ge::NodePtr next_node = nullptr;
  FUSION_PASS_CHECK(!CheckNextNode(fused_node, next_node),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "do not match elemwise fusion scenario3!"), return false);
  auto out_node_num = next_node->GetOutDataNodesSize();
  if (out_node_num != 2) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "next_node out_node_num is not 2.");
    return false;
  }
  auto peer_in_anchors = next_node->GetOutDataAnchor(0)->GetPeerInDataAnchors();
  auto next_next_node_0 = peer_in_anchors.at(0)->GetOwnerNode();
  auto next_next_node_1 = peer_in_anchors.at(1)->GetOwnerNode();
  FUSION_PASS_CHECK(next_next_node_0 == nullptr || next_next_node_1 == nullptr,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "node is null."), return false);
  FUSION_PASS_CHECK(next_next_node_0->GetType() != "Mul" || next_next_node_1->GetType() != "Mul",
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "The next node is not Mul."), return false);
  auto out_anchor = next_next_node_0->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(out_anchor == nullptr, OP_LOGD(FUSED_OP_TYPE.c_str(), "out_anchor is null."), return false);
  peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  FUSION_PASS_CHECK(peer_in_anchors.size() != 1, OP_LOGD(FUSED_OP_TYPE.c_str(), "peer_in_anchors.size() != 1."),
                    return false);
  auto sigmoid_node = peer_in_anchors.at(0)->GetOwnerNode();
  FUSION_PASS_CHECK(sigmoid_node == nullptr || sigmoid_node->GetType() != "Sigmoid",
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "The node is not Sigmoid."), return false);
  out_anchor = sigmoid_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(out_anchor == nullptr, OP_LOGD(FUSED_OP_TYPE.c_str(), "out_anchor is null."), return false);
  peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  FUSION_PASS_CHECK(peer_in_anchors.size() != 1, OP_LOGD(FUSED_OP_TYPE.c_str(), "peer_in_anchors.size() != 1."),
                    return false);
  auto mul_node = peer_in_anchors.at(0)->GetOwnerNode();
  FUSION_PASS_CHECK(mul_node == nullptr || mul_node->GetName() != next_next_node_1->GetName(),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "The end node must be same Mul node."), return false);
  out_anchor = mul_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(out_anchor == nullptr, OP_LOGD(FUSED_OP_TYPE.c_str(), "out_anchor is null."), return false);
  peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  FUSION_PASS_CHECK(peer_in_anchors.size() != 1, OP_LOGD(FUSED_OP_TYPE.c_str(), "peer_in_anchors.size() != 1."),
                    return false);
  return true;
}

bool BatchMatMulV2ReshapeFusionPass::CheckValidDim(const int64_t &x_dims, const int64_t &y_dims) const {
  if (x_dims == 0 || y_dims == 0) {
    return false;
  }
  return true;
}

bool BatchMatMulV2ReshapeFusionPass::IsElemwiseFusionScenario(const ge::NodePtr &fused_node,
                                                              const vector<int64_t> &shape_x,
                                                              const bool &trans_a) const {
  bool is_elemwise_fusion =
      IsMatchScenario1(fused_node) || IsMatchScenario2(fused_node) || IsMatchScenario3(fused_node);
  is_elemwise_fusion = is_elemwise_fusion && shape_x[0] >= kBatchMinValue && shape_x[1] <= kMaxValue && !trans_a;
  return is_elemwise_fusion;
}

bool BatchMatMulV2ReshapeFusionPass::IsBigBatchFusionScenario(const ge::NodePtr &fused_node,
                                                              const vector<int64_t> &shape_x,
                                                              const bool &trans_a) const {
  bool is_big_batch_fusion = shape_x[0] >= kBigBatchBatchDim && shape_x[1] <= kBigBatchMDim && !trans_a;
  return is_big_batch_fusion;
}

bool BatchMatMulV2ReshapeFusionPass::CheckNeedChange(const ge::NodePtr &fused_node, const vector<int64_t> &shape_x,
                                                     const vector<int64_t> &shape_y, bool &is_elemwise_fusion,
                                                     bool &is_big_batch_fusion) {
  auto x_dims = shape_x.size();
  auto y_dims = shape_y.size();
  if (!CheckValidDim(x_dims, y_dims)) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "shape_x or shape_y is empty.");
    return false;
  }

  // step1: compatible with the original scenario
  if (x_dims == 1 || y_dims == 1) {
    if (shape_x[0] == kDynamicFlagUnrank || shape_y[0] == kDynamicFlagUnrank) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "shape_x[0] or shape_y[0] is -2.");
      return false;
    }
    return true;
  }

  // step2: handling the new elemwise fusion scenario and the big batch scenario
  if (x_dims == kLeftShapeDim && y_dims == kRightShapeDim) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "input shape x1=[%ld, %ld, %ld], x2=[%ld, %ld].", shape_x[0], shape_x[1], shape_x[2],
            shape_y[0], shape_y[1]);
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
    bool trans_a = false;
    bool trans_b = false;
    FUSION_PASS_CHECK(!AttrUtils::GetBool(op_desc, "adj_x1", trans_a),
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GetBool adj_x1 failed!"), return false);
    FUSION_PASS_CHECK(!AttrUtils::GetBool(op_desc, "adj_x2", trans_b),
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GetBool adj_x2 failed!"), return false);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "batchmatmul trans_a=%d, trans_b=%d.", trans_a, trans_b);
    is_elemwise_fusion =
        IsElemwiseFusionScenario(fused_node, shape_x, trans_a) && CheckProduct(x0_desc->GetOriginShape().GetDims(), 2);
    if (is_elemwise_fusion) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "node name [%s], match elemwise fusion senario, return need change!",
              fused_node->GetName().c_str());
      return true;
    }
    is_big_batch_fusion =
        IsBigBatchFusionScenario(fused_node, shape_x, trans_a) && CheckProduct(x0_desc->GetOriginShape().GetDims(), 2);
    if (is_big_batch_fusion) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "node name [%s], match big batch senario, return need change!",
              fused_node->GetName().c_str());
      return true;
    }
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "node name [%s], do not match elemwise fusion or big batch senario, return no change!",
          fused_node->GetName().c_str());
  return false;
}

Status BatchMatMulV2ReshapeFusionPass::InputInsertReshapeNode(ge::ComputeGraph &graph, const ge::NodePtr &fused_node,
                                                              int32_t index, const vector<int64_t> &new_shape) {
  OP_LOGD(FUSED_OP_TYPE.c_str(),
          "Begin to insert reshape node for input of node name [%s], type [%s], at in data anchor index [%d]!",
          fused_node->GetName().c_str(), fused_node->GetType().c_str(), index);
  ge::NodePtr x1_reshape_node = nullptr;
  auto in_anchor = fused_node->GetInDataAnchor(index);
  FUSION_PASS_CHECK(in_anchor == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "in_anchor is null."),
                    return FAILED);
  auto out_anchor = in_anchor->GetPeerOutAnchor();
  FUSION_PASS_CHECK(out_anchor == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "out_anchor is null."),
                    return FAILED);

  CreateReshapeNode(graph, fused_node, out_anchor, new_shape, x1_reshape_node);
  auto input_desc_ptr = GetCurrNodeMutableInputDesc(fused_node, index);
  FUSION_PASS_CHECK(input_desc_ptr == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_desc_ptr is null."),
                    return FAILED);
  input_desc_ptr->SetShape(ge::GeShape(new_shape));
  input_desc_ptr->SetOriginShape(ge::GeShape(new_shape));

  auto ret = ge::GraphUtils::InsertNodeBetweenDataAnchors(out_anchor, in_anchor, x1_reshape_node);
  FUSION_PASS_CHECK(ret != ge::GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "InsertNodeBetweenDataAnchors failed."),
                    return FAILED);

  return SUCCESS;
}

Status BatchMatMulV2ReshapeFusionPass::OutputInsertReshapeNode(ge::ComputeGraph &graph, const ge::NodePtr &fused_node,
                                                               int32_t index, const vector<int64_t> &out_shape) {
  OP_LOGD(FUSED_OP_TYPE.c_str(),
          "Begin to insert reshape node for output of node name [%s], type [%s], at out data anchor index [%d]!",
          fused_node->GetName().c_str(), fused_node->GetType().c_str(), index);
  ge::NodePtr out_reshape_node = nullptr;
  auto out_anchor = fused_node->GetOutDataAnchor(index);
  FUSION_PASS_CHECK(out_anchor == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "out_anchor is null."),
                    return FAILED);
  for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(peer_in_anchor == nullptr,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "peer_in_anchor is null."), return FAILED);

    auto next_node = peer_in_anchor->GetOwnerNode();
    FUSION_PASS_CHECK(next_node == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "next_node is null."),
                      return FAILED);
    CreateReshapeNode(graph, next_node, out_anchor, out_shape, out_reshape_node);
    int32_t idx = peer_in_anchor->GetIdx();
    OP_LOGD(FUSED_OP_TYPE.c_str(), "next node type=[%s], in anchor index=[%d].", next_node->GetType().c_str(), idx);

    auto input_desc_ptr = GetCurrNodeMutableInputDesc(next_node, idx);
    FUSION_PASS_CHECK(input_desc_ptr == nullptr,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_desc_ptr is null."), return FAILED);
    input_desc_ptr->SetShape(ge::GeShape(out_shape));
    input_desc_ptr->SetOriginShape(ge::GeShape(out_shape));
    auto ret = ge::GraphUtils::InsertNodeBetweenDataAnchors(out_anchor, peer_in_anchor, out_reshape_node);
    FUSION_PASS_CHECK(ret != ge::GRAPH_SUCCESS,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "InsertNodeBetweenDataAnchors failed."),
                      return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "insert reshape node after peer in anchor index [%d].", idx);
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "insert reshape node after node [%s] success.", fused_node->GetName().c_str());
  return SUCCESS;
}

Status BatchMatMulV2ReshapeFusionPass::UpdateOpDescByIndex(const ge::NodePtr &node, const vector<int64_t> &new_shape,
                                                           int32_t index) {
  auto input_desc_ptr = GetCurrNodeMutableInputDesc(node, index);
  FUSION_PASS_CHECK(input_desc_ptr == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_desc_ptr is null."),
                    return FAILED);
  int32_t dims = input_desc_ptr->GetOriginShape().GetDimNum();
  OP_LOGD(FUSED_OP_TYPE.c_str(), "index = %d, node input dims=%d.", index, dims);
  if (dims > 1) {
    input_desc_ptr->SetShape(ge::GeShape(new_shape));
    input_desc_ptr->SetOriginShape(ge::GeShape(new_shape));
  }

  return SUCCESS;
}

Status BatchMatMulV2ReshapeFusionPass::UpdateOpDesc(const ge::NodePtr &node, const vector<int64_t> &new_shape) {
  uint32_t input_num = node->GetAllInDataAnchorsSize();
  for (uint32_t i = 0; i < input_num; ++i) {
    FUSION_PASS_CHECK(SUCCESS != UpdateOpDescByIndex(node, new_shape, i),
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "UpdateOpDescByIndex failed!"), return FAILED);
  }

  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != node->InferShapeAndType(),
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "InferShapeAndType failed!"), return FAILED);

  return SUCCESS;
}

Status BatchMatMulV2ReshapeFusionPass::ConnectOneElemwise(ge::ComputeGraph &graph, const ge::NodePtr &next_node,
                                                          const vector<int64_t> &new_shape,
                                                          const vector<int64_t> &out_shape) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Do fusion on connect one elemwise branch");
  auto out_node = next_node;
  auto peer_in_anchors = next_node->GetOutDataAnchor(0)->GetPeerInDataAnchors();
  auto next_next_node = peer_in_anchors.at(0)->GetOwnerNode();
  FUSION_PASS_CHECK(next_next_node == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "next_next_node is null."),
                    return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "next next node type is %s", next_next_node->GetType().c_str());
  auto idx = peer_in_anchors.at(0)->GetIdx();
  if (next_next_node->GetType() == "Add") {
    out_node = next_next_node;
    FUSION_PASS_CHECK(SUCCESS != UpdateOpDesc(next_next_node, new_shape),
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "UpdateOpDesc failed!"), return FAILED);
    auto input_dims = next_next_node->GetOpDesc()->GetInputDesc(1 - idx).GetOriginShape().GetDimNum();
    if (input_dims > 1 && SUCCESS != InputInsertReshapeNode(graph, next_next_node, (1 - idx), new_shape)) {
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "InputInsertReshapeNode failed.");
      return FAILED;
    }
  } else if (next_next_node->GetType() == "Relu") {
    out_node = next_next_node;
    FUSION_PASS_CHECK(SUCCESS != UpdateOpDesc(next_next_node, new_shape),
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "UpdateOpDesc failed!"), return FAILED);
  }

  if (SUCCESS != OutputInsertReshapeNode(graph, out_node, 0, out_shape)) {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "OutputInsertReshapeNode failed.");
    return FAILED;
  }

  return SUCCESS;
}

Status BatchMatMulV2ReshapeFusionPass::ConnectTwoElemwise(ge::ComputeGraph &graph, const ge::NodePtr &next_node,
                                                          const vector<int64_t> &new_shape,
                                                          const vector<int64_t> &out_shape) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Do fusion on connect two elemwise branch");
  auto peer_in_anchors = next_node->GetOutDataAnchor(0)->GetPeerInDataAnchors();
  auto next_next_node_0 = peer_in_anchors.at(0)->GetOwnerNode();
  auto next_next_node_1 = peer_in_anchors.at(1)->GetOwnerNode();
  OP_LOGD(FUSED_OP_TYPE.c_str(), "next node1 name = %s, node2 name = %s.", next_next_node_0->GetName().c_str(),
          next_next_node_1->GetName().c_str());

  auto input2desc = GetCurrNodeInputDesc(next_next_node_0, 1);
  FUSION_PASS_CHECK(input2desc == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input2desc is null"),
                    return FAILED);
  auto x2_shape = input2desc->GetOriginShape().GetDims();
  if (x2_shape.size() == kLeftShapeDim) {
    FUSION_PASS_CHECK(true != CheckProduct(x2_shape, 2),
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "CheckProduct x2_shape failed!"), return FAILED);
    vector<int64_t> new_x2_shape = {x2_shape[0] * x2_shape[1], x2_shape[2]};
    FUSION_PASS_CHECK(SUCCESS != InputInsertReshapeNode(graph, next_next_node_0, 1, new_x2_shape),
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "InputInsertReshapeNode failed!"), return FAILED);
  }
  FUSION_PASS_CHECK(SUCCESS != UpdateOpDescByIndex(next_next_node_0, new_shape, 0),
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "UpdateOpDescByIndex failed!"), return FAILED);
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != next_next_node_0->InferShapeAndType(),
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "InferShapeAndType failed!"), return FAILED);

  FUSION_PASS_CHECK(SUCCESS != UpdateOpDesc(next_next_node_1, new_shape),
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "UpdateOpDesc failed."), return FAILED);

  auto out_anchor = next_next_node_0->GetOutDataAnchor(0);
  peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  auto sigmoid_node = peer_in_anchors.at(0)->GetOwnerNode();
  FUSION_PASS_CHECK(SUCCESS != UpdateOpDesc(sigmoid_node, new_shape),
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "UpdateOpDesc failed."), return FAILED);
  out_anchor = sigmoid_node->GetOutDataAnchor(0);
  peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  auto mul_node = peer_in_anchors.at(0)->GetOwnerNode();
  FUSION_PASS_CHECK(SUCCESS != UpdateOpDesc(mul_node, new_shape),
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "UpdateOpDesc failed."), return FAILED);
  FUSION_PASS_CHECK(SUCCESS != OutputInsertReshapeNode(graph, mul_node, 0, out_shape),
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "OutputInsertReshapeNode failed."), return FAILED);

  return SUCCESS;
}

Status BatchMatMulV2ReshapeFusionPass::ProcessOutNode(ge::ComputeGraph &graph, const ge::NodePtr &fused_node,
                                                      const vector<int64_t> &new_shape,
                                                      const vector<int64_t> &out_shape) {
  auto out_anchor = fused_node->GetOutDataAnchor(0);
  auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  auto next_node = peer_in_anchors.at(0)->GetOwnerNode();

  int32_t idx = peer_in_anchors.at(0)->GetIdx();
  auto input_desc_ptr = GetCurrNodeMutableInputDesc(next_node, idx);
  FUSION_PASS_CHECK(input_desc_ptr == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "input_desc_ptr is null."),
                    return FAILED);
  input_desc_ptr->SetShape(ge::GeShape(new_shape));
  input_desc_ptr->SetOriginShape(ge::GeShape(new_shape));
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != next_node->InferShapeAndType(),
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "InferShapeAndType failed!"), return FAILED);
  auto out_node_num = next_node->GetOutDataNodesSize();
  if (out_node_num == 1) {
    FUSION_PASS_CHECK(SUCCESS != ConnectOneElemwise(graph, next_node, new_shape, out_shape),
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ConnectOneElemwise failed!"), return FAILED);
  } else if (out_node_num == 2) {
    FUSION_PASS_CHECK(SUCCESS != ConnectTwoElemwise(graph, next_node, new_shape, out_shape),
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ConnectTwoElemwise failed!"), return FAILED);
  }

  return SUCCESS;
}

Status BatchMatMulV2ReshapeFusionPass::ProcessOutNodeBigBatch(ge::ComputeGraph &graph, const ge::NodePtr &fused_node,
                                                              const vector<int64_t> &new_shape,
                                                              const vector<int64_t> &out_shape) {
  auto out_anchor = fused_node->GetOutDataAnchor(0);
  auto peer_in_anchors = out_anchor->GetPeerInDataAnchors();
  FUSION_PASS_CHECK(
      peer_in_anchors.size() == 0,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Error: fused_node peer_in_anchors.size() is at least 1."),
      return FAILED);
  FUSION_PASS_CHECK(SUCCESS != OutputInsertReshapeNode(graph, fused_node, 0, out_shape),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Failed to insert output node"), return FAILED);
  return SUCCESS;
}

Status BatchMatMulV2ReshapeFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                              vector<ge::NodePtr> & /* fuion_nodes */) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter BatchMatMulV2ReshapeFusionPass.");
  ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_BATCHMATMULV2, mapping);
  FUSION_PASS_CHECK(fused_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Fuse node is null, fusion failed."),
                    return PARAM_INVALID);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "The name of fused node is %s.", fused_node->GetName().c_str());
  auto input0desc = GetCurrNodeInputDesc(fused_node, 0);
  auto input1desc = GetCurrNodeInputDesc(fused_node, 1);
  FUSION_PASS_CHECK(input0desc == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputDesc0 is null"),
                    return FAILED);
  FUSION_PASS_CHECK(input1desc == nullptr, CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputDesc1 is null"),
                    return FAILED);
  auto x1_shape = input0desc->GetOriginShape().GetDims();
  auto x2_shape = input1desc->GetOriginShape().GetDims();
  auto op_desc = fused_node->GetOpDesc();
  bool is_elemwise_fusion = false;
  bool is_big_batch_fusion = false;

  FUSION_PASS_CHECK(!CheckNeedChange(fused_node, x1_shape, x2_shape, is_elemwise_fusion, is_big_batch_fusion),
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "The graph doesn't need to be changed."), return NOT_CHANGED);
  if (is_elemwise_fusion || is_big_batch_fusion) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "fuse node name [%s], is_elemwise_fusion [%d], is_big_batch_fusion [%d].",
            fused_node->GetName().c_str(), is_elemwise_fusion, is_big_batch_fusion);
    vector<int64_t> new_shape = {x1_shape[0] * x1_shape[1], x1_shape[2]};
    FUSION_PASS_CHECK(SUCCESS != InputInsertReshapeNode(graph, fused_node, 0, new_shape),
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "InputInsertReshapeNode failed!"), return FAILED);
    auto out_shape = fused_node->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDims();
    vector<int64_t> new_out_shape = {new_shape[0], x2_shape[1]};
    bool trans_b = false;
    FUSION_PASS_CHECK(!AttrUtils::GetBool(op_desc, "adj_x2", trans_b),
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GetBool adj_x2 failed!"), return FAILED);
    if (trans_b) {
      new_out_shape.at(1) = x2_shape[0];
    }
    OP_LOGD(FUSED_OP_TYPE.c_str(), "out_shape[0] is %d, new_out_shape is [%d, %d]", out_shape[0], new_out_shape[0],
            new_out_shape[1]);
    auto out_desc_ptr = fused_node->GetOpDesc()->MutableOutputDesc(0);
    out_desc_ptr->SetShape(ge::GeShape(new_out_shape));
    out_desc_ptr->SetOriginShape(ge::GeShape(new_out_shape));
    if (is_elemwise_fusion) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Do output insert reshape node for elemwise fusion scenario");
      FUSION_PASS_CHECK(SUCCESS != ProcessOutNode(graph, fused_node, new_out_shape, out_shape),
                        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Process out node failed!"), return FAILED);
    } else if (is_big_batch_fusion) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Do output insert reshape node for big batch fusion scenario");
      FUSION_PASS_CHECK(SUCCESS != ProcessOutNodeBigBatch(graph, fused_node, new_out_shape, out_shape),
                        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Process out node failed!"), return FAILED);
    }
    OP_LOGD(FUSED_OP_TYPE.c_str(), "BatchMatMulV2ReshapeFusionPass new scenario success.");
    return SUCCESS;
  }

  // step1: creat reshape op and add to graph for x1
  if (x1_shape.size() == 1) {
    ge::NodePtr x1_reshape_node = nullptr;
    vector<int64_t> new_shape = {1, x1_shape[0]};
    FUSION_PASS_CHECK(SUCCESS != InputInsertReshapeNode(graph, fused_node, 0, new_shape),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNodeInputDescPtr is null."), return FAILED);
  }
  // step2: creat reshape op and add to graph for x2
  if (x2_shape.size() == 1) {
    ge::NodePtr x2_reshape_node = nullptr;
    vector<int64_t> new_shape = {x2_shape[0], 1};
    FUSION_PASS_CHECK(SUCCESS != InputInsertReshapeNode(graph, fused_node, 1, new_shape),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNodeInputDescPtr is null."), return FAILED);
  }
  // step3: create reshape op and add to graph for batch matmul
  auto out_shape = fused_node->GetOpDesc()->GetOutputDesc(0).GetOriginShape().GetDims();
  if (x1_shape.size() == 1) {
    out_shape.erase(out_shape.begin());
  }
  if (x2_shape.size() == 1 and out_shape.size() > 1) {
    out_shape.erase(out_shape.end() - 1);
  }

  FUSION_PASS_CHECK(SUCCESS != OutputInsertReshapeNode(graph, fused_node, 0, out_shape),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNodeInputDescPtr is null."), return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "BatchMatMulV2ReshapeFusionPass success.");
  return SUCCESS;
}

Status BatchMatMulV2ReshapeFusionPass::CreateReshapeNode(ge::ComputeGraph &graph, const ge::NodePtr &next_node,
                                                         const ge::OutDataAnchorPtr &out_anchor,
                                                         const vector<int64_t> &shape, ge::NodePtr &shape_node) {
  auto previous_node = out_anchor->GetOwnerNode();
  int idx = out_anchor->GetIdx();
  auto previous_node_desc = previous_node->GetOpDesc()->GetOutputDesc(idx);
  ge::GeTensorDesc next_in_desc = previous_node_desc.Clone();
  next_in_desc.SetShape(ge::GeShape(shape));
  next_in_desc.SetOriginShape(ge::GeShape(shape));

  ge::OpDescPtr reshape_desc;
  FUSION_PASS_MAKE_SHARED((reshape_desc = std::make_shared<ge::OpDesc>(
                               next_node->GetName() + "_cann" + "_Reshape_" + previous_node->GetName(), "Reshape")),
                          return FAILED);
  FUSION_PASS_CHECK(reshape_desc->AddInputDesc("x", previous_node_desc) != GRAPH_SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add input desc x to reshape."), return FAILED);
  FUSION_PASS_CHECK(reshape_desc->AddOutputDesc("y", next_in_desc) != GRAPH_SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add input desc y to reshape."), return FAILED);
  ge::AttrUtils::SetListInt(reshape_desc, "shape", shape);

  auto new_shape_node = graph.AddNode(reshape_desc);
  FUSION_PASS_CHECK(new_shape_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add reshape to graph."),
                    return FAILED);
  shape_node = new_shape_node;
  return SUCCESS;
}

REGISTER_PASS("BatchMatMulV2ReshapeFusionPass", BUILT_IN_GRAPH_PASS, BatchMatMulV2ReshapeFusionPass);
}  // namespace fe
