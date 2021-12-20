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
 * \file batch_matmul_reduce_mean_fusion_pass.cc
 * \brief align axis n to 16 so as to delete transdata between batch_matmul and reduce_mean
 * pattern:
 *    x        const0                           const0
 *     \      /                                 /
 *   batch_matmul    const1          x       pad0      const1
 *              \    /                \      /         /
 *               add                 batch_matmul    pad1
 *                |                             \    /
 *               relu    const2  ->              add
 *                 \    /                         |
 *               reduce_mean                     relu    const2
 *                                                 \    /
 *                                               reduce_mean
 *                                                   |
 *                                                 slice
 */
#include "batch_matmul_reduce_mean_fusion_pass.h"

#include "anchor_util.h"
#include "error_util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
static const string PATTERN_BATCHMATMUL = "batch_matmul";
static const string PATTERN_ADD = "add";
static const string PATTERN_RELU = "relu";
static const string PATTERN_REDUCEMEAN = "reduce_mean";

static const string MATMUL = "MatMul";
static const string MATMULV2 = "MatMulV2";
static const string BATCHMATMUL = "BatchMatMul";
static const string BATCHMATMULV2 = "BatchMatMulV2";
static const string ADD = "Add";
static const string RELU = "Relu";
static const string REDUCEMEAND = "ReduceMeanD";
static const string PADD = "PadD";
static const string SLICED = "SliceD";
static const int ALIGN_UNIT = 16;

vector<FusionPattern *> BatchMatMulReduceMeanFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = new (std::nothrow) FusionPattern(kNameFusionPass);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGW(kNameFusionPass.c_str(), "Failed to create pattern."),
                    return patterns);

  OP_LOGD(kNameFusionPass.c_str(), "Start to define pattern.");
  pattern->AddOpDesc(PATTERN_BATCHMATMUL, {MATMUL, MATMULV2, BATCHMATMUL, BATCHMATMULV2})
    .AddOpDesc(PATTERN_ADD, {ADD})
    .AddOpDesc(PATTERN_RELU, {RELU})
    .AddOpDesc(PATTERN_REDUCEMEAN, {REDUCEMEAND})
    .SetInputs(PATTERN_ADD, {PATTERN_BATCHMATMUL})
    .SetInputs(PATTERN_RELU, {PATTERN_ADD})
    .SetInputs(PATTERN_REDUCEMEAN, {PATTERN_RELU})
    .SetOutput(PATTERN_REDUCEMEAN);
  patterns.push_back(pattern);
  OP_LOGD(kNameFusionPass.c_str(), "End to define pattern.");

  FusionPattern *pattern1 = new (std::nothrow) FusionPattern(kNameFusionPass);
  FUSION_PASS_CHECK(pattern1 == nullptr, OP_LOGW(kNameFusionPass.c_str(), "Failed to create pattern1."),
                    return patterns);

  OP_LOGD(kNameFusionPass.c_str(), "Start to define pattern1.");
  pattern1->AddOpDesc(PATTERN_BATCHMATMUL, {MATMUL, MATMULV2, BATCHMATMUL, BATCHMATMULV2})
    .AddOpDesc(PATTERN_RELU, {RELU})
    .AddOpDesc(PATTERN_REDUCEMEAN, {REDUCEMEAND})
    .SetInputs(PATTERN_RELU, {PATTERN_BATCHMATMUL})
    .SetInputs(PATTERN_REDUCEMEAN, {PATTERN_RELU})
    .SetOutput(PATTERN_REDUCEMEAN);
  patterns.push_back(pattern1);
  OP_LOGD(kNameFusionPass.c_str(), "End to define pattern1.");

  return patterns;
}

Status BatchMatMulReduceMeanFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                               vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGI(kNameFusionPass.c_str(), "Start BatchMatMulReduceMeanFusionPass.");
  FUSION_PASS_CHECK(GetNodes(mapping) != SUCCESS, OP_LOGW(kNameFusionPass.c_str(), "Failed to get nodes."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(CheckStaticShape() != SUCCESS, OP_LOGW(kNameFusionPass.c_str(), "There is an unknown shape node."),
                    return NOT_CHANGED);
  // constant to attribution of pad node is valid when input is Const node
  FUSION_PASS_CHECK(OpDescUtils::IsNonConstInput(batch_matmul_node, 1),
                    OP_LOGW(kNameFusionPass.c_str(), "Input1 of %s is not const node.",
                            batch_matmul_node->GetName().c_str()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(CheckAligned() != SUCCESS, OP_LOGW(kNameFusionPass.c_str(), "Axis m or n is not aligned."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(CheckReduceMean() != SUCCESS, OP_LOGW(kNameFusionPass.c_str(), "Check ReduceMean failed."),
                    return NOT_CHANGED);

  FUSION_PASS_CHECK(InsertBatchMatMulPadD(graph) != SUCCESS,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to insert PadD node for BatchMatMul."),
                    return FAILED);
  FUSION_PASS_CHECK(InsertAddPadD(graph) != SUCCESS,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to insert PadD node for Add."),
                    return FAILED);
  FUSION_PASS_CHECK(InsertReduceMeanSliceD(graph) != SUCCESS,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to insert SliceD for ReduceMeanD."),
                    return FAILED);
  FUSION_PASS_CHECK(UpdateAllShape(batch_matmul_node, slice_node) != SUCCESS,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to update shape of each node."),
                    return FAILED);

  OP_LOGI(kNameFusionPass.c_str(), "End BatchMatMulReduceMeanFusionPass.");
  return SUCCESS;
}

Status BatchMatMulReduceMeanFusionPass::GetNodes(const Mapping &mapping) {
  batch_matmul_node = GetNodeFromMapping(PATTERN_BATCHMATMUL, mapping);
  FUSION_PASS_CHECK(batch_matmul_node == nullptr,
                    CUBE_CALL_ERR_REPORT(kNameFusionPass.c_str(), "New batch_matmul_node not success."),
                    return PARAM_INVALID);
  add_node = GetNodeFromMapping(PATTERN_ADD, mapping);
  relu_node = GetNodeFromMapping(PATTERN_RELU, mapping);
  FUSION_PASS_CHECK(relu_node == nullptr,
                    CUBE_CALL_ERR_REPORT(kNameFusionPass.c_str(), "New relu_node not success."),
                    return PARAM_INVALID);
  reduce_mean_node = GetNodeFromMapping(PATTERN_REDUCEMEAN, mapping);
  FUSION_PASS_CHECK(reduce_mean_node == nullptr,
                    CUBE_CALL_ERR_REPORT(kNameFusionPass.c_str(), "New reduce_mean_node not success."),
                    return PARAM_INVALID);
  return SUCCESS;
}

Status BatchMatMulReduceMeanFusionPass::CheckStaticShape() {
  FUSION_PASS_CHECK(
    CheckNodeShape(batch_matmul_node) != SUCCESS,
    OP_LOGW(kNameFusionPass.c_str(), "Check %s ori-shape and shape failed.", batch_matmul_node->GetName().c_str()),
    return NOT_CHANGED);
  if (add_node != nullptr) {
    FUSION_PASS_CHECK(
      CheckNodeShape(add_node) != SUCCESS,
      OP_LOGW(kNameFusionPass.c_str(), "Check %s ori-shape and shape failed.", add_node->GetName().c_str()),
      return NOT_CHANGED);
  }
  FUSION_PASS_CHECK(
    CheckNodeShape(relu_node) != SUCCESS,
    OP_LOGW(kNameFusionPass.c_str(), "Check %s ori-shape and shape failed.", relu_node->GetName().c_str()),
    return NOT_CHANGED);
  FUSION_PASS_CHECK(
    CheckNodeShape(reduce_mean_node) != SUCCESS,
    OP_LOGW(kNameFusionPass.c_str(), "Check %s ori-shape and shape failed.", reduce_mean_node->GetName().c_str()),
    return NOT_CHANGED);
  return SUCCESS;
}

Status BatchMatMulReduceMeanFusionPass::CheckNodeShape(ge::NodePtr &node) {
  auto node_inputs = node->GetInDataNodes();
  ge::OpDescPtr node_desc = node->GetOpDesc();
  for (size_t i = 0; i < node_inputs.size(); ++i) {
    ge::GeShape input_ori_shape = node_desc->MutableInputDesc(i)->GetOriginShape();
    FUSION_PASS_CHECK(
      input_ori_shape.IsUnknownShape(),
      OP_LOGW(kNameFusionPass.c_str(), "Input %zu of %s ori-shape is unknown shape.", i, node->GetName().c_str()),
      return NOT_CHANGED);
    ge::GeShape input_shape = node_desc->MutableInputDesc(i)->GetShape();
    FUSION_PASS_CHECK(
      input_shape.IsUnknownShape(),
      OP_LOGW(kNameFusionPass.c_str(), "Input %zu of %s shape is unknown shape.", i, node->GetName().c_str()),
      return NOT_CHANGED);
  }
  return SUCCESS;
}

Status BatchMatMulReduceMeanFusionPass::CheckAligned() {
  OP_LOGD(kNameFusionPass.c_str(), "Check axis m, n of BatchMatMul whether aligned.");
  auto batch_matmul_shape = batch_matmul_node->GetOpDesc()->MutableOutputDesc(0)->GetOriginShape();
  vector<int64_t> output_dims = batch_matmul_shape.GetDims();
  // 2 means axis m
  int64_t m_dim = output_dims[output_dims.size() - 2];
  FUSION_PASS_CHECK((m_dim % ALIGN_UNIT != 0),
                    OP_LOGW(kNameFusionPass.c_str(), "Axis m of batch_matmul is not aligned to 16."),
                    return NOT_CHANGED);
  n_dim = output_dims[output_dims.size() - 1];
  FUSION_PASS_CHECK((n_dim % ALIGN_UNIT == 0),
                    OP_LOGW(kNameFusionPass.c_str(), "Axis n of batch_matmul is aligned to 16."),
                    return NOT_CHANGED);
  return SUCCESS;
}

Status BatchMatMulReduceMeanFusionPass::CheckReduceMean() {
  auto reduce_mean_input_shape = reduce_mean_node->GetOpDesc()->MutableInputDesc(0)->GetOriginShape();
  vector<int32_t> axes;
  int32_t axis_n = reduce_mean_input_shape.GetDimNum() - 1;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(reduce_mean_node->GetOpDesc(), "axes", axes),
                    OP_LOGW(kNameFusionPass.c_str(), "Get attr axes of ReduceMeanD failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(axes[0] == axis_n, OP_LOGW(kNameFusionPass.c_str(), "Reduce axis is axis n."),
                    return NOT_CHANGED);
  return SUCCESS;
}

Status BatchMatMulReduceMeanFusionPass::InsertBatchMatMulPadD(ge::ComputeGraph &graph) {
  OP_LOGD(kNameFusionPass.c_str(), "Calculate paddings for BatchMatMul.");
  n_dim_aligned = (n_dim + ALIGN_UNIT - 1) / ALIGN_UNIT * ALIGN_UNIT;
  vector<vector<int64_t> > paddings;
  auto bmm_input_1_shape = batch_matmul_node->GetOpDesc()->MutableInputDesc(1)->GetOriginShape();
  size_t len_bmm_input_1 = bmm_input_1_shape.GetDimNum();
  bool trans_b = false;
  FUSION_PASS_CHECK(
    !ge::AttrUtils::GetBool(batch_matmul_node->GetOpDesc(), "adj_x2", trans_b),
    OP_LOGE(kNameFusionPass.c_str(), "Failed to get adj_x2 of %s.", batch_matmul_node->GetName().c_str()),
    return FAILED);
  for (size_t i = 0; i < len_bmm_input_1; ++i) {
    // 2 means axis n when trans_b is true
    if ((trans_b && (i == (len_bmm_input_1 - 2))) || (! trans_b && (i == (len_bmm_input_1 - 1)))) {
      paddings.push_back({0, n_dim_aligned - n_dim});
    } else {
      paddings.push_back({0, 0});
    }
  }

  OP_LOGD(kNameFusionPass.c_str(), "Create PadD operator for BatchMatMul.");
  ge::NodePtr pad_node = nullptr;
  vector<int64_t> pad_output_shape = bmm_input_1_shape.GetDims();
  pad_output_shape[pad_output_shape.size() - 1] = n_dim_aligned;
  PadParams pad_params;
  pad_params.op_pre_peer_idx = 1;
  pad_params.shape = pad_output_shape;
  pad_params.paddings = paddings;
  FUSION_PASS_CHECK(CreatePadDNode(graph, pad_node, batch_matmul_node, pad_params) != SUCCESS,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to create PadD node for BatchMatMul."),
                    return FAILED);

  OP_LOGD(kNameFusionPass.c_str(), "Insert PadD node for BatchMatMul");
  auto dst_anchor = batch_matmul_node->GetInDataAnchor(1);
  auto src_anchor = dst_anchor->GetPeerOutAnchor();
  FUSION_PASS_CHECK(
    ge::GraphUtils::InsertNodeBetweenDataAnchors(src_anchor, dst_anchor, pad_node) != ge::GRAPH_SUCCESS,
    OP_LOGE(kNameFusionPass.c_str(), "Failed to insert PadD node for BatchMatMul."),
    return FAILED);

  return SUCCESS;
}

Status BatchMatMulReduceMeanFusionPass::InsertAddPadD(ge::ComputeGraph &graph) {
  OP_LOGD(kNameFusionPass.c_str(), "Calculate paddings for Add.");
  vector<vector<int64_t> > paddings = {{0, n_dim_aligned - n_dim}};

  OP_LOGD(kNameFusionPass.c_str(), "Create PadD operator for Add.");
  ge::NodePtr pad_node = nullptr;
  vector<int64_t> pad_output_shape = {n_dim_aligned};
  ge::NodePtr op_node = nullptr;
  int idx = 0;
  op_node = (add_node == nullptr) ? batch_matmul_node : add_node;
  // 2 means that node add is the bias input of batch_matmul
  idx = (add_node == nullptr) ? 2 : 1;
  PadParams pad_params;
  pad_params.op_pre_peer_idx = idx;
  pad_params.shape = pad_output_shape;
  pad_params.paddings = paddings;
  FUSION_PASS_CHECK(CreatePadDNode(graph, pad_node, op_node, pad_params) != SUCCESS,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to create PadD node for Add."),
                    return FAILED);

  OP_LOGD(kNameFusionPass.c_str(), "Insert PadD node for Add.");
  auto dst_anchor = op_node->GetInDataAnchor(idx);
  auto src_anchor = dst_anchor->GetPeerOutAnchor();
  FUSION_PASS_CHECK(
    ge::GraphUtils::InsertNodeBetweenDataAnchors(src_anchor, dst_anchor, pad_node) != ge::GRAPH_SUCCESS,
    OP_LOGE(kNameFusionPass.c_str(), "Failed to insert PadD node for Add."),
    return FAILED);

  return SUCCESS;
}

Status BatchMatMulReduceMeanFusionPass::InsertReduceMeanSliceD(ge::ComputeGraph &graph) {
  OP_LOGD(kNameFusionPass.c_str(), "Calculate offsets and size for SliceD.");
  auto reduce_mean_shape = reduce_mean_node->GetOpDesc()->MutableOutputDesc(0)->GetOriginShape();

  size_t len_reduce_mean_shape = reduce_mean_shape.GetDimNum();
  vector<int64_t> slice_input_shape;
  vector<int64_t> offsets;
  vector<int64_t> size = reduce_mean_shape.GetDims();
  for (size_t i = 0; i < len_reduce_mean_shape; ++i) {
    offsets.push_back(0);
    if (i == (len_reduce_mean_shape - 1)) {
      slice_input_shape.push_back(n_dim_aligned);
    } else {
      slice_input_shape.push_back(reduce_mean_shape.GetDim(i));
    }
  }

  OP_LOGD(kNameFusionPass.c_str(), "Create SliceD operator for ReduceMeanD.");
  SliceParams slice_params;
  slice_params.shape = slice_input_shape;
  slice_params.offsets = offsets;
  slice_params.size = size;
  FUSION_PASS_CHECK(CreateSliceDNode(graph, reduce_mean_node, slice_params) != ge::GRAPH_SUCCESS,
                    OP_LOGE(kNameFusionPass.c_str(), "Failed to insert SliceD node."),
                    return FAILED);

  OP_LOGD(kNameFusionPass.c_str(), "Insert SliceD node for ReduceMeanD.");
  auto src_anchor = reduce_mean_node->GetOutDataAnchor(0);
  auto dst_anchor = GetPeerInAnchorByOutDataAnchor(src_anchor, 0);
  FUSION_PASS_CHECK(
    ge::GraphUtils::InsertNodeBetweenDataAnchors(src_anchor, dst_anchor, slice_node) != ge::GRAPH_SUCCESS,
    OP_LOGE(kNameFusionPass.c_str(), "Failed to insert SliceD node for ReduceMeanD."),
    return FAILED);

  return SUCCESS;
}

Status BatchMatMulReduceMeanFusionPass::CreatePadDNode(ge::ComputeGraph &graph,
                                                       ge::NodePtr &pad_node,
                                                       const ge::NodePtr &op_node,
                                                       PadParams &pad_params) {
  // pre_peer_node (out_anchor) -> (in_anchor) pad_node (out_anchor) -> post_peer_node
  int op_pre_peer_idx = pad_params.op_pre_peer_idx;
  vector<int64_t> shape = pad_params.shape;
  vector<vector<int64_t> > paddings = pad_params.paddings;

  auto pre_peer_node = GetPeerOutNodeWithInDataAnchor(op_node, op_pre_peer_idx);
  if (op_node->GetName().find("MatMul") != string::npos) {
    FUSION_PASS_CHECK(pre_peer_node == nullptr,
                      OP_LOGE(kNameFusionPass.c_str(), "No bias in %s node.", op_node->GetName().c_str()),
                      return FAILED);
  }
  int idx = GetPeerOutAnchorWithInDataAnchor(op_node, op_pre_peer_idx)->GetIdx();
  auto pre_peer_node_desc = pre_peer_node->GetOpDesc()->MutableOutputDesc(idx);

  std::shared_ptr<ge::OpDesc> pad_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    pad_desc = std::make_shared<ge::OpDesc>(pre_peer_node->GetName() + "/" + PADD, PADD),
    return FAILED);
  ge::GeTensorDesc pad_input_desc(pre_peer_node_desc->GetShape(), pre_peer_node_desc->GetFormat(),
                                  pre_peer_node_desc->GetDataType());
  pad_input_desc.SetOriginFormat(pre_peer_node_desc->GetOriginFormat());
  pad_input_desc.SetOriginShape(pre_peer_node_desc->GetOriginShape());
  ge::GeTensorDesc pad_output_desc(ge::GeShape(shape), pre_peer_node_desc->GetFormat(),
                                   pre_peer_node_desc->GetDataType());
  pad_output_desc.SetOriginFormat(pre_peer_node_desc->GetOriginFormat());
  pad_output_desc.SetOriginShape(ge::GeShape(shape));

  FUSION_PASS_CHECK(pad_desc->AddInputDesc(pad_input_desc) != SUCCESS,
                    OP_LOGE(kNameFusionPass.c_str(), "Add input of PadD failed."),
                    return FAILED);
  FUSION_PASS_CHECK(pad_desc->AddOutputDesc(pad_output_desc) != SUCCESS,
                    OP_LOGE(kNameFusionPass.c_str(), "Add output of PadD failed."),
                    return FAILED);
  ge::AttrUtils::SetListListInt(pad_desc, "paddings", paddings);

  pad_node = graph.AddNode(pad_desc);
  FUSION_PASS_CHECK(pad_node == nullptr,
                    OP_LOGE(kNameFusionPass.c_str(), "Add PadD to graph failed."),
                    return FAILED);

  return SUCCESS;
}

Status BatchMatMulReduceMeanFusionPass::CreateSliceDNode(ge::ComputeGraph &graph,
                                                         const ge::NodePtr &op_node,
                                                         SliceParams &slice_params) {
  // pre_peer_node (out_anchor) -> slice_node
  vector<int64_t> shape = slice_params.shape;
  vector<int64_t> offsets = slice_params.offsets;
  vector<int64_t> size = slice_params.size;

  auto pre_peer_node_desc = op_node->GetOpDesc()->MutableOutputDesc(0);

  std::shared_ptr<ge::OpDesc> slice_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    slice_desc = std::make_shared<ge::OpDesc>(SLICED, SLICED),
    return FAILED);
  ge::GeTensorDesc slice_input_desc(ge::GeShape(shape), pre_peer_node_desc->GetFormat(),
                                    pre_peer_node_desc->GetDataType());
  slice_input_desc.SetOriginFormat(pre_peer_node_desc->GetOriginFormat());
  slice_input_desc.SetOriginShape(ge::GeShape(shape));
  ge::GeTensorDesc slice_output_desc(pre_peer_node_desc->GetShape(), pre_peer_node_desc->GetFormat(),
                                     pre_peer_node_desc->GetDataType());
  slice_output_desc.SetOriginFormat(pre_peer_node_desc->GetOriginFormat());
  slice_output_desc.SetOriginShape(pre_peer_node_desc->GetOriginShape());

  FUSION_PASS_CHECK(slice_desc->AddInputDesc(slice_input_desc) != SUCCESS,
                    OP_LOGE(kNameFusionPass.c_str(), "Add input of SliceD failed."),
                    return FAILED);
  FUSION_PASS_CHECK(slice_desc->AddOutputDesc(slice_output_desc) != SUCCESS,
                    OP_LOGE(kNameFusionPass.c_str(), "Add output of SliceD failed."),
                    return FAILED);
  ge::AttrUtils::SetListInt(slice_desc, "offsets", offsets);
  ge::AttrUtils::SetListInt(slice_desc, "size", size);

  slice_node = graph.AddNode(slice_desc);
  FUSION_PASS_CHECK(slice_node == nullptr,
                    OP_LOGE(kNameFusionPass.c_str(), "Add SliceD to graph failed."),
                    return FAILED);

  return SUCCESS;
}

Status BatchMatMulReduceMeanFusionPass::UpdateAllShape(ge::NodePtr &cur_node, ge::NodePtr &end_node) {
  while (cur_node != end_node) {
    auto node_inputs = cur_node->GetInDataNodes();
    for (size_t i = 0; i < node_inputs.size(); ++i) {
      auto node_input = node_inputs.at(i);
      auto node_input_shape = node_input->GetOpDesc()->MutableOutputDesc(0)->GetShape();
      auto node_input_shape_ori = node_input->GetOpDesc()->MutableOutputDesc(0)->GetOriginShape();
      cur_node->GetOpDesc()->MutableInputDesc(i)->SetShape(node_input_shape);
      cur_node->GetOpDesc()->MutableInputDesc(i)->SetOriginShape(node_input_shape_ori);
    }
    FUSION_PASS_CHECK(cur_node->InferShapeAndType() != ge::GRAPH_SUCCESS,
                      OP_LOGE(kNameFusionPass.c_str(), "%s infershape failed.", cur_node->GetName().c_str()),
                      return FAILED);
    cur_node = cur_node->GetOutDataNodes().at(0);
  }
  return SUCCESS;
}
REGISTER_PASS("BatchMatMulReduceMeanFusionPass", BUILT_IN_GRAPH_PASS, BatchMatMulReduceMeanFusionPass);
}  // namespace fe
