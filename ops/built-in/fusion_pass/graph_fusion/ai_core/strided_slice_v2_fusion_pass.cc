/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief split fusion pass(strided_slice_v2 --> strided_slice_d /
 * strided_slice)
 *
 */

#include "strided_slice_v2_fusion_pass.h"

#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <numeric>

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "tbe_fusion_pass_util.h"
#include "tbe_ops_pass_util.h"
#include "external/graph/operator_factory.h"

using namespace ge;
namespace {
const int32_t INDEX_X_TENSOR = 0;
const int32_t INDEX_BEGIN_TENSOR = 1;
const int32_t INDEX_STRIDE_TENSOR = 4;
std::vector<std::string> need_del_attr = {"begin",    "end",           "strides",       "begin_mask",
                                          "end_mask", "ellipsis_mask", "new_axis_mask", "shrink_axis_mask"};
}  // namespace

namespace fe {
const std::string ConstToAttrStridedSliceV2Pass::FUSEDNODE = "StridedSliceV2";
const std::string ConstToAttrStridedSliceV2Pass::PATTERN_FUSEDNODE = "FusedNodeStridedSlice";

bool ConstToAttrStridedSliceV2Pass::CheckMask(const int64_t new_mask,
                                              const int64_t shrink_mask,
                                              const size_t dim_num) const {
  int64_t new_axis_flag = 0;
  size_t delete_flag = 0;
  size_t base_number = 2.0;
  bool shrink_last_dim_flag = false;

  for (size_t i = 0; i < dim_num; i++) {
    if ((static_cast<uint64_t>(new_mask) & (static_cast<uint64_t>(pow(base_number, i)))) ==
        (static_cast<uint64_t>(pow(base_number, i)))) {
      new_axis_flag += 1;
    }
    if ((static_cast<uint64_t>(shrink_mask) & (static_cast<uint64_t>(pow(base_number, i)))) ==
        (static_cast<uint64_t>(pow(base_number, i)))) {
      delete_flag += 1;
      if (i == dim_num - 1) {
        shrink_last_dim_flag = true;
      }
    }
  }

  if (dim_num + new_axis_flag - delete_flag == 1) {
    OP_LOGW(FUSEDNODE.c_str(),
            "The output of strided slice is 1D, need go to aicpu");
    return false;
  }

  if ((shrink_last_dim_flag) && (dim_num != 1)) {
    OP_LOGW(FUSEDNODE.c_str(), "Shrink the last dim, need go to aicpu");
    return false;
  }

  return true;
}

bool ConstToAttrStridedSliceV2Pass::AutoRemoveInput(ge::ComputeGraph &graph, ge::NodePtr &p_node,
                                                    int64_t index) const {
  ge::OpDescPtr p_desc = p_node->GetOpDesc();
  ge::InDataAnchorPtr in_anchor_ptr = p_node->GetInDataAnchor(index);
  ge::NodeUtils::ClearInDataAnchor(p_node, in_anchor_ptr);

  // delete input node, edge if has
  ge::OutDataAnchorPtr const_anchor_ptr = in_anchor_ptr->GetPeerOutAnchor();
  if (const_anchor_ptr != nullptr) {
    ge::GraphUtils::RemoveEdge(const_anchor_ptr, in_anchor_ptr);
    ge::NodePtr const_node = const_anchor_ptr->GetOwnerNode();
    if (PatternFusionUtil::GetOutEdgeSize(const_node) == 0) {
      for (auto &control_anchor : const_node->GetInControlAnchor()->GetPeerOutControlAnchors()) {
        FUSION_PASS_CHECK(
            ge::GraphUtils::RemoveEdge(control_anchor, const_node->GetInControlAnchor()) != ge::GRAPH_SUCCESS,
            OP_LOGE(FUSEDNODE.c_str(), "Remove out control edge failed."), return false);
        FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(control_anchor, p_node->GetInControlAnchor()) != ge::GRAPH_SUCCESS,
                          OP_LOGE(FUSEDNODE.c_str(), "Faile to add input control edge for fusion node: %s.",
                                  p_node->GetName().c_str()),
                          return false);
      }
      FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(const_node),
                        OP_LOGE(FUSEDNODE.c_str(), "Remove Node[%s] failed", const_node->GetName().c_str()),
                        return false);
      OP_LOGD(FUSEDNODE.c_str(), "Remove const Node:[%s].", const_node->GetName().c_str());
    } else {
      OP_LOGD(FUSEDNODE.c_str(), "Node:[%s] have output link to other node.", const_node->GetName().c_str());
    }
  }

  if (!ge::OpDescUtils::ClearInputDesc(p_desc, index)) {
    OP_LOGE(FUSEDNODE.c_str(), "Fail to clear input desc[%ld]", index);
  }
  return true;
}

vector<FusionPattern *> ConstToAttrStridedSliceV2Pass::DefinePatterns() {
  vector<FusionPattern *> patterns;

  FusionPattern *pattern =
      new (std::nothrow) FusionPattern("ConstToAttrStridedSliceV2Fusion");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSEDNODE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSEDNODE})
      .SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

void ConstToAttrStridedSliceV2Pass::SetConstDesc(const vector<int64_t> &tensor_shape, ge::GeTensorDesc &tensor_desc,
                                                 ge::GeTensorDesc &des_desc) const {
  ge::GeShape tenShapes(tensor_shape);
  tensor_desc.SetOriginFormat(des_desc.GetOriginFormat());
  tensor_desc.SetFormat(des_desc.GetFormat());
  tensor_desc.SetOriginDataType(DataType::DT_INT64);
  tensor_desc.SetDataType(DataType::DT_INT64);
  tensor_desc.SetOriginShape(tenShapes);
  tensor_desc.SetShape(tenShapes);
  return;
}

void ConstToAttrStridedSliceV2Pass::SetConstDesc(const ge::GeShape &tensor_shape,
                                                 ge::GeTensorDesc &tensor_desc) const {
  tensor_desc.SetOriginDataType(DataType::DT_INT64);
  tensor_desc.SetDataType(DataType::DT_INT64);
  tensor_desc.SetOriginShape(tensor_shape);
  tensor_desc.SetShape(tensor_shape);
  return;
}

bool ConstToAttrStridedSliceV2Pass::GetConstValue(const Operator &op, const Tensor &const_tensor,
                                                  const DataType &dtype, std::vector<int64_t> &const_data) const {
  size_t size = 0;
  if (dtype == ge::DT_INT32) {
    int32_t *const_data_ptr = (int32_t *)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back((int32_t)((*(const_data_ptr + i))));
      OP_LOGD(FUSEDNODE.c_str(), "const data int32 fusion pass ====== %d",
              (int32_t)(*(const_data_ptr + i)));
    }
  } else if (dtype == ge::DT_INT64) {
    int64_t *const_data_ptr = (int64_t *)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back(((int64_t)(*(const_data_ptr + i))));
      OP_LOGD(FUSEDNODE.c_str(), "const data int64 fusion pass ====== %ld",
              (int64_t)(*(const_data_ptr + i)));
    }
  } else {
    OP_LOGE(FUSEDNODE.c_str(), "not support this type");
    return false;
  }
  return true;
}

Status ConstToAttrStridedSliceV2Pass::GetStridedSliceV2CpuState(const Operator &op, ge::OpDescPtr fuse_desc,
                                                                bool &need_to_cpu) const {
  size_t dim_num = op.GetInputDescByName("x").GetShape().GetDimNum();
  int64_t new_mask = 0;
  int64_t shrink_mask = 0;
  if ((ge::GRAPH_SUCCESS != op.GetAttr("new_axis_mask", new_mask)) ||
      (ge::GRAPH_SUCCESS != op.GetAttr("shrink_axis_mask", shrink_mask))) {
    OP_LOGE(FUSEDNODE.c_str(),
            "op strided_slice get attribute new axis mask or shrink axis mask "
            "failed");
    return PARAM_INVALID;
  }
  if (!CheckMask(new_mask, shrink_mask, dim_num)) {
    need_to_cpu = true;
  }

  std::vector<int64_t> dims = fuse_desc->GetOutputDesc(0).GetShape().GetDims();
  for (int64_t ele : dims) {
    FUSION_PASS_CHECK(
        ele == 0,
        OP_LOGW(FUSEDNODE.c_str(),
                "The output of strided slice's dim is 0, need go to aicpu"),
        need_to_cpu = true);
  }
  std::vector<int64_t> strides;
  op.GetAttr("strides", strides);
  for (auto s : strides) {
    if (s < 0) {
      OP_LOGW(FUSEDNODE.c_str(), "The stride less than 0, need go to aicpu");
      need_to_cpu = true;
      break;
    }
  }
  return SUCCESS;
}

Status ConstToAttrStridedSliceV2Pass::GetReverseState(const Operator &op, ge::NodePtr &fused_node,
                                                      ge::OpDescPtr fuse_desc, std::vector<int64_t> &axes_list,
                                                      bool &need_to_reverse) const {
  Tensor axes_tensor;
  GeTensorDescPtr axes_desc = fuse_desc->MutableInputDesc("axes");
  if (axes_desc != nullptr && axes_desc->GetShape().GetDimNum() > 0) {
    if (op.GetInputConstData("axes", axes_tensor) != ge::GRAPH_SUCCESS) {
      OP_LOGE(FUSEDNODE.c_str(), "Get constValue of [axes] failed.");
      return FAILED;
    }
    DataType dtype = op.GetInputDescByName("axes").GetDataType();
    GetConstValue(op, axes_tensor, dtype, axes_list);
  }
  std::vector<int64_t> end;
  std::vector<int64_t> begin;
  std::vector<int64_t> strides;
  ge::AttrUtils::GetListInt(fused_node->GetOpDesc(), "begin", begin);
  ge::AttrUtils::GetListInt(fused_node->GetOpDesc(), "end", end);
  ge::AttrUtils::GetListInt(fused_node->GetOpDesc(), "strides", strides);

  int64_t begin_dim = begin.size();
  std::vector<int64_t> x_shape = fuse_desc->GetInputDesc(0).GetShape().GetDims();
  int64_t dim_num = x_shape.size();
  if (axes_list.empty()) {
    axes_list.resize(begin_dim);
    std::iota(axes_list.begin(), axes_list.end(), 0);
  } else {
    // set new axis in case that it is negtive
    for (size_t i = 0; i < axes_list.size(); i++) {
      int64_t& indice = axes_list[i];
      indice = indice < 0 ? indice + dim_num : indice;
      if (indice < 0) {
        indice = 0;
      } else if (indice > dim_num - 1) {
        indice = dim_num - 1;
      }
    }
  }

  int64_t axes_dim_num = axes_list.size();
  int64_t strides_dim_num = strides.size();
  int64_t end_dim_num = end.size();
  for (int32_t i = 0; i < axes_dim_num; i++) {
    int64_t cur_axes = axes_list[i];
    // cur_axes is valid for x_shape
    if ((cur_axes < begin_dim && begin[cur_axes] != -1) ||
        (cur_axes < end_dim_num && end[cur_axes] > -x_shape[cur_axes] - 1) ||
        (cur_axes < strides_dim_num && strides[cur_axes] != -1)) {
      need_to_reverse = false;
      OP_LOGW(FUSEDNODE.c_str(), "The stride need not go to ReverseV2D");
      break;
    }
  }
  return SUCCESS;
}

void ConstToAttrStridedSliceV2Pass::UpdateShapeAndDataType(ge::NodePtr &fused_node, ge::OpDescPtr fuse_desc) const {
  std::vector<int64_t> begin;
  ge::AttrUtils::GetListInt(fused_node->GetOpDesc(), "begin", begin);
  int64_t dim_num = begin.size();
  std::vector<int64_t> begin_dim = {dim_num};
  ge::GeShape begin_shape(begin_dim);
  begin_shape.SetDim(0, dim_num);

  // update shape
  size_t stride_idx = fuse_desc->GetInputIndexByName("strides");
  auto stride_anchor = fused_node->GetInDataAnchor(stride_idx);
  auto stride_out_anchor = stride_anchor->GetPeerOutAnchor();
  auto stride_out_node = stride_out_anchor->GetOwnerNode();
  ge::OpDescPtr stride_desc = stride_out_node->GetOpDesc();
  ge::GeTensorDesc stride_desc_output = stride_desc->GetOutputDesc(0);
  SetConstDesc(begin_shape, stride_desc_output);
  stride_desc->UpdateOutputDesc(0, stride_desc_output);

  size_t ends_idx = fuse_desc->GetInputIndexByName("end");
  auto end_anchor = fused_node->GetInDataAnchor(ends_idx);
  auto end_out_anchor = end_anchor->GetPeerOutAnchor();
  auto end_out_node = end_out_anchor->GetOwnerNode();
  ge::OpDescPtr end_desc = end_out_node->GetOpDesc();
  ge::GeTensorDesc end_desc_output = end_desc->GetOutputDesc(0);
  SetConstDesc(begin_shape, end_desc_output);
  end_desc->UpdateOutputDesc(0, end_desc_output);

  size_t begin_idx = fuse_desc->GetInputIndexByName("begin");
  auto begin_anchor = fused_node->GetInDataAnchor(begin_idx);
  auto begin_out_anchor = begin_anchor->GetPeerOutAnchor();
  auto begin_out_node = begin_out_anchor->GetOwnerNode();
  ge::OpDescPtr begin_desc = begin_out_node->GetOpDesc();
  ge::GeTensorDesc begin_desc_output = begin_desc->GetOutputDesc(0);
  SetConstDesc(begin_shape, begin_desc_output);
  begin_desc->UpdateOutputDesc(0, begin_desc_output);

  fuse_desc->UpdateInputDesc(stride_idx, stride_desc->GetOutputDesc(0));
  fuse_desc->UpdateInputDesc(ends_idx, end_desc->GetOutputDesc(0));
  fuse_desc->UpdateInputDesc(begin_idx, begin_desc->GetOutputDesc(0));
}

void ConstToAttrStridedSliceV2Pass::MakeConstNode(ge::NodePtr &fused_node, ge::OpDescPtr fuse_desc) const {
  std::vector<int64_t> begin;
  std::vector<int64_t> end;
  std::vector<int64_t> strides;

  ge::AttrUtils::GetListInt(fused_node->GetOpDesc(), "begin", begin);
  ge::AttrUtils::GetListInt(fused_node->GetOpDesc(), "end", end);
  ge::AttrUtils::GetListInt(fused_node->GetOpDesc(), "strides", strides);

  int64_t dim_num = begin.size();
  std::vector<int64_t> inputs = fuse_desc->GetInputDesc(0).GetShape().GetDims();
  for (int i = 0; i < dim_num; i++) {
    if (end[i] < -inputs[i] - 1) {
      int64_t old_end = end[i];
      end[i] = -inputs[i] - 1;
      OP_LOGW("StridedSlice", "update end %ld --> %ld", old_end, end[i]);
    }
  }

  std::vector<int64_t> begin_dim = {dim_num};
  ge::GeShape beginShape(begin_dim);

  vector<ge::GeTensorPtr> slice_tensor_ptr = ge::OpDescUtils::MutableWeights(fused_node);
  // handle begin
  ge::GeTensorPtr begin_tensor_ptr = slice_tensor_ptr[0];
  ge::GeTensorDesc op_tensor_desc = begin_tensor_ptr->MutableTensorDesc();
  SetConstDesc(begin_dim, begin_tensor_ptr->MutableTensorDesc(), op_tensor_desc);
  begin_tensor_ptr->SetData(reinterpret_cast<uint8_t *>(begin.data()), dim_num * sizeof(int64_t));

  // handle end
  ge::GeTensorPtr end_tensor_ptr = slice_tensor_ptr[1];
  SetConstDesc(begin_dim, end_tensor_ptr->MutableTensorDesc(), op_tensor_desc);
  end_tensor_ptr->SetData(reinterpret_cast<uint8_t *>(end.data()), dim_num * sizeof(int64_t));

  // optional input
  size_t stride_idx = fuse_desc->GetInputIndexByName("strides");
  if (fuse_desc->MutableInputDesc(stride_idx) != nullptr) {
    OP_LOGW("StridedSlice", "update strides const node");
    // update const node
    ge::GeTensorPtr strides_tensor_ptr = slice_tensor_ptr[stride_idx - 1];  // index strides except x
    SetConstDesc(begin_dim, strides_tensor_ptr->MutableTensorDesc(), op_tensor_desc);
    strides_tensor_ptr->SetData(reinterpret_cast<uint8_t *>(strides.data()), dim_num * sizeof(int64_t));
  } else {
    OP_LOGW("StridedSlice", "add strides const node");
    // update stride input desc of slice op
    ge::GeTensorDesc strides_tensor_desc;
    SetConstDesc(begin_dim, strides_tensor_desc, op_tensor_desc);
    fuse_desc->UpdateInputDesc(stride_idx, strides_tensor_desc);

    // add const node and link to slice op
    ge::OpDescPtr const_op_desc = nullptr;
    FUSION_PASS_MAKE_SHARED(const_op_desc = std::make_shared<ge::OpDesc>(fuse_desc->GetName() + "_strides", "Const"),
                            return );
    ge::GeTensorPtr strides_tensor_ptr = nullptr;
    FUSION_PASS_MAKE_SHARED(
        (strides_tensor_ptr = std::make_shared<ge::GeTensor>(
             strides_tensor_desc, reinterpret_cast<uint8_t *>(strides.data()), dim_num * sizeof(int64_t))),
        return );
    AttrUtils::SetTensor(const_op_desc, ATTR_NAME_WEIGHTS, strides_tensor_ptr);
    const_op_desc->AddOutputDesc(fuse_desc->GetInputDesc(stride_idx));
    auto owner_graph = fused_node->GetOwnerComputeGraph();
    NodePtr const_node = owner_graph->AddNode(const_op_desc);
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), fused_node->GetInDataAnchor(stride_idx));
  }
}

bool ConstToAttrStridedSliceV2Pass::CheckDynamicShape(ge::OpDescPtr fuse_desc) const {
  vector<int64_t> dims = fuse_desc->GetOutputDesc(0).GetShape().GetDims();
  if (dims == UNKNOWN_RANK) {
    OP_LOGD(FUSEDNODE.c_str(), "is dynamic shape scene.");
    return true;
  }
  for (int64_t ele : dims) {
    if (ele == UNKNOWN_DIM) {
      OP_LOGD(FUSEDNODE.c_str(), "is dynamic shape scene.");
      return true;
    }
  }
  OP_LOGD(FUSEDNODE.c_str(), "is static shape scene.");
  return false;
}

Status ConstToAttrStridedSliceV2Pass::CreateReverseNode(ge::ComputeGraph &graph, ge::NodePtr &fused_node,
                                                        ge::OpDescPtr &fuse_desc,
                                                        std::vector<int64_t> &new_axes) const {
  for (int32_t index = INDEX_STRIDE_TENSOR; index > INDEX_BEGIN_TENSOR; index--) {
    FUSION_PASS_CHECK(!AutoRemoveInput(graph, fused_node, index),
                      OP_LOGE("StridedSliceV2", "remove input %s failed, fusion failed.",
                              fuse_desc->GetInputNameByIndex(index).c_str()),
                      return FAILED);
  }
  for (size_t i = 0; i < need_del_attr.size(); i++) {
    fuse_desc->DelAttr(need_del_attr[i]);
  }
  std::map<string, uint32_t> name_index_map = {{"x", 0}, {"axis", 1}};
  std::vector<std::string> const_tensor_name = {"axis"};
  int64_t new_axis_len = new_axes.size();
  std::vector<int64_t> axes_dim = {new_axis_len};
  std::vector<ge::GeTensorPtr> slice_tensor_ptr = ge::OpDescUtils::MutableWeights(fused_node);
  // handle axes
  ge::GeTensorPtr axes_tensor_ptr = slice_tensor_ptr[0];
  ge::GeTensorDesc op_tensor_desc = axes_tensor_ptr->MutableTensorDesc();
  SetConstDesc(axes_dim, axes_tensor_ptr->MutableTensorDesc(), op_tensor_desc);
  axes_tensor_ptr->SetData(reinterpret_cast<uint8_t *>(new_axes.data()), new_axis_len * sizeof(int64_t));
  fuse_desc->SetType("ReverseV2");
  fuse_desc->UpdateInputName(name_index_map);
  fuse_desc->SetOpInferDepends(const_tensor_name);
  return SUCCESS;
}

Status ConstToAttrStridedSliceV2Pass::CreateReverseDNode(Operator &op, ge::ComputeGraph &graph,
                                                         ge::NodePtr &fused_node, ge::OpDescPtr &fuse_desc,
                                                         std::vector<int64_t> &new_axes) const {
  for (int32_t index = INDEX_STRIDE_TENSOR; index > INDEX_BEGIN_TENSOR; index--) {
    FUSION_PASS_CHECK(!AutoRemoveInput(graph, fused_node, index),
                      OP_LOGE("StridedSliceD", "remove input %s failed, fusion failed.",
                              fuse_desc->GetInputNameByIndex(index).c_str()),
                      return FAILED);
  }
  for (size_t i = 0; i < need_del_attr.size(); i++) {
    fuse_desc->DelAttr(need_del_attr[i]);
  }
  std::map<string, uint32_t> name_index_map = {{"x", 0}};
  FUSION_PASS_CHECK(!AutoRemoveInput(graph, fused_node, INDEX_BEGIN_TENSOR),
                    OP_LOGE("StridedSliceD", "remove input %s failed, fusion failed.",
                            fuse_desc->GetInputNameByIndex(INDEX_BEGIN_TENSOR).c_str()),
                    return FAILED);
  op.SetAttr("axis", new_axes);
  fuse_desc->SetType("ReverseV2D");
  fuse_desc->UpdateInputName(name_index_map);
  ClearOpInferDepends(fused_node);
  return SUCCESS;
}

Status ConstToAttrStridedSliceV2Pass::CreateStridedSliceNode(ge::ComputeGraph &graph, ge::NodePtr &fused_node,
                                                             ge::OpDescPtr &fuse_desc) const {
  // construct const tensor : begin, end, strides
  MakeConstNode(fused_node, fuse_desc);
  UpdateShapeAndDataType(fused_node, fuse_desc);
  FUSION_PASS_CHECK(!AutoRemoveInput(graph, fused_node, fuse_desc->GetInputIndexByName("axes")),
                    OP_LOGE(FUSEDNODE.c_str(), "remove input axes failed, fusion failed."), return FAILED);
  fuse_desc->DelAttr("begin");
  fuse_desc->DelAttr("end");
  fuse_desc->DelAttr("strides");
  fuse_desc->SetType("StridedSlice");  // AICPU Operator
  std::map<string, uint32_t> name_index_map = {{"x", 0}, {"begin", 1}, {"end", 2}, {"strides", 3}};
  fuse_desc->UpdateInputName(name_index_map);
  vector<string> infer_depends_vec = {"begin", "end", "strides"};
  fuse_desc->SetOpInferDepends(infer_depends_vec);
  return SUCCESS;
}

Status ConstToAttrStridedSliceV2Pass::CreateStridedSliceV3Node(ge::OpDescPtr &fuse_desc) const {
  for (size_t i = 0; i < need_del_attr.size(); i++) {
    fuse_desc->DelAttr(need_del_attr[i]);
  }
  fuse_desc->SetType("StridedSliceV3");

  auto realFusedOp = ge::OperatorFactory::CreateOperator("realFusedOp", "StridedSliceV3");
  if (realFusedOp.IsEmpty()) {
    OP_LOGE(FUSEDNODE.c_str(), "Create op [StridedSliceV3] failed.");
    return FAILED;
  }
  auto realFusedOpDescPtr = ge::OpDescUtils::GetOpDescFromOperator(realFusedOp);
  realFusedOp.BreakConnect();
  fuse_desc->AddInferFunc(realFusedOpDescPtr->GetInferFunc());
  return SUCCESS;
}

Status ConstToAttrStridedSliceV2Pass::CreateStridedSliceDNode(ge::ComputeGraph &graph, ge::NodePtr &fused_node,
                                                              ge::OpDescPtr &fuse_desc) const {
  // remove input node as index descend
  for (int32_t index = INDEX_STRIDE_TENSOR; index > INDEX_X_TENSOR; index--) {
    FUSION_PASS_CHECK(!AutoRemoveInput(graph, fused_node, index),
                      OP_LOGE(FUSEDNODE.c_str(), "remove input %s failed, fusion failed.",
                              fuse_desc->GetInputNameByIndex(index).c_str()),
                      return FAILED);
  }
  fuse_desc->SetType("StridedSliceD");  // AICORE Operator
  std::map<string, uint32_t> name_index_map = {{"x", 0}};
  fuse_desc->UpdateInputName(name_index_map);
  ClearOpInferDepends(fused_node);
  return SUCCESS;
}

Status ConstToAttrStridedSliceV2Pass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                             vector<ge::NodePtr> &fusion_node) {
  ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fused_node == nullptr, OP_LOGE(FUSEDNODE.c_str(), "fused_node is null"), return PARAM_INVALID);
  ge::OpDescPtr fuse_desc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(fuse_desc == nullptr, OP_LOGE(FUSEDNODE.c_str(), "fused_node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fused_node);

  // get fuzz build attr
  bool is_dynamic_shape = CheckDynamicShape(fuse_desc);
  bool need_to_reverse = true;
  std::vector<int64_t> new_axes;
  FUSION_PASS_CHECK(GetReverseState(op, fused_node, fuse_desc, new_axes, need_to_reverse) != SUCCESS,
                    OP_LOGE(FUSEDNODE.c_str(), "GetReverseState failed."), return FAILED);
  bool need_to_cpu = false;
  if (need_to_reverse == false) {
    FUSION_PASS_CHECK(GetStridedSliceV2CpuState(op, fuse_desc, need_to_cpu) != SUCCESS,
                      OP_LOGE(FUSEDNODE.c_str(), "GetStridedSliceV2CpuState failed."), return FAILED);
  }
  if (need_to_reverse) {
    if (is_dynamic_shape) {
      FUSION_PASS_CHECK(CreateReverseNode(graph, fused_node, fuse_desc, new_axes) != SUCCESS,
                        OP_LOGE(FUSEDNODE.c_str(), "CreateReverseNode failed."), return FAILED);
    } else {
      FUSION_PASS_CHECK(CreateReverseDNode(op, graph, fused_node, fuse_desc, new_axes) != SUCCESS,
                        OP_LOGE(FUSEDNODE.c_str(), "CreateReverseDNode failed."), return FAILED);
    }
  } else if (need_to_cpu) {
    FUSION_PASS_CHECK(CreateStridedSliceNode(graph, fused_node, fuse_desc) != SUCCESS,
                      OP_LOGE(FUSEDNODE.c_str(), "CreateStridedSliceNode failed."), return FAILED);
  } else {
    if (is_dynamic_shape) {
      FUSION_PASS_CHECK(CreateStridedSliceV3Node(fuse_desc) != SUCCESS,
                        OP_LOGE(FUSEDNODE.c_str(), "CreateStridedSliceV3Node failed."), return FAILED);
      return SUCCESS;
    }
    FUSION_PASS_CHECK(CreateStridedSliceDNode(graph, fused_node, fuse_desc) != SUCCESS,
                      OP_LOGE(FUSEDNODE.c_str(), "CreateStridedSliceV3Node failed."), return FAILED);
  }
  OP_LOGD(FUSEDNODE.c_str(), "Set OpType to [ %s ]", fuse_desc->GetType().c_str());
  fusion_node.push_back(fused_node);
  return SUCCESS;
}
REGISTER_PASS("ConstToAttrStridedSliceV2Fusion", BUILT_IN_GRAPH_PASS, ConstToAttrStridedSliceV2Pass);
}  // namespace fe
