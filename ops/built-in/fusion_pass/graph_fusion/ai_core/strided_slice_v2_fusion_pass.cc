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
namespace fe {
const int64_t ConstToAttrStridedSliceV2Pass::DEFAULT_STRIDE = 1;
const std::string ConstToAttrStridedSliceV2Pass::FUSEDNODE = "StridedSliceV2";
const std::string ConstToAttrStridedSliceV2Pass::PATTERN_FUSEDNODE =
    "FusedNodeStridedSlice";

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

bool ConstToAttrStridedSliceV2Pass::AutoRemoveInput(ge::ComputeGraph &graph,
                                                    ge::NodePtr &p_node,
                                                    const int64_t index) {
  ge::OpDescPtr p_desc = p_node->GetOpDesc();
  ge::InDataAnchorPtr in_anchor_ptr = p_node->GetInDataAnchor(index);
  ge::NodeUtils::ClearInDataAnchor(p_node, in_anchor_ptr);

  // delete input node, edge if has
  ge::OutDataAnchorPtr const_anchor_ptr = in_anchor_ptr->GetPeerOutAnchor();
  if (const_anchor_ptr != nullptr) {
    ge::GraphUtils::RemoveEdge(const_anchor_ptr, in_anchor_ptr);
    ge::NodePtr const_node_1 = const_anchor_ptr->GetOwnerNode();
    if (PatternFusionUtil::GetOutEdgeSize(const_node_1) == 0) {
      for (auto &const_peer_control_anchor : const_node_1->GetInControlAnchor()->GetPeerOutControlAnchors()){
        FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(const_peer_control_anchor, const_node_1->GetInControlAnchor()) != SUCCESS,
                          OP_LOGE(FUSEDNODE.c_str(), "Remove out control edge failed."), return FAILED);
        FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(const_peer_control_anchor, p_node->GetInControlAnchor()) != SUCCESS,
                          OP_LOGE(FUSEDNODE.c_str(), "Faile to add input control edge for fusion node: %s.",
                                  p_node->GetName().c_str()),
                          return FAILED);
      }
      FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(const_node_1),
                        OP_LOGE(FUSEDNODE.c_str(), "Remove Node[%s] failed",
                                const_node_1->GetName().c_str()),
                        return false);
      OP_LOGD(FUSEDNODE.c_str(), "Remove const Node:[%s].",
              const_node_1->GetName().c_str());
    } else {
      OP_LOGD(FUSEDNODE.c_str(), "Node:[%s] have output link to other node.",
              const_node_1->GetName().c_str());
    }
  }

  if (!ge::OpDescUtils::ClearInputDesc(p_desc, index)) {
    OP_LOGE(FUSEDNODE.c_str(), "Fail to clear input desc[%d]", index);
  }

  return true;
}

vector<FusionPattern *> ConstToAttrStridedSliceV2Pass::DefinePatterns() {
  vector<FusionPattern *> patterns;

  FusionPattern *pattern =
      new (std::nothrow) FusionPattern("ConstToAttrStridedSliceV2Fusion");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE("new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSEDNODE})
      .SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

Status ConstToAttrStridedSliceV2Pass::SetConstDesc(
    vector<int64_t> &tensor_shape, ge::GeTensorDesc &tensor_desc,
    ge::GeTensorDesc &des_desc) const {
  ge::GeShape tenShapes(tensor_shape);
  tensor_desc.SetOriginFormat(des_desc.GetOriginFormat());
  tensor_desc.SetFormat(des_desc.GetFormat());
  tensor_desc.SetOriginDataType(DataType::DT_INT64);
  tensor_desc.SetDataType(DataType::DT_INT64);
  tensor_desc.SetOriginShape(tenShapes);
  tensor_desc.SetShape(tenShapes);
  return SUCCESS;
}

bool ConstToAttrStridedSliceV2Pass::GetConstValue(
    const Operator &op, const Tensor &const_tensor, const DataType &dtype,
    std::vector<int64_t> &const_data) {
  size_t size = 0;
  if (dtype == ge::DT_INT32) {
    int32_t *const_data_ptr = (int32_t *)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back((int32_t)((*(const_data_ptr + i))));
      OP_LOGD(op.GetName().c_str(), "const data int32 fusion pass ====== %d",
              (int32_t)(*(const_data_ptr + i)));
    }
  } else if (dtype == ge::DT_INT64) {
    int64_t *const_data_ptr = (int64_t *)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back(((int64_t)(*(const_data_ptr + i))));
      OP_LOGD(op.GetName().c_str(), "const data int64 fusion pass ====== %d",
              (int64_t)(*(const_data_ptr + i)));
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "not support this type");
    return false;
  }
  return true;
}

Status ConstToAttrStridedSliceV2Pass::GetStridedSliceV2CpuState(
    const Operator &op, ge::NodePtr &fused_node, ge::OpDescPtr fuse_desc,
    bool &need_to_cpu) {
  size_t dim_num = op.GetInputDesc("x").GetShape().GetDimNum();
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
  if (strides.back() != 1) {
    OP_LOGW(FUSEDNODE.c_str(),
            "The stride of last dim is not equal 1, need go to aicpu");
    need_to_cpu = true;
  }
  for (auto s : strides) {
    if (s < 0) {
      OP_LOGW(FUSEDNODE.c_str(), "The stride less than 0, need go to aicpu");
      need_to_cpu = true;
      break;
    }
  }
  return SUCCESS;
}

Status ConstToAttrStridedSliceV2Pass::GetReverseState(
    const Operator &op, ge::NodePtr &fused_node, ge::OpDescPtr fuse_desc,
    std::vector<int64_t> &new_axes, bool &need_to_reverse) {
  Tensor axes_tensor;
  vector<int64_t> axes_list;
  bool no_axes = false;
  if (fuse_desc->MutableInputDesc(fuse_desc->GetInputIndexByName("axes")) !=
      nullptr) {
    if (op.GetInputConstData("axes", axes_tensor) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "Get constValue failed of [axes]");
      return GRAPH_FAILED;
    }
    DataType dtype = op.GetInputDesc("axes").GetDataType();
    GetConstValue(op, axes_tensor, dtype, axes_list);
  } else {
    no_axes = true;
  }
  std::vector<int64_t> end;
  std::vector<int64_t> begin;
  std::vector<int64_t> strides;
  ge::AttrUtils::GetListInt(fused_node->GetOpDesc(), "begin", begin);
  ge::AttrUtils::GetListInt(fused_node->GetOpDesc(), "end", end);
  ge::AttrUtils::GetListInt(fused_node->GetOpDesc(), "strides", strides);
  int64_t indice = 0;
  int64_t begin_dim = begin.size();
  int64_t dim_num = op.GetInputDesc("x").GetShape().GetDimNum();
  if (no_axes) {
    for (int32_t i = 0; i < begin_dim; i++) {
      new_axes.push_back(i);
    }
  } else {
    // set new axis in case that it is negtive
    for (size_t i = 0; i < axes_list.size(); i++) {
      indice = axes_list[i] < 0 ? axes_list[i] + dim_num : axes_list[i];
      if (indice < 0) {
        indice = 0;
      } else if (indice > dim_num - 1) {
        indice = dim_num - 1;
      }
      new_axes.push_back(indice);
    }
  }
  int64_t axes_dim_num = new_axes.size();
  std::vector<int64_t> inputs = fuse_desc->GetInputDesc(0).GetShape().GetDims();
  for (int32_t i = 0; i < axes_dim_num; i++) {
    if (begin[new_axes[i]] != -1 ||
        end[new_axes[i]] > -inputs[new_axes[i]] - 1 ||
        strides[new_axes[i]] != -1) {
      need_to_reverse = false;
      OP_LOGW(FUSEDNODE.c_str(), "The stride need not go to ReverseV2D");
    }
  }
  return SUCCESS;
}

void ConstToAttrStridedSliceV2Pass::UpdateShapeAndDataType(ge::NodePtr &fused_node, ge::OpDescPtr fuse_desc) {
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
  stride_desc_output.SetOriginShape(begin_shape);
  stride_desc_output.SetShape(begin_shape);
  stride_desc_output.SetOriginDataType(DataType::DT_INT64);
  stride_desc_output.SetDataType(DataType::DT_INT64);
  stride_desc->UpdateOutputDesc(0, stride_desc_output);

  size_t ends_idx = fuse_desc->GetInputIndexByName("end");
  auto end_anchor = fused_node->GetInDataAnchor(ends_idx);
  auto end_out_anchor = end_anchor->GetPeerOutAnchor();
  auto end_out_node = end_out_anchor->GetOwnerNode();
  ge::OpDescPtr end_desc = end_out_node->GetOpDesc();
  ge::GeTensorDesc end_desc_output = end_desc->GetOutputDesc(0);
  end_desc_output.SetOriginShape(begin_shape);
  end_desc_output.SetShape(begin_shape);
  end_desc_output.SetOriginDataType(DataType::DT_INT64);
  end_desc_output.SetDataType(DataType::DT_INT64);
  end_desc->UpdateOutputDesc(0, end_desc_output);

  size_t begin_idx = fuse_desc->GetInputIndexByName("begin");
  auto begin_anchor = fused_node->GetInDataAnchor(begin_idx);
  auto begin_out_anchor = begin_anchor->GetPeerOutAnchor();
  auto begin_out_node = begin_out_anchor->GetOwnerNode();
  ge::OpDescPtr begin_desc = begin_out_node->GetOpDesc();
  ge::GeTensorDesc begin_desc_output = begin_desc->GetOutputDesc(0);
  begin_desc_output.SetOriginShape(begin_shape);
  begin_desc_output.SetShape(begin_shape);
  begin_desc_output.SetOriginDataType(DataType::DT_INT64);
  begin_desc_output.SetDataType(DataType::DT_INT64);
  begin_desc->UpdateOutputDesc(0, begin_desc_output);

  fuse_desc->UpdateInputDesc(stride_idx, stride_desc->GetOutputDesc(0));
  fuse_desc->UpdateInputDesc(ends_idx, end_desc->GetOutputDesc(0));
  fuse_desc->UpdateInputDesc(begin_idx, begin_desc->GetOutputDesc(0));
}

void ConstToAttrStridedSliceV2Pass::MakeConstNode(ge::NodePtr &fused_node,
                                                  ge::OpDescPtr fuse_desc) {
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

  vector<ge::GeTensorPtr> slice_tensor_ptr =
      ge::OpDescUtils::MutableWeights(fused_node);
  ge::GeTensorPtr begin_tensor_ptr = slice_tensor_ptr[0];  // begin
  ge::GeTensorDesc op_tensor_desc = begin_tensor_ptr->MutableTensorDesc();
  (void)SetConstDesc(begin_dim, begin_tensor_ptr->MutableTensorDesc(),
                     op_tensor_desc);
  begin_tensor_ptr->SetData(reinterpret_cast<uint8_t *>(begin.data()),
                            dim_num * sizeof(int64_t));

  ge::GeTensorPtr end_tensor_ptr = slice_tensor_ptr[1];  // end
  (void)SetConstDesc(begin_dim, end_tensor_ptr->MutableTensorDesc(),
                     op_tensor_desc);
  end_tensor_ptr->SetData(reinterpret_cast<uint8_t *>(end.data()),
                          dim_num * sizeof(int64_t));

  // optional input
  size_t stride_idx = fuse_desc->GetInputIndexByName("strides");
  if (fuse_desc->MutableInputDesc(stride_idx) != nullptr) {
    OP_LOGW("StridedSlice", "update strides const node");
    // update const node
    ge::GeTensorPtr strides_tensor_ptr =
        slice_tensor_ptr[stride_idx - 1];  // index strides except x
    (void)SetConstDesc(begin_dim, strides_tensor_ptr->MutableTensorDesc(),
                       op_tensor_desc);
    strides_tensor_ptr->SetData(reinterpret_cast<uint8_t *>(strides.data()),
                                dim_num * sizeof(int64_t));
  } else {
    OP_LOGW("StridedSlice", "add strides const node");
    // update stride input desc of slice op
    ge::GeTensorDesc strides_tensor_desc;
    (void)SetConstDesc(begin_dim, strides_tensor_desc, op_tensor_desc);
    fuse_desc->UpdateInputDesc(stride_idx, strides_tensor_desc);

    // add const node and link to slice op
    ge::OpDescPtr const_op_desc = nullptr;
    FUSION_PASS_MAKE_SHARED(
        const_op_desc = std::make_shared<ge::OpDesc>(fuse_desc->GetName() + "_strides", "Const"), return);
    ge::GeTensorPtr strides_tensor_ptr = nullptr;
    FUSION_PASS_MAKE_SHARED(
        (strides_tensor_ptr =
             std::make_shared<ge::GeTensor>(strides_tensor_desc, reinterpret_cast<uint8_t *>(strides.data()),
                                            dim_num * sizeof(int64_t))),
        return);
    AttrUtils::SetTensor(const_op_desc, ATTR_NAME_WEIGHTS, strides_tensor_ptr);
    const_op_desc->AddOutputDesc(fuse_desc->GetInputDesc(stride_idx));
    auto owner_graph = fused_node->GetOwnerComputeGraph();
    NodePtr const_node = owner_graph->AddNode(const_op_desc);
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0),
                        fused_node->GetInDataAnchor(stride_idx));
  }
}

Status ConstToAttrStridedSliceV2Pass::Fusion(ge::ComputeGraph &graph,
                                             Mapping &mapping,
                                             vector<ge::NodePtr> &fusion_node) {
  ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fused_node == nullptr,
                    OP_LOGE(FUSEDNODE.c_str(), "fused_node is null"),
                    return PARAM_INVALID);
  ge::OpDescPtr fuse_desc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(
      fuse_desc == nullptr,
      OP_LOGE(FUSEDNODE.c_str(), "fused_node's OpDesc is null, fusion failed."),
      return PARAM_INVALID);
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fused_node);

  // get fuzz build attr
  bool is_dynamic_shape = false;
  vector<int64_t> dims = fuse_desc->GetOutputDesc("y").GetShape().GetDims();
  if (dims == UNKNOWN_RANK) {
    is_dynamic_shape = true;
  } else {
    for (int64_t ele : dims) {
      if (ele == UNKNOWN_DIM) {
        is_dynamic_shape = true;
        break;
      }
    }
  }
  if (is_dynamic_shape) {
    OP_LOGD(FUSEDNODE.c_str(), "is dynamic shape.");
    std::vector<string> need_del_attr = {"begin", "end", "strides", "begin_mask", "end_mask",
                                         "ellipsis_mask", "new_axis_mask", "shrink_axis_mask"};
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
  } else {
    OP_LOGD(FUSEDNODE.c_str(), "is not dynamic shape.");
  }

  bool need_to_reverse = true;
  std::vector<int64_t> new_axes;
  FUSION_PASS_CHECK(GetReverseState(op, fused_node, fuse_desc, new_axes,
                                    need_to_reverse) != SUCCESS,
                    OP_LOGE(FUSEDNODE.c_str(), "GetReverseState failed."),
                    return GRAPH_FAILED);
  bool need_to_cpu = false;
  if (need_to_reverse == false) {
    FUSION_PASS_CHECK(
        GetStridedSliceV2CpuState(op, fused_node, fuse_desc, need_to_cpu) ==
            PARAM_INVALID,
        OP_LOGE(FUSEDNODE.c_str(), "GetStridedSliceV2CpuState failed."),
        return PARAM_INVALID);
  }
  if (need_to_reverse) {
    for (int32_t index = 4; index > 0; index--) {
      FUSION_PASS_CHECK(
          !AutoRemoveInput(graph, fused_node, index),
          OP_LOGE("StridedSliceD", "remove input %s failed, fusion failed.",
                  fuse_desc->GetInputNameByIndex(index).c_str()),
          return GRAPH_FAILED);
    }
    std::vector<string> need_del_attr = {"begin", "end", "strides", "begin_mask", "end_mask",
                                         "ellipsis_mask", "new_axis_mask", "shrink_axis_mask"};
    for (size_t i = 0; i < need_del_attr.size(); i++) {
      fuse_desc->DelAttr(need_del_attr[i]);
    }
    // update stride input desc of slice op
    op.SetAttr("axis", new_axes);
    fuse_desc->SetType("ReverseV2D");
    std::map<string, uint32_t> name_index_map = {{"x", 0}};
    fuse_desc->UpdateInputName(name_index_map);
    ClearOpInferDepends(fused_node);
  } else if (need_to_cpu) {
    // construct const tensor : begin, end, strides
    MakeConstNode(fused_node, fuse_desc);
    UpdateShapeAndDataType(fused_node, fuse_desc);
    FUSION_PASS_CHECK(
        !AutoRemoveInput(graph, fused_node,
                         fuse_desc->GetInputIndexByName("axes")),
        OP_LOGE(FUSEDNODE.c_str(), "remove input axes failed, fusion failed."),
        return GRAPH_FAILED);
    fuse_desc->DelAttr("begin");
    fuse_desc->DelAttr("end");
    fuse_desc->DelAttr("strides");
    fuse_desc->SetType("StridedSlice");  // AICPU Operator
    std::map<string, uint32_t> name_index_map = {{"x", 0}, {"begin", 1}, {"end", 2}, {"strides", 3}};
    fuse_desc->UpdateInputName(name_index_map);
    vector<string> infer_depends_vec = {"begin", "end", "strides"};
    fuse_desc->SetOpInferDepends(infer_depends_vec);
  } else {
    // remove input node as index descend
    for (int32_t index = 4; index > 0; index--) {
      FUSION_PASS_CHECK(
          !AutoRemoveInput(graph, fused_node, index),
          OP_LOGE(FUSEDNODE.c_str(), "remove input %s failed, fusion failed.",
                  fuse_desc->GetInputNameByIndex(index).c_str()),
          return GRAPH_FAILED);
    }
    fuse_desc->SetType("StridedSliceD");  // AICORE Operator
    std::map<string, uint32_t> name_index_map = {{"x", 0}};
    fuse_desc->UpdateInputName(name_index_map);
    ClearOpInferDepends(fused_node);
  }
  OP_LOGD(FUSEDNODE.c_str(), "SetType to [ %s ]", fuse_desc->GetType().c_str());
  fusion_node.push_back(fused_node);
  return SUCCESS;
}
REGISTER_PASS("ConstToAttrStridedSliceV2Fusion", BUILT_IN_GRAPH_PASS,
              ConstToAttrStridedSliceV2Pass);
}  // namespace fe
