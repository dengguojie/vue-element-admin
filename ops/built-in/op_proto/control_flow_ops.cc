/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file control_flow_ops.cpp
 * \brief
 */
#include "inc/control_flow_ops.h"
#include "common/inc/op_log.h"
#include "common_shape_fns.h"
#include "util/error_util.h"
#include "util/util.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
namespace {
struct DimGroup {
  int64_t dim_value_min{INT64_MAX};
  int64_t dim_value_max{INT64_MIN};
  bool is_unknown_shape{false};
  bool is_max_unknown{false};
};

static graphStatus MergeAsMaxInput(Operator &op) {
  const auto x0_type = op.GetDynamicInputDesc("x", 0).GetDataType();
  const auto x0_dims = op.GetDynamicInputDesc("x", 0).GetShape().GetDims();

  // find the input with the max size from all inputs, and set it's data type/shape to the output
  OP_LOGD(op.GetName(), "Begin to calculate the shape of merge node, input size %zu", op.GetInputsSize());
  std::map<int64_t, size_t> size_to_index;
  for (size_t i = 0; i < op.GetInputsSize(); ++i) {
    // check "x" be same type.
    const auto xi_type = op.GetDynamicInputDesc("x", i).GetDataType();
    if (xi_type != x0_type) {
      string reason = "x[0]'s dtype[" + std::to_string(x0_type) + "] must be equal to x[" + std::to_string(i) +
                      "]'s dtype[" + std::to_string(xi_type) + "]";
      GeInfershapeErrReport(op.GetName(), op.GetOpType(), "dtype", reason);
      REPORT_INNER_ERROR("E19999", "[Node:%s] Check dtype failed, as %s", op.GetName().c_str(), reason.c_str());
      GE_OP_LOGE(op.GetName().c_str(), "[InferShape][Check] Check dtype failed, as %s", reason.c_str());
      return GRAPH_FAILED;
    }

    // find "x" the max dims.
    const auto xi_dims = op.GetDynamicInputDesc("x", i).GetShape().GetDims();
    bool xi_unknown = (xi_dims.size() == 1) && (xi_dims[0] == 0);
    if (xi_unknown) {
      OP_LOGW(op.GetName(), "Check shape size failed: 0:%zu, %zu:%zu.", x0_dims.size(), i, xi_dims.size());
      continue;
    }
    int64_t size = static_cast<int64_t>(GetSizeByDataType(op.GetDynamicInputDesc("x", i).GetDataType()));
    if (size < 0) {
      OP_LOGW(op.GetName(), "Invalid data type in input %zu.", i);
      continue;
    }

    if (!xi_dims.empty()) {
      for (auto& dim : xi_dims) {
        if (dim <= 0) {
          size = -1;
          OP_LOGW(op.GetName(), "Invalid dim found %d", dim);
          break;
        }
        if (size != 0 && INT64_MAX / size < dim) {
          std::string reason = "size of input " + std::to_string(i) + " overflow int64";
          GeInfershapeErrReport(op.GetName(), op.GetOpType(), "dim", reason);
          REPORT_INNER_ERROR("E19999", "[Node:%s] Check shape failed, as %s", op.GetName().c_str(), reason.c_str());
          GE_OP_LOGE(op.GetName().c_str(), "[InferShape][Check] Check shape failed, as %s", reason.c_str());
          return GRAPH_FAILED;
        }
        size *= dim;
      }
      if (size < 0) {
        continue;
      }
    }

    OP_LOGD(op.GetName(), "Input index %zu, size %ld", i, size);
    if (size_to_index.count(size) == 0) {
      size_to_index[size] = i;
    }
  }
  if (size_to_index.empty()) {
    REPORT_INNER_ERROR("E19999", "[Node:%s] Check valid shape failed, no valid input shape", op.GetName().c_str());
    GE_OP_LOGE(op.GetName().c_str(), "[InferShape][Check] Check valid shape failed, no valid input shape");
    return GRAPH_FAILED;
  }
  auto index = size_to_index.rbegin()->second;
  OP_LOGD(op.GetName(), "The max size of index is %zu, size %ld, data type %u, dim count %zu", index,
          size_to_index.rbegin()->first, op.GetDynamicInputDesc("x", index).GetDataType(),
          op.GetDynamicInputDesc("x", index).GetShape().GetDims().size());

  TensorDesc td_y = op.GetOutputDescByName("y");
  td_y.SetShape(ge::Shape(op.GetDynamicInputDesc("x", index).GetShape().GetDims()));
  td_y.SetDataType(op.GetDynamicInputDesc("x", index).GetDataType());
  return op.UpdateOutputDesc("y", td_y);
}

static graphStatus MergeAsRunInput(Operator &op, int merge_index) {
  size_t in_num = op.GetInputsSize();
  if (0 <= merge_index && static_cast<uint32_t>(merge_index) < in_num) {
    OP_LOGD(op.GetName(), "Update output shape by merge input index: %d", merge_index);
    TensorDesc td_x = op.GetDynamicInputDesc("x", merge_index);
    (void)td_x.SetShapeRange({});
    return op.UpdateOutputDesc("y", td_x);
  }
  string reason = "specific merge index[" + std::to_string(merge_index) + "] not in range [0, " +
                  std::to_string(in_num) + ")";
  GeInfershapeErrReport(op.GetName(), op.GetOpType(), "input", reason);
  REPORT_INNER_ERROR("E19999", "[Node:%s] Check input index failed, as %s", op.GetName().c_str(), reason.c_str());
  GE_OP_LOGE(op.GetName().c_str(), "[InferShapeWhenRun][Check] Check input index failed, as %s", reason.c_str());
  return GRAPH_FAILED;
}

graphStatus MergeInferImpl(Operator &op) {
  size_t in_num = op.GetInputsSize();
  OP_LOGD(op.GetName(), "Begin to infer merge node shape, input size %zu", in_num);
  // Check N of "x" >= 1
  if (in_num < 1) {
    string reason = "input num should >= 1, actually input_num=" + std::to_string(in_num);
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), "input", reason);
    REPORT_INNER_ERROR("E19999", "[Node:%s] Check input num failed, as %s", op.GetName().c_str(), reason.c_str());
    GE_OP_LOGE(op.GetName().c_str(), "[InferShape][Check] Check input num failed, as %s", reason.c_str());
    return GRAPH_FAILED;
  }

  TensorDesc td_v = op.GetOutputDescByName("value_index");
  td_v.SetShape(Shape());
  td_v.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("value_index", td_v) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "[Node:%s] Update output desc of value_index failed", op.GetName().c_str());
    GE_OP_LOGE(op.GetName().c_str(), "[InferShape][Update] Update output desc of value_index failed");
    return GRAPH_FAILED;
  }

  // For dynamic shape running calculation.
  int merge_index = -1;
  if (op.GetAttr("_merge_input_index", merge_index) == GRAPH_SUCCESS) {
    return MergeAsRunInput(op, merge_index);
  }

  // For dynamic multi batch.
  bool is_multi_batch = false;
  if (op.GetAttr(ATTR_INSERT_BY_MBATCH, is_multi_batch) == GRAPH_SUCCESS) {
    return MergeAsMaxInput(op);
  }

  // check N of "x" == 2
  const auto &node = NodeUtils::GetNodeFromOperator(op);
  if (in_num == 2 && node != nullptr) {
    // Check is while_loop, order of InferShape: Enter -> Merge -> Switch -> NextIteration -> Merge -> Switch -> Exit
    // So when processing InferShape on Merge op, shape & datatype of NextIteration op is set as default.
    // Therefore, shape & datatype of Merge op should be set as the Enter op.
    const auto &node_x1 = node->GetInDataNodes().at(1);   // NextIteration
    if (node_x1->GetType() == "NextIteration" || node_x1->GetType() == "RefNextIteration") {
      bool need_infer_again = false;
      if (op.GetAttr(ATTR_NAME_NEED_INFER_AGAIN, need_infer_again) != GRAPH_SUCCESS) {  // first time infer.
        OP_LOGD(op.GetName(), "Update output shape by merge input enter");
        TensorDesc td_x = op.GetDynamicInputDesc("x", 0);
        return op.UpdateOutputDesc("y", td_x);
      }
    }
  }

  auto x0_type = op.GetDynamicInputDesc("x", 0).GetDataType();
  auto x0_dims = op.GetDynamicInputDesc("x", 0).GetShape().GetDims();
  std::vector<std::pair<int64_t, int64_t>> x0_shape_range;
  op.GetDynamicInputDesc("x", 0).GetShapeRange(x0_shape_range);
  auto td_y = op.GetOutputDescByName("y");
  td_y.SetDataType(x0_type);

  // Infer "y": Find the input with the max size from all inputs, and set it`s data type/shape to the output
  OP_LOGD(op.GetName(), "Begin to calculate the shape of merge node, input size: %zu", in_num);
  std::vector<DimGroup> dims_to_range(x0_dims.size());
  for (size_t i = 0; i < in_num; ++i) {
    // check "x" be same data type
    const auto xi_type = op.GetDynamicInputDesc("x", i).GetDataType();
    if (xi_type != x0_type) {
      string reason = "x[0]'s dtype[" + std::to_string(x0_type) + "] must be equal to x[" + std::to_string(i) +
                      "]'s dtype["  + std::to_string(xi_type) + "]";
      GeInfershapeErrReport(op.GetName(), op.GetOpType(), "dtype", reason);
      REPORT_INNER_ERROR("E19999", "[Node:%s] Check dtype failed, as %s", op.GetName().c_str(), reason.c_str());
      GE_OP_LOGE(op.GetName().c_str(), "[InferShape][Check] Check dtype failed, as %s", reason.c_str());
      return GRAPH_FAILED;
    }

    const auto xi_dims = op.GetDynamicInputDesc("x", i).GetShape().GetDims();
    if (x0_dims.size() != xi_dims.size() || xi_dims == UNKNOWN_RANK) {
      OP_LOGW(op.GetName(), "Check dims failed: x[0]'s dim size[%zu] not equal to x[%zu]'s dim size[%zu]",
              x0_dims.size(), i, xi_dims.size());
      td_y.SetShape(Shape(UNKNOWN_RANK));
      return op.UpdateOutputDesc("y", td_y);
    }

    std::vector<std::pair<int64_t, int64_t>> xi_range;
    op.GetDynamicInputDesc("x", i).GetShapeRange(xi_range);
    for (size_t j = 0; j < xi_dims.size(); ++j) {
      DimGroup &dim_group = dims_to_range[j];
      if (xi_dims[j] >= 0) {
        if (xi_dims[j] < dim_group.dim_value_min) {
          dim_group.dim_value_min = xi_dims[j];
        }
        if (xi_dims[j] > dim_group.dim_value_max) {
          dim_group.dim_value_max = xi_dims[j];
        }
      } else {
        dim_group.is_unknown_shape = true;
      }

      if (j < xi_range.size()) {
        if (dim_group.dim_value_min > xi_range[j].first) {
          dim_group.dim_value_min = xi_range[j].first;
        }
        if (dim_group.dim_value_max < xi_range[j].second) {
          dim_group.dim_value_max = xi_range[j].second;
        }
        if (xi_range[j].second < 0) {
          dim_group.is_unknown_shape = true;
          dim_group.is_max_unknown = true;
        }
      }
    }
  }

  // calculation final shape
  bool is_unknown_shape = false;
  std::vector<int64_t> out_dims(x0_dims.size());
  std::vector<std::pair<int64_t, int64_t>> out_shape_range(x0_dims.size());
  for (size_t i = 0; i < dims_to_range.size(); ++i) {
    DimGroup &dim_group = dims_to_range[i];
    if (dim_group.is_unknown_shape) {
      is_unknown_shape = true;
      out_dims[i] = -1;
    } else {
      if (dim_group.dim_value_min == dim_group.dim_value_max) {
        out_dims[i] = dim_group.dim_value_min;
      } else {
        is_unknown_shape = true;
        out_dims[i] = -1;
      }
    }

    out_shape_range[i].first = dim_group.dim_value_min < 0 ? 0 : dim_group.dim_value_min;
    out_shape_range[i].second = dim_group.is_max_unknown ? -1 : dim_group.dim_value_max;
  }

  // Update output shape
  td_y.SetShape(Shape(out_dims));
  if (is_unknown_shape) {
    (void)td_y.SetShapeRange(out_shape_range);
  }
  OP_LOGD(op.GetName(), "data type: %u, dims size %zu, is unknown shape: %s",
          x0_type, out_dims.size(), is_unknown_shape ? "Yes" : "No");
  return op.UpdateOutputDesc("y", td_y);
}

graphStatus SwitchInferImpl(Operator& op) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  auto data_desc = op_desc->MutableInputDesc("data");
  auto pred_desc = op_desc->MutableInputDesc("pred");
  auto output_false_desc = op_desc->MutableOutputDesc("output_false");
  auto output_true_desc = op_desc->MutableOutputDesc("output_true");

  std::vector<std::pair<int64_t, int64_t>> data_range;
  data_desc->GetShapeRange(data_range);
  // check "pred" scalar type be bool
  auto pred_dims = pred_desc->GetShape().GetDims();
  if (pred_dims.size() != 0) {
    string reason = "input pred should be a scalar, actually rank=" + std::to_string(pred_dims.size());
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), "pred dims", reason);
    REPORT_INNER_ERROR("E19999", "[Node:%s] Check shape rank failed, as %s", op.GetName().c_str(), reason.c_str());
    GE_OP_LOGE(op.GetName().c_str(), "[InferShape][Check] Check shape rank failed, as %s", reason.c_str());
    return GRAPH_FAILED;
  }
  DataType pred_type = pred_desc->GetDataType();
  if (pred_type != DT_BOOL) {
    string reason = "input pred should be DT_BOOL, actually is " + DataTypeToStringDesc(pred_type);
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), "dtype", reason);
    REPORT_INNER_ERROR("E19999", "[Node:%s] Check dtype failed, as %s", op.GetName().c_str(), reason.c_str());
    GE_OP_LOGE(op.GetName().c_str(), "[InferShape][Check] Check dtype failed, as %s", reason.c_str());
    return GRAPH_FAILED;
  }

  DataType data_type = data_desc->GetDataType();
  auto data_dims = data_desc->GetShape().GetDims();

  output_false_desc->SetShapeRange(data_range);
  output_true_desc->SetShapeRange(data_range);
  output_false_desc->SetShape(GeShape(data_dims));
  output_false_desc->SetOriginShape(GeShape(data_dims));
  output_true_desc->SetShape(GeShape(data_dims));
  output_true_desc->SetOriginShape(GeShape(data_dims));
  output_false_desc->SetDataType(data_type);
  output_true_desc->SetDataType(data_type);

  auto context = op.GetInferenceContext();
  std::vector<std::vector<ShapeAndType>> in_shapes_and_types = context->GetInputHandleShapesAndTypes();
  if ((!in_shapes_and_types.empty()) && (!in_shapes_and_types.at(0).empty())) {
    ShapeAndType shape_and_type = in_shapes_and_types.at(0).at(0);
    std::vector<ShapeAndType> grad_handle_shape_and_type;
    grad_handle_shape_and_type.reserve(1);
    grad_handle_shape_and_type.emplace_back(shape_and_type);

    std::vector<std::vector<ShapeAndType>> shapes_and_types(2);
    shapes_and_types[0] = grad_handle_shape_and_type;
    shapes_and_types[1] = grad_handle_shape_and_type;
    context->SetOutputHandleShapesAndTypes(shapes_and_types);
  }

  return GRAPH_SUCCESS;
}

graphStatus EnterInferImpl(Operator& op) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc_x = op_desc->MutableInputDesc("x");
  auto output_desc_y = op_desc->MutableOutputDesc("y");
  std::vector<std::pair<int64_t, int64_t>> x_range;
  std::vector<std::pair<int64_t, int64_t>> y_range;
  input_desc_x->GetShapeRange(x_range);

  auto input_dims = input_desc_x->MutableShape().GetDims();
  DataType input_type = input_desc_x->GetDataType();
  output_desc_y->SetShape(ge::GeShape(input_dims));
  output_desc_y->SetOriginShape(ge::GeShape(input_dims));
  output_desc_y->SetDataType(input_type);

  if (!x_range.empty()) {
    output_desc_y->SetShapeRange(x_range);
  }

  auto context = op.GetInferenceContext();
  std::vector<std::vector<ShapeAndType>> in_shapes_and_types = context->GetInputHandleShapesAndTypes();
  if ((!in_shapes_and_types.empty()) && (!in_shapes_and_types.at(0).empty())) {
    ShapeAndType shape_and_type = in_shapes_and_types.at(0).at(0);
    std::vector<ShapeAndType> grad_handle_shape_and_type;
    grad_handle_shape_and_type.reserve(1);
    grad_handle_shape_and_type.emplace_back(shape_and_type);

    std::vector<std::vector<ShapeAndType>> shapes_and_types(1);
    shapes_and_types[0] = grad_handle_shape_and_type;
    context->SetOutputHandleShapesAndTypes(shapes_and_types);
  }

  return GRAPH_SUCCESS;
}

graphStatus PassThroughInferImpl(Operator& op, const std::string& in_name, const std::string& out_name) {
  auto input_dims = op.GetInputDesc(in_name).GetShape().GetDims();
  DataType input_type = op.GetInputDesc(in_name).GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc(out_name);
  tensordesc_output.SetShape(ge::Shape(input_dims));
  tensordesc_output.SetDataType(input_type);
  std::vector<std::pair<int64_t, int64_t>> input_range;
  (void)op.GetInputDesc(in_name).GetShapeRange(input_range);
  (void)tensordesc_output.SetShapeRange(input_range);
  (void)op.UpdateOutputDesc(out_name, tensordesc_output);

  return GRAPH_SUCCESS;
}

graphStatus LoopCondInferImpl(Operator& op) {
  auto input_dims = op.GetInputDesc("x").GetShape().GetDims();
  if (input_dims.size() != 0) {
    string reason = "input x should be a scalar, actually rank=" + std::to_string(input_dims.size());
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), "pred dims", reason);
    REPORT_INNER_ERROR("E19999", "[Node:%s] Check shape rank failed, as %s", op.GetName().c_str(), reason.c_str());
    GE_OP_LOGE(op.GetName().c_str(), "[InferShape][Check] Check shape rank failed, as %s", reason.c_str());
    return GRAPH_FAILED;
  }
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(ge::Shape(input_dims));
  DataType input_type = op.GetInputDesc("x").GetDataType();
  if (input_type != DT_BOOL) {
    string reason = "input x should be DT_BOOL, actually is " + DataTypeToStringDesc(input_type);
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), "dtype", reason);
    REPORT_INNER_ERROR("E19999", "[Node:%s] Check dtype failed, as %s", op.GetName().c_str(), reason.c_str());
    GE_OP_LOGE(op.GetName().c_str(), "[InferShape][Check] Check dtype failed, as %s", reason.c_str());
    return GRAPH_FAILED;
  }
  tensordesc_output.SetDataType(input_type);
  (void)op.UpdateOutputDesc("y", tensordesc_output);

  return GRAPH_SUCCESS;
}

}  // namespace

IMPLEMT_INFERFUNC(Merge, MergeInfer) {
  return MergeInferImpl(op);
}

INFER_FUNC_REG(Merge, MergeInfer);

IMPLEMT_INFERFUNC(RefMerge, RefMergeInfer) {
  return MergeInferImpl(op);
}

INFER_FUNC_REG(RefMerge, RefMergeInfer);

IMPLEMT_INFERFUNC(Switch, SwitchInfer) {
  return SwitchInferImpl(op);
}

INFER_FUNC_REG(Switch, SwitchInfer);

IMPLEMT_INFERFUNC(RefSwitch, RefSwitchInfer) {
  return SwitchInferImpl(op);
}

INFER_FUNC_REG(RefSwitch, RefSwitchInfer);

IMPLEMT_INFERFUNC(SwitchN, SwitchNInfer) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SwitchN, SwitchNInfer);

IMPLEMT_INFERFUNC(Enter, EnterInfer) {
  return EnterInferImpl(op);
}

INFER_FUNC_REG(Enter, EnterInfer);

IMPLEMT_INFERFUNC(RefEnter, RefEnterInfer) {
  return PassThroughInferImpl(op, "x", "y");
}

INFER_FUNC_REG(RefEnter, RefEnterInfer);

IMPLEMT_INFERFUNC(LoopCond, LoopCondInfer) {
  return LoopCondInferImpl(op);
}

INFER_FUNC_REG(LoopCond, LoopCondInfer);

IMPLEMT_INFERFUNC(NextIteration, NextIterationInfer) {
  return PassThroughInferImpl(op, "x", "y");
}

INFER_FUNC_REG(NextIteration, NextIterationInfer);

IMPLEMT_INFERFUNC(RefNextIteration, RefNextIterationInfer) {
  return PassThroughInferImpl(op, "x", "y");
}

INFER_FUNC_REG(RefNextIteration, RefNextIterationInfer);

IMPLEMT_INFERFUNC(Exit, ExitInfer) {
  return PassThroughInferImpl(op, "x", "y");
}

INFER_FUNC_REG(Exit, ExitInfer);

IMPLEMT_INFERFUNC(RefExit, RefExitInfer) {
  return PassThroughInferImpl(op, "x", "y");
}

INFER_FUNC_REG(RefExit, RefExitInfer);

// ----------------MapIndex-------------------
IMPLEMT_VERIFIER(MapIndex, MapIndexVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(MapIndexInferShape) {
  OP_LOGI("MapIndex", "infer shape begin---");
  auto x_shape = op.GetInputDesc("x").GetShape().GetDims();
  if (x_shape.empty()) {
    OP_LOGE(op.GetName().c_str(), "x_shape is empty");
    OpsOneInputShapeErrReport(op.GetName().c_str(), "x", "x_shape is empty");
    return GRAPH_FAILED;
  }
  int64_t x_length = x_shape[0];

  auto data_seq_shape = op.GetInputDesc("data_seq").GetShape().GetDims();
  if (data_seq_shape.empty()) {
    OP_LOGE(op.GetName().c_str(), "data_seq_shape is empty");
    OpsOneInputShapeErrReport(op.GetName().c_str(), "data_seq", "data_seq_shape is empty");
    return GRAPH_FAILED;
  }
  int64_t data_seq_length = data_seq_shape[0];

  if (x_length > 128 || x_length <= 0) {
    OP_LOGE(op.GetName().c_str(), "the length of x should be less than or equal to 128");
    OpsOneInputShapeErrReport(op.GetName().c_str(), "x",
                              "the length of x should be less than or equal to 128 and not 0");
    return GRAPH_FAILED;
  }

  if (data_seq_length % x_length != 0) {
    OP_LOGE(op.GetName().c_str(), "the length of data_seq must be multiple of the length of x");
    OpsTwoInputShapeErrReport(op.GetName().c_str(), "data_seq", "x",
                              "the length of data_seq must be multiple of the length of x");
    return GRAPH_FAILED;
  }

  if (data_seq_length / x_length > 100) {
    OP_LOGE(op.GetName().c_str(), "data_seq_length / x_length should be be less than or equal to 100");
    OpsTwoInputShapeErrReport(op.GetName().c_str(), "data_seq", "x",
                              "data_seq_length / x_length should be be less than or equal to 100");
    return GRAPH_FAILED;
  }

  auto level_index_shape = op.GetInputDesc("level_index").GetShape().GetDims();
  if (!level_index_shape.empty()) {
    int64_t level_index_length = level_index_shape[0];
    if (level_index_length != (data_seq_length / x_length)) {
      OP_LOGE(op.GetName().c_str(),
              "the length of level_index must be equal to "
              "the length of data_seq divided by the length of x");
      OpsOneInputShapeErrReport(op.GetName().c_str(), "level_index",
                                "the length of level_index must be equal to "
                                "the length of data_seq divided by the length of x");
      return GRAPH_FAILED;
    }
  }

  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(ge::Shape());
  y_desc.SetDataType(ge::DT_INT32);

  (void)op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(MapIndex, MapIndexInferShape);
VERIFY_FUNC_REG(MapIndex, MapIndexVerify);

}  // namespace ge
