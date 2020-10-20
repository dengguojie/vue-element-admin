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
#include "./util/error_util.h"
#include "util/util.h"

namespace ge {

namespace {
graphStatus MergeInferImpl(Operator& op) {
  GE_OP_LOGD(op.GetName().c_str(), "Begin to infer merge node shape");
  TensorDesc td = op.GetOutputDesc("value_index");
  TensorDesc td_y = op.GetOutputDesc("y");
  td.SetShape(ge::Shape());
  td.SetDataType(DT_INT32);
  auto ret = op.UpdateOutputDesc("value_index", td);
  if (ret != GRAPH_SUCCESS) {
    GE_OP_LOGE(op.GetName().c_str(), "Failed to update value_index tensor");
    return GRAPH_FAILED;
  }
  GE_OP_LOGD(op.GetName().c_str(), "Update value_index shape info ok");

  // check N of "x" >= 1
  size_t in_num = op.GetInputsSize();
  if (in_num < 1) {
    string reason = "inputs size[" + std::to_string(in_num) + "] must be greater than or equal to 1";
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), "input", reason);
    GE_OP_LOGE(op.GetName().c_str(), "Check inputs size[%zu] failed, it must be greater than or equal to 1 ", in_num);
    return GRAPH_FAILED;
  } else if (in_num == 2) {
    // Check is loop_merge, order of InferShape: Enter->Merge->NextIteration
    // So when processing InferShape on Merge op, shape & datatype of NextIteration op is set as default.
    // Therefore, shape & datatype of Merge op should be set as the Enter op.
    auto x0_type = op.GetDynamicInputDesc("x", 0).GetDataType();
    auto x0_dims = op.GetDynamicInputDesc("x", 0).GetShape().GetDims();
    bool not_handle_flag0 = (x0_type == DT_FLOAT) && (x0_dims.size() == 0);
    auto x1_type = op.GetDynamicInputDesc("x", 1).GetDataType();
    auto x1_dims = op.GetDynamicInputDesc("x", 1).GetShape().GetDims();
    bool not_handle_flag1 = (x1_type == DT_FLOAT) && (x1_dims.size() == 0);
    if ((x0_type != x1_type) && (not_handle_flag0 || not_handle_flag1)) {
      GE_OP_LOGD(
          op.GetName().c_str(),
          "Maybe is loop_merge node, input_0: data_type[%u] dim_count[%zu], input_1: data_type[%u] dim_count[%zu].",
          x0_type, x0_dims.size(), x1_type, x1_dims.size());
      if (not_handle_flag0) {
        td_y.SetShape(ge::Shape(x1_dims));
        td_y.SetDataType(x1_type);
      } else {
        td_y.SetShape(ge::Shape(x0_dims));
        td_y.SetDataType(x0_type);
      }
      (void)op.UpdateOutputDesc("y", td_y);
      return GRAPH_SUCCESS;
    }
  }

  // check "x" be same type
  auto x0_type = op.GetDynamicInputDesc("x", 0).GetDataType();
  for (size_t i = 1; i < op.GetInputsSize(); i++) {
    auto xi_type = op.GetDynamicInputDesc("x", i).GetDataType();
    if (xi_type != x0_type) {
      string reason = "x[0]'s dtype[" + std::to_string(x0_type) + "] must be equal to x[" + std::to_string(i) +
                      "]'s dtype[" + std::to_string(xi_type) + "]";
      GeInfershapeErrReport(op.GetName(), op.GetOpType(), "dtype", reason);
      GE_OP_LOGE(op.GetName().c_str(), "Check type failed: x[0]'s dtype[%u] must be equal to x[%zu]'s dtype[%u]",
                 x0_type, i, xi_type);
      return GRAPH_FAILED;
    }
  }

  // infer "y" be unknown shape
  auto x0_dims = op.GetDynamicInputDesc("x", 0).GetShape().GetDims();
  bool x0_unknown = (x0_dims.size() == 1) && (x0_dims[0] == 0);
  if (x0_unknown) {
    Shape unknown_shape(ge::UNKNOWN_SHAPE);
    td_y.SetShape(unknown_shape);
    td_y.SetDataType(x0_type);
    (void)op.UpdateOutputDesc("y", td_y);
    return GRAPH_SUCCESS;
  }

  // find the input with the max size from all inputs, and set it's data type/shape to the output
  GE_OP_LOGD(op.GetName().c_str(), "Begin to calculate the shape of merge node, input size %zu", op.GetInputsSize());
  std::map<int64_t, size_t> size_to_index;
  for (size_t i = 0; i < op.GetInputsSize(); i++) {
    auto xi_dims = op.GetDynamicInputDesc("x", i).GetShape().GetDims();
    bool xi_unknown = (xi_dims.size() == 1) && (xi_dims[0] == 0);
    if (xi_unknown) {
      GE_OP_LOGW(op.GetName().c_str(), "Check shape size failed: 0:%zu, %zu:%zu.", x0_dims.size(), i, xi_dims.size());
      continue;
    }
    int64_t size = static_cast<int64_t>(GetSizeByDataType(op.GetDynamicInputDesc("x", i).GetDataType()));
    if (size < 0) {
      GE_OP_LOGW(op.GetName().c_str(), "Invalid data type in input %zu.", i);
      continue;
    }

    if (!xi_dims.empty()) {
      for (auto& dim : xi_dims) {
        if (dim <= 0) {
          size = -1;
          GE_OP_LOGW(op.GetName().c_str(), "Invalid dim found %d", dim);
          break;
        }
        if (INT64_MAX / size < dim) {
          GeInfershapeErrReport(op.GetName(), op.GetOpType(), "dim", "the dim size is overflow");
          GE_OP_LOGE(op.GetName().c_str(), "The dim size is overflow");
          return GRAPH_FAILED;
        }
        size *= dim;
      }
      if (size < 0) {
        continue;
      }
    }

    GE_OP_LOGD(op.GetName().c_str(), "Input index %zu, size %ld", i, size);

    if (size_to_index.count(size) == 0) {
      size_to_index[size] = i;
    }
  }
  if (size_to_index.empty()) {
    GE_OP_LOGE(op.GetName().c_str(), "No valid input shape");
    return GRAPH_FAILED;
  }
  auto index = size_to_index.rbegin()->second;
  GE_OP_LOGD(op.GetName().c_str(), "The max size of index is %zu, size %ld, data type %u, dim count %zu", index,
             size_to_index.rbegin()->first, op.GetDynamicInputDesc("x", index).GetDataType(),
             op.GetDynamicInputDesc("x", index).GetShape().GetDims().size());
  td_y.SetShape(ge::Shape(op.GetDynamicInputDesc("x", index).GetShape().GetDims()));
  td_y.SetDataType(op.GetDynamicInputDesc("x", index).GetDataType());
  (void)op.UpdateOutputDesc("y", td_y);
  return GRAPH_SUCCESS;
}

graphStatus SwitchInferImpl(Operator& op) {
  // check "pred" scalar type be bool
  auto pred_dims = op.GetInputDesc("pred").GetShape().GetDims();
  if (pred_dims.size() != 0) {
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), "pred dims", "pred should be a scalar");
    GE_OP_LOGE(op.GetName().c_str(), "pred should be a scalar, actually size=%zu", pred_dims.size());
    return GRAPH_FAILED;
  }
  DataType pred_type = op.GetInputDesc("pred").GetDataType();
  if (pred_type != DT_BOOL) {
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), "dtype", "pred should be bool type");
    GE_OP_LOGE(op.GetName().c_str(), "pred should be bool type, actually type=%u", pred_type);
    return GRAPH_FAILED;
  }
  auto data_dims = op.GetInputDesc("data").GetShape().GetDims();
  TensorDesc output_false = op.GetOutputDesc("output_false");
  TensorDesc output_true = op.GetOutputDesc("output_true");
  output_false.SetShape(ge::Shape(data_dims));
  output_true.SetShape(ge::Shape(data_dims));
  DataType data_type = op.GetInputDesc("data").GetDataType();
  output_false.SetDataType(data_type);
  output_true.SetDataType(data_type);
  (void)op.UpdateOutputDesc("output_false", output_false);
  (void)op.UpdateOutputDesc("output_true", output_true);
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
  (void)op.UpdateOutputDesc(out_name, tensordesc_output);

  return GRAPH_SUCCESS;
}

graphStatus LoopCondInferImpl(Operator& op) {
  auto input_dims = op.GetInputDesc("x").GetShape().GetDims();
  if (input_dims.size() != 0) {
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), "x dims", "x should be a scalar");
    GE_OP_LOGE(op.GetName().c_str(), "x should be a scalar, actually size=%zu", input_dims.size());
    return GRAPH_FAILED;
  }
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(ge::Shape(input_dims));
  DataType input_type = op.GetInputDesc("x").GetDataType();
  if (input_type != DT_BOOL) {
    GeInfershapeErrReport(op.GetName(), op.GetOpType(), "dtype", "x should be bool type");
    GE_OP_LOGE(op.GetName().c_str(), "x should be bool type, actually type=%u", input_type);
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

  if (x_length > 8) {
    OP_LOGE(op.GetName().c_str(), "the length of x should be less than or equal to 8");
    OpsOneInputShapeErrReport(op.GetName().c_str(), "x", "the length of x should be less than or equal to 8");
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
