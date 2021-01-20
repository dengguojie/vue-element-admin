/**
 * Copyright 2018 Huawei Technologies Co., Ltd
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
 * \file data_flow_ops.cpp
 * \brief
 */
#include "inc/data_flow_ops.h"
#include <utility>
#include "graph/operator.h"
#include "op_log.h"
#include "util/common_shape_fns.h"
#include "util/util.h"
#include "util/error_util.h"
#include "securec.h"
#include <regex>

namespace ge {
static std::map<std::string, std::vector<ShapeAndType>> shape_and_type_map;
namespace {
graphStatus SetAttrsToShapesAndTypes(Operator& op,
                                     const std::string& dtypes,
                                     const std::string& shapes) {
  std::vector<DataType> elem_types;
  if (op.GetAttr(dtypes, elem_types) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), dtypes);
    OP_LOGE(op.GetName().c_str(), "Get attr [%s] failed.", dtypes.c_str());
    return GRAPH_FAILED;
  }
  Operator::OpListListInt elem_shapes;
  if (op.GetAttr(shapes, elem_shapes) == GRAPH_SUCCESS) {
    size_t num = std::min(elem_shapes.size(), elem_types.size());
    std::vector<ShapeAndType> handle_shapes_and_types;
    handle_shapes_and_types.reserve(num);

    for (size_t i = 0; i < num; ++i) {
      Shape elem_shape(std::move(elem_shapes[i]));
      DataType elem_type(std::move(elem_types[i]));
      ShapeAndType shape_and_type(elem_shape, elem_type);
      handle_shapes_and_types.emplace_back(std::move(shape_and_type));
    }

    std::vector<std::vector<ShapeAndType>> shapes_and_types(2);
    shapes_and_types[0] = handle_shapes_and_types;
    auto context = op.GetInferenceContext();
    context->SetOutputHandleShapesAndTypes(shapes_and_types);
  }
  return GRAPH_SUCCESS;
}

graphStatus InferMapShapes(Operator& op,
                           const std::string& dtypes,
                           const std::string& out_name) {
  std::vector<ge::DataType> list_type;
  if (op.GetAttr(dtypes, list_type) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), dtypes);
    OP_LOGE(op.GetName().c_str(), "Get attr [%s] failed.", dtypes.c_str());
    return GRAPH_FAILED;
  }

  for (size_t i = 0; i < list_type.size(); ++i) {
    TensorDesc output_desc = op.GetDynamicOutputDesc(out_name, i);
    output_desc.SetShape(Shape(ge::UNKNOWN_RANK));
    output_desc.SetDataType(list_type[i]);
    if (op.UpdateDynamicOutputDesc(out_name, i, output_desc) != GRAPH_SUCCESS) {
      OpsOPUpdateErrReport(op.GetName(), out_name);
      OP_LOGE(op.GetName().c_str(),
              "Update [%s:%zu] desc failed.", out_name.c_str(), i);
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus DequeueManyShape(Operator& op,
                             const Shape& n_shape,
                             const std::string& out_name) {
  auto operator_context = op.GetInferenceContext();
  std::vector<std::vector<ShapeAndType>> handle_shapes_and_types;
  handle_shapes_and_types = operator_context->GetInputHandleShapesAndTypes();

  std::vector<ge::DataType> component_types;
  if (op.GetAttr("component_types", component_types) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "component_types");
    OP_LOGE(op.GetName().c_str(), "Get attr [component_types] failed.");
    return GRAPH_FAILED;
  }

  graphStatus output_status = GRAPH_SUCCESS;
  size_t num_outputs_data = component_types.size();
  if ((handle_shapes_and_types.size() != 0) &&
      (handle_shapes_and_types[0].size() != 0) &&
      (handle_shapes_and_types[0].size() == num_outputs_data)) {
    for (size_t i = 0; i < handle_shapes_and_types[0].size(); ++i) {
      Shape comibined_shape;
      Shape handle_shape = handle_shapes_and_types[0][i].GetShape();
      graphStatus concatenate_status = Concatenate(n_shape,
                                                   handle_shape,
                                                   comibined_shape);
      if (concatenate_status != GRAPH_SUCCESS) {
        InferShapeOtherErrReport(op.GetName(), "concatenate failed");
        OP_LOGE(op.GetName().c_str(), "Concatenate failed.");
        return GRAPH_FAILED;
      }
      TensorDesc output_desc = op.GetDynamicOutputDesc(out_name, i);
      output_desc.SetShape(comibined_shape);
      output_desc.SetDataType(component_types[i]);
      output_status = op.UpdateDynamicOutputDesc(out_name, i, output_desc);
      if (output_status != GRAPH_SUCCESS) {
        OpsOPUpdateErrReport(op.GetName(), out_name);
        OP_LOGE(op.GetName().c_str(), "Update [%s:%d] desc failed.",
                out_name.c_str(), i);
        return GRAPH_FAILED;
      }
    }
  } else {
    for (size_t i = 0; i < num_outputs_data; ++i) {
      TensorDesc output_desc = op.GetDynamicOutputDesc(out_name, i);
      output_desc.SetShape(Shape(ge::UNKNOWN_RANK));
      output_desc.SetDataType(component_types[i]);
      output_status = op.UpdateDynamicOutputDesc(out_name, i, output_desc);
      if (output_status != GRAPH_SUCCESS) {
        OpsOPUpdateErrReport(op.GetName(), out_name);
        OP_LOGE(op.GetName().c_str(), "Update [%s:%d] desc failed.",
                out_name.c_str(), i);
        return GRAPH_FAILED;
      }
    }
  }
  return GRAPH_SUCCESS;
}
}  // namespace

IMPLEMT_INFERFUNC(QueueIsClosed, QueueIsClosedInfer) {
  Shape scalar_shape;
  (void)Scalar(scalar_shape);
  TensorDesc output_desc = op.GetOutputDesc("is_closed");
  output_desc.SetShape(scalar_shape);
  output_desc.SetDataType(DT_BOOL);
  if (op.UpdateOutputDesc("is_closed", output_desc) != GRAPH_SUCCESS) {
    OpsOPUpdateErrReport(op.GetName(), "is_closed");
    OP_LOGE(op.GetName().c_str(), "Update [is_closed] desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(QueueIsClosed, QueueIsClosedInfer);

IMPLEMT_INFERFUNC(QueueSize, QueueSizeInfer) {
  Shape input_shape = op.GetInputDesc("handle").GetShape();
  TensorDesc size_desc = op.GetOutputDesc("size");
  size_desc.SetShape(input_shape);
  size_desc.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("size", size_desc) != GRAPH_SUCCESS) {
    OpsOPUpdateErrReport(op.GetName(), "size");
    OP_LOGE(op.GetName().c_str(), "Update [size] desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(QueueSize, QueueSizeInfer);

IMPLEMT_INFERFUNC(FIFOQueue, FIFOQueueInfer) {
  TensorDesc output_desc = op.GetOutputDesc("handle");
  DataType output_type = DT_RESOURCE;
  output_desc.SetShape(Shape());
  output_desc.SetDataType(output_type);
  if (op.UpdateOutputDesc("handle", output_desc) != GRAPH_SUCCESS) {
    OpsOPUpdateErrReport(op.GetName(), "handle");
    OP_LOGE(op.GetName().c_str(), "Update [handle] desc failed.");
    return GRAPH_FAILED;
  }
  return SetAttrsToShapesAndTypes(op, "component_types", "shapes");
}

INFER_FUNC_REG(FIFOQueue, FIFOQueueInfer);

IMPLEMT_INFERFUNC(QueueEnqueue, QueueEnqueueInfer) {
  auto context = op.GetInferenceContext();
  std::vector<std::vector<ShapeAndType>> input_shapes_and_types;
  input_shapes_and_types = context->GetInputHandleShapesAndTypes();
  size_t components_size = op.GetInputsSize() - 1;
  if ((input_shapes_and_types.size() != 0) &&
      (input_shapes_and_types[0].size() != 0) &&
      (input_shapes_and_types[0].size() == components_size)) {
    return GRAPH_SUCCESS;
  }

  std::vector<ShapeAndType> handle_shapes_and_types;
  handle_shapes_and_types.reserve(components_size);
  for (size_t i = 0; i < components_size; ++i) {
    const TensorDesc input_desc = op.GetDynamicInputDesc("components", i);
    Shape elem_shape(input_desc.GetShape());
    DataType elem_type(input_desc.GetDataType());
    ShapeAndType shape_and_type(elem_shape, elem_type);
    handle_shapes_and_types.emplace_back(std::move(shape_and_type));
  }
  std::vector<std::vector<ShapeAndType>> output_shapes_and_types(2);
  output_shapes_and_types[0] = handle_shapes_and_types;
  context->SetOutputHandleShapesAndTypes(output_shapes_and_types);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(QueueEnqueue, QueueEnqueueInfer);

IMPLEMT_INFERFUNC(QueueEnqueueMany, QueueEnqueueManyInfer) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(QueueEnqueueMany, QueueEnqueueManyInfer);

IMPLEMT_INFERFUNC(QueueDequeue, QueueDequeueInfer) {
  auto operator_context = op.GetInferenceContext();
  std::vector<std::vector<ShapeAndType>> handle_shapes_and_types;
  handle_shapes_and_types = operator_context->GetInputHandleShapesAndTypes();

  std::vector<ge::DataType> component_types;
  if (op.GetAttr("component_types", component_types) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "component_types");
    OP_LOGE(op.GetName().c_str(), "Get attr [component_types] failed.");
    return GRAPH_FAILED;
  }

  graphStatus output_status = GRAPH_SUCCESS;
  size_t num_outputs_data = component_types.size();
  if ((handle_shapes_and_types.size() != 0) &&
      (handle_shapes_and_types[0].size() != 0) &&
      (handle_shapes_and_types[0].size() == num_outputs_data)) {
    for (size_t i = 0; i < handle_shapes_and_types[0].size(); ++i) {
      TensorDesc output_desc = op.GetDynamicOutputDesc("components", i);
      output_desc.SetShape(handle_shapes_and_types[0][i].GetShape());
      output_desc.SetDataType(component_types[i]);
      output_status = op.UpdateDynamicOutputDesc("components", i, output_desc);
      if (output_status != GRAPH_SUCCESS) {
        OpsOPUpdateErrReport(op.GetName(), "components");
        OP_LOGE(op.GetName().c_str(), "Update [%s:%d] desc failed.",
                "components", i);
        return GRAPH_FAILED;
      }
    }
  } else {
    for (size_t i = 0; i < num_outputs_data; ++i) {
      TensorDesc output_desc = op.GetDynamicOutputDesc("components", i);
      output_desc.SetShape(Shape(ge::UNKNOWN_RANK));
      output_desc.SetDataType(component_types[i]);
      output_status = op.UpdateDynamicOutputDesc("components", i, output_desc);
      if (output_status != GRAPH_SUCCESS) {
        OpsOPUpdateErrReport(op.GetName(), "components");
        OP_LOGE(op.GetName().c_str(), "Update [%s:%d] desc failed.",
                "components", i);
        return GRAPH_FAILED;
      }
    }
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(QueueDequeue, QueueDequeueInfer);

IMPLEMT_INFERFUNC(QueueDequeueMany, QueueDequeueManyInfer) {
  Tensor n_tensor;
  Shape n_shape;
  if (op.GetInputConstData("n", n_tensor) == GRAPH_SUCCESS) {
    const uint8_t* n = n_tensor.GetData();
    const int32_t* n_data = reinterpret_cast<const int32_t*>(n);
    if (*n_data < 0) {
      OpsAttrValueErrReport(op.GetName(), "n", ">=0", std::to_string(*n_data));
      OP_LOGE(op.GetName().c_str(), "Input [n] must >= 0, but got [%d].",
              *n_data);
      return GRAPH_FAILED;
    }
    n_shape = Shape({*n_data});
  } else {
    n_shape = Shape({ge::UNKNOWN_DIM});
  }
  return DequeueManyShape(op, n_shape, "components");
}

INFER_FUNC_REG(QueueDequeueMany, QueueDequeueManyInfer);

IMPLEMT_INFERFUNC(QueueDequeueUpTo, QueueDequeueUpToInfer) {
  Shape n_shape({ge::UNKNOWN_DIM});
  return DequeueManyShape(op, n_shape, "components");
}

INFER_FUNC_REG(QueueDequeueUpTo, QueueDequeueUpToInfer);

IMPLEMT_INFERFUNC(Stage, StageInfer) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Stage, StageInfer);

IMPLEMT_INFERFUNC(StageClear, StageClearInfer) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StageClear, StageClearInfer);

IMPLEMT_INFERFUNC(StagePeek, StagePeekInfer) {
  Shape unknown_shape(ge::UNKNOWN_SHAPE);

  Operator::OpListType dtypes;
  if (op.GetAttr("dtypes", dtypes) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Op get attr dtypes failed");
    return GRAPH_FAILED;
  }

  size_t size = dtypes.size();
  for (size_t i = 0; i < size; ++i) {
    TensorDesc output_desc = op.GetDynamicOutputDesc("y", i);
    output_desc.SetShape(unknown_shape);
    output_desc.SetDataType(dtypes[i]);
    op.UpdateDynamicOutputDesc("y", i, output_desc);
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StagePeek, StagePeekInfer);

IMPLEMT_INFERFUNC(StageSize, StageSizeInfer) {
  Shape out;
  (void)Scalar(out);

  TensorDesc output_desc = op.GetOutputDesc("size");
  output_desc.SetShape(out);
  output_desc.SetDataType(DT_INT32);
  return op.UpdateOutputDesc("size", output_desc);
}

INFER_FUNC_REG(StageSize, StageSizeInfer);

IMPLEMT_INFERFUNC(StackPop, StackPopInfer) {
  auto operator_context = op.GetInferenceContext();
  Shape unknown_shape(ge::UNKNOWN_SHAPE);

  DataType type;
  if (op.GetAttr("elem_type", type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Op get attr elem_type failed");
    return GRAPH_FAILED;
  }

  TensorDesc output_desc = op.GetOutputDesc("element");
  output_desc.SetShape(unknown_shape);

  if (operator_context->GetMarks().size() != 0) {
    bool is_set_unknown = false;
    std::string stack_name = operator_context->GetMarks()[0];
    std::vector<std::vector<int64_t>> shape_vec;
    for (auto elem : shape_and_type_map[stack_name]) {
      auto shape = elem.GetShape().GetDims();
      shape_vec.push_back(shape);
    }
    if (shape_vec.size() == 0) {
      OP_LOGE(op.GetName().c_str(), "stack is null");
      return GRAPH_FAILED;
    }
    std::vector<int64_t> val = shape_vec[0];
    auto unknown_cnt = std::count(shape_vec.begin(), shape_vec.end(), ge::UNKNOWN_SHAPE);
    auto same_cnt = static_cast<size_t>(std::count(shape_vec.begin(), shape_vec.end(), val));
    if (unknown_cnt != 0 || (same_cnt != shape_vec.size())) {
      is_set_unknown = true;
    }
    if (is_set_unknown) {
      output_desc.SetShape(unknown_shape);
    } else {
      output_desc.SetShape(shape_and_type_map[stack_name][0].GetShape());
    }
  }

  output_desc.SetDataType(type);
  return op.UpdateOutputDesc("element", output_desc);
}

INFER_FUNC_REG(StackPop, StackPopInfer);

IMPLEMT_INFERFUNC(StackPush, StackPushInfer) {
  auto operator_context = op.GetInferenceContext();
  Shape out = op.GetInputDesc("element").GetShape();

  DataType type = op.GetInputDesc("element").GetDataType();
  ShapeAndType shape_and_type(out, type);

  if (operator_context->GetMarks().size() != 0) {
    // get stack name
    std::string stack_name = operator_context->GetMarks()[0];
    shape_and_type_map[stack_name].emplace_back(shape_and_type);
  }

  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(out);
  output_desc.SetDataType(type);
  return op.UpdateOutputDesc("y", output_desc);
}

INFER_FUNC_REG(StackPush, StackPushInfer);

IMPLEMT_INFERFUNC(StackClose, StackCloseInfer) {
  Shape shape;
  if (WithRank(op.GetInputDesc("handle"), 1, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input resource handle's rank must be 1");
    return GRAPH_PARAM_INVALID;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StackClose, StackCloseInfer);

IMPLEMT_INFERFUNC(Stack, StackInfer) {
  Shape out;
  (void)Vector(2, out);
  auto operator_context = op.GetInferenceContext();
  std::vector<std::string> marks = {op.GetName()};

  operator_context->SetMarks(marks);
  TensorDesc output_desc = op.GetOutputDesc("handle");
  output_desc.SetShape(out);
  output_desc.SetDataType(DT_RESOURCE);
  return op.UpdateOutputDesc("handle", output_desc);
}

INFER_FUNC_REG(Stack, StackInfer);

IMPLEMT_INFERFUNC(MapClear, MapClearInfer) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MapClear, MapClearInfer);

IMPLEMT_INFERFUNC(MapIncompleteSize, MapIncompleteSizeInfer) {
  Shape scalar_shape;
  (void)Scalar(scalar_shape);
  TensorDesc output_desc = op.GetOutputDesc("size");
  output_desc.SetShape(scalar_shape);
  output_desc.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("size", output_desc) != GRAPH_SUCCESS) {
    OpsOPUpdateErrReport(op.GetName(), "size");
    OP_LOGE(op.GetName().c_str(), "Update [size] desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MapIncompleteSize, MapIncompleteSizeInfer);

IMPLEMT_INFERFUNC(Unstage, UnstageInfer) {
  Shape unknown_shape(UNKNOWN_SHAPE);

  std::vector<DataType> dtypes;
  if (op.GetAttr("dtypes", dtypes) == GRAPH_FAILED) {
    return GRAPH_FAILED;
  }
  size_t size = dtypes.size();

  for (size_t i = 0; i < size; ++i) {
    TensorDesc output_desc = op.GetDynamicOutputDesc("y", i);
    output_desc.SetShape(unknown_shape);
    output_desc.SetDataType(dtypes[i]);
    if (op.UpdateDynamicOutputDesc("y", i, output_desc) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Unstage, UnstageInfer);

IMPLEMT_INFERFUNC(TensorArray, TensorArrayInfer) {
  Shape unused;
  if (WithRank(op.GetInputDesc(0), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArray: size must be 0-D");
    return GRAPH_FAILED;
  }

  TensorDesc handle_desc = op.GetOutputDesc("handle");
  handle_desc.SetShape(Shape());
  handle_desc.SetDataType(DT_RESOURCE);
  if (op.UpdateOutputDesc("handle", handle_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArray: update handle desc failed");
    return GRAPH_FAILED;
  }

  TensorDesc flow_desc = op.GetOutputDesc("flow");
  flow_desc.SetShape(Shape());
  flow_desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("flow", flow_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArray: update flow desc failed");
    return GRAPH_FAILED;
  }

  bool identical_shapes;
  if (op.GetAttr("identical_element_shapes", identical_shapes) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArray: get attr identical_element_shapes failed");
    return GRAPH_FAILED;
  }

  Operator::OpType type;
  if (op.GetAttr("dtype", type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArray: get attr dtype failed");
    return GRAPH_FAILED;
  }

  Operator::OpListInt elem_dims;
  if (op.GetAttr("element_shape", elem_dims) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArray: get attr element_shape failed");
    return GRAPH_FAILED;
  }

  Shape elemShape(std::move(elem_dims));
  if (ShapeFullDefined(elemShape) || identical_shapes) {
    ShapeAndType shape_and_type(elemShape, type);
    std::vector<ShapeAndType> handle_shapes_and_types;
    handle_shapes_and_types.reserve(1);
    handle_shapes_and_types.emplace_back(shape_and_type);
    std::vector<std::vector<ShapeAndType>> shapes_and_types(2);
    shapes_and_types[0] = handle_shapes_and_types;
    auto context = op.GetInferenceContext();
    context->SetOutputHandleShapesAndTypes(shapes_and_types);
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorArray, TensorArrayInfer);

IMPLEMT_INFERFUNC(TensorArrayClose, TensorArrayCloseInfer) {
  Shape unused;
  if (WithRank(op.GetInputDesc(0), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArrayClose: handle must be 0-D");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorArrayClose, TensorArrayCloseInfer);

IMPLEMT_INFERFUNC(TensorArrayConcat, TensorArrayConcatInfer) {
  Shape unused;
  if (WithRank(op.GetInputDesc(0), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArrayConcat: handle must be 0-D");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArrayConcat: flow_in must be 0-D");
    return GRAPH_FAILED;
  }

  Operator::OpType type;
  if (op.GetAttr("dtype", type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArrayConcat: get attr dtype failed");
    return GRAPH_FAILED;
  }
  TensorDesc value_desc = op.GetOutputDesc("value");
  value_desc.SetDataType(type);
  // unknown rank
  value_desc.SetShape(Shape(ge::UNKNOWN_RANK));
  if (op.UpdateOutputDesc("value", value_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArrayConcat: update value desc failed");
    return GRAPH_FAILED;
  }

  TensorDesc lengths_desc = op.GetOutputDesc("lengths");
  lengths_desc.SetDataType(ge::DT_INT64);
  // 1-D, unknown dim
  lengths_desc.SetShape(Shape({ge::UNKNOWN_DIM}));
  if (op.UpdateOutputDesc("lengths", lengths_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArrayConcat: update lengths desc failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorArrayConcat, TensorArrayConcatInfer);

IMPLEMT_INFERFUNC(TensorArrayGather, TensorArrayGatherInfer) {
  Shape unused;
  Shape indices_shape;
  if (WithRank(op.GetInputDesc(0), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArrayGather: handle must be 0-D");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 1, indices_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArrayGather: indices must be 1-D");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(2), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArrayGather: flow_in must be 0-D");
    return GRAPH_FAILED;
  }
  auto context = op.GetInferenceContext();
  if (context == nullptr) {
    OP_LOGE(op.GetName().c_str(), "TensorArrayGather: inference context is nullptr");
    return GRAPH_FAILED;
  }
  std::vector<std::vector<ShapeAndType>> shapes_and_types = context->GetInputHandleShapesAndTypes();
  Shape input_or_attr_shape;
  if ((!shapes_and_types.empty()) && (!shapes_and_types.at(0).empty())) {
    input_or_attr_shape = shapes_and_types.at(0).at(0).GetShape();
  } else {
    Operator::OpListInt elem_dims;
    if (op.GetAttr("element_shape", elem_dims) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "TensorArrayGather: get attr element_shape failed");
      return GRAPH_FAILED;
    }
    input_or_attr_shape = Shape(std::move(elem_dims));
  }
  Shape output_shape;
  if (Concatenate(indices_shape, input_or_attr_shape, output_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArrayGather: concatenate failed, please check indices shape");
    return GRAPH_FAILED;
  }

  Operator::OpType type;
  if (op.GetAttr("dtype", type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArrayGather: get attr dtype failed");
    return GRAPH_FAILED;
  }
  TensorDesc value_desc = op.GetOutputDesc("value");
  value_desc.SetDataType(type);
  value_desc.SetShape(output_shape);
  if (op.UpdateOutputDesc("value", value_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArrayGather: update value desc failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorArrayGather, TensorArrayGatherInfer);

IMPLEMT_INFERFUNC(TensorArrayGrad, TensorArrayGradInfer) {
  Shape unused;
  if (WithRank(op.GetInputDesc(0), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArrayGrad: handle must be 0-D");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArrayGrad: flow_in must be 0-D");
    return GRAPH_FAILED;
  }

  TensorDesc grad_handle_desc = op.GetOutputDesc("grad_handle");
  grad_handle_desc.SetShape(Shape());
  grad_handle_desc.SetDataType(DT_RESOURCE);
  if (op.UpdateOutputDesc("grad_handle", grad_handle_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArrayGrad: update grad_handle desc failed");
    return GRAPH_FAILED;
  }

  TensorDesc flow_out_desc = op.GetOutputDesc("flow_out");
  flow_out_desc.SetShape(Shape());
  flow_out_desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("flow_out", flow_out_desc) != GRAPH_SUCCESS) {
    OP_LOGW(op.GetName().c_str(), "TensorArrayGrad: update flow_out desc failed");
  }

  auto context = op.GetInferenceContext();
  if (context == nullptr) {
    OP_LOGE(op.GetName().c_str(), "TensorArrayGrad: inference context is nullptr");
    return GRAPH_FAILED;
  }
  std::vector<std::vector<ShapeAndType>> input_shapes_and_types = context->GetInputHandleShapesAndTypes();
  if ((!input_shapes_and_types.empty()) && (!input_shapes_and_types.at(0).empty())) {
    ShapeAndType shape_and_type = input_shapes_and_types.at(0).at(0);
    std::vector<ShapeAndType> grad_handle_shape_and_type;
    grad_handle_shape_and_type.reserve(1);
    grad_handle_shape_and_type.emplace_back(shape_and_type);
    std::vector<std::vector<ShapeAndType>> shapes_and_types(2);
    shapes_and_types[0] = grad_handle_shape_and_type;
    context->SetOutputHandleShapesAndTypes(shapes_and_types);
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorArrayGrad, TensorArrayGradInfer);

IMPLEMT_INFERFUNC(TensorArrayWrite, TensorArrayWriteInfer) {
  Shape unused;
  if (WithRank(op.GetInputDesc(0), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArrayWrite: handle must be 0-D");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArrayWrite: index must be 0-D");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(3), 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArrayWrite: flow_in must be 0-D");
    return GRAPH_FAILED;
  }

  auto context = op.GetInferenceContext();
  if (context == nullptr) {
    OP_LOGE(op.GetName().c_str(), "TensorArrayWrite: inference context is nullptr");
    return GRAPH_FAILED;
  }
  std::vector<std::vector<ShapeAndType>> shapes_and_types = context->GetInputHandleShapesAndTypes();

  if (!shapes_and_types.empty() && !shapes_and_types.at(0).empty()) {
    ShapeAndType shape_and_type = shapes_and_types.at(0).at(0);
    Shape value_shape = op.GetInputDesc(2).GetShape();

    if (Merge(shape_and_type.GetShape(), value_shape, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "TensorArrayWrite: merge failed, please check the shape of 'value'");
      return GRAPH_FAILED;
    }
  }

  TensorDesc flow_out_desc = op.GetOutputDesc("flow_out");
  flow_out_desc.SetShape(Shape());
  flow_out_desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("flow_out", flow_out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArrayWrite: update flow_out desc failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorArrayWrite, TensorArrayWriteInfer);

IMPLEMT_INFERFUNC(TensorArrayGradWithShape, TensorArrayGradWithShapeInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(0), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Input handle must be Scalar.");
    return GRAPH_FAILED;
  }

  auto context = op.GetInferenceContext();
  std::vector<std::vector<ShapeAndType>> input_shapes_and_types = context->GetInputHandleShapesAndTypes();
  if ((!input_shapes_and_types.empty()) && (!input_shapes_and_types.at(0).empty())) {
    Shape input_shape = input_shapes_and_types.at(0).at(0).GetShape();
    DataType input_type = input_shapes_and_types.at(0).at(0).GetDataType();

    int64_t prepend_rank = op.GetInputDesc(2).GetShape().GetDim(0);
    std::vector<std::vector<ShapeAndType>> out_shape_and_type_list(2);
    std::vector<ShapeAndType> out_shapes_and_types;
    if (RankKnown(input_shape) && (prepend_rank != ge::UNKNOWN_DIM)) {
      size_t input_rank = input_shape.GetDimNum();
      std::vector<int64_t> dims;
      dims.reserve(input_rank + prepend_rank);
      for (int i = 0; i < prepend_rank; i++) {
        dims.push_back(ge::UNKNOWN_DIM);
      }

      for (size_t i = 0; i < input_rank; i++) {
        dims.push_back(input_shape.GetDim(i));
      }

      Shape out_shape(dims);
      ShapeAndType shape_and_type(out_shape, input_type);
      out_shapes_and_types.emplace_back(shape_and_type);
    } else {
      Shape unknown_shape(ge::UNKNOWN_SHAPE);
      ShapeAndType shape_and_type(unknown_shape, input_type);
      out_shapes_and_types.emplace_back(shape_and_type);
    }

    out_shape_and_type_list[0] = out_shapes_and_types;
    context->SetOutputHandleShapesAndTypes(out_shape_and_type_list);
  }

  TensorDesc grad_output_desc = op.GetOutputDesc("grad_handle");
  grad_output_desc.SetShape(Shape());
  grad_output_desc.SetDataType(DT_RESOURCE);
  if (op.UpdateOutputDesc("grad_handle", grad_output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update grad_handle desc failed.");
    return GRAPH_FAILED;
  }

  TensorDesc flow_output_desc = op.GetOutputDesc("flow_out");
  flow_output_desc.SetShape(Shape());
  flow_output_desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("flow_out", flow_output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update flow_out desc failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorArrayGradWithShape, TensorArrayGradWithShapeInfer);

IMPLEMT_INFERFUNC(TensorArrayRead, TensorArrayReadInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(0), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Input handle must be Scalar.");
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(1), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Input index must be Scalar.");
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(2), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Input flow_in must be Scalar.");
    return GRAPH_FAILED;
  }

  auto context = op.GetInferenceContext();
  std::vector<std::vector<ShapeAndType>> shapes_and_types = context->GetInputHandleShapesAndTypes();

  Operator::OpType type;
  if (op.GetAttr("dtype", type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "TensorArray: get attr dtype failed");
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetDataType(type);
  if ((!shapes_and_types.empty()) && (!shapes_and_types.at(0).empty())) {
    Shape output_shape = shapes_and_types.at(0).at(0).GetShape();
    output_desc.SetShape(output_shape);
  } else {
    output_desc.SetShape(Shape(ge::UNKNOWN_RANK));
  }

  if (op.UpdateOutputDesc("y", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update y desc failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorArrayRead, TensorArrayReadInfer);

IMPLEMT_INFERFUNC(TensorArrayScatter, TensorArrayScatterInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(0), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    ShapeErrReport(0, op.GetName(), DebugString(unused_shape.GetDims()), "scalar");
    OP_LOGE(op.GetName().c_str(), "Input handle must be Scalar.");
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(3), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    ShapeErrReport(3, op.GetName(), DebugString(unused_shape.GetDims()), "scalar");
    OP_LOGE(op.GetName().c_str(), "Input flow_in must be Scalar.");
    return GRAPH_FAILED;
  }

  Shape indices_shape;
  if (WithRank(op.GetInputDesc(1), 1, indices_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    ShapeErrReport(1, op.GetName(), DebugString(indices_shape.GetDims()), "1D");
    OP_LOGE(op.GetName().c_str(), "Input indices must be 1-D.");
    return GRAPH_FAILED;
  }

  Shape value_shape;
  int indices_rank = indices_shape.GetDimNum();
  if (WithRankAtLeast(op.GetInputDesc(2), indices_rank, value_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    ShapeErrReport(2, op.GetName(), DebugString(indices_shape.GetDims()), ConcatString("at least ", indices_rank, "D"));
    OP_LOGE(op.GetName().c_str(), "Input indices must be at least %dD.", indices_rank);
    return GRAPH_FAILED;
  }

  std::string err_msg;
  for (int i = 0; i < indices_rank; i++) {
    if ((value_shape.GetDim(i) >= 0) && (indices_shape.GetDim(i) >= 0) &&
        (value_shape.GetDim(i) != indices_shape.GetDim(i))) {
      err_msg = ConcatString(i, "th dim of input value shape", DebugString(value_shape.GetDims()),
                             " must equal to that of input indices shape", DebugString(indices_shape.GetDims()));
      InferShapeOtherErrReport(op.GetName(), ConcatString("the ", err_msg));
      OP_LOGE(op.GetName().c_str(), "The %s.", err_msg.c_str());
      return GRAPH_FAILED;
    }
  }

  auto context = op.GetInferenceContext();
  std::vector<std::vector<ShapeAndType>> shapes_and_types = context->GetInputHandleShapesAndTypes();
  if ((!shapes_and_types.empty()) && (!shapes_and_types.at(0).empty())) {
    Shape tensor_shape = shapes_and_types.at(0).at(0).GetShape();
    Shape fed_shape;
    if (SubShape(value_shape, 1, value_shape.GetDimNum(), 1, fed_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
      err_msg = ConcatString("subshape from 1 to ", value_shape.GetDimNum(), " of value shape",
                             DebugString(value_shape.GetDims()), " failed");
      InferShapeOtherErrReport(op.GetName(), ConcatString("get ", err_msg));
      OP_LOGE(op.GetName().c_str(), "Get %s.", err_msg.c_str());
      return GRAPH_FAILED;
    }

    if (Merge(tensor_shape, fed_shape, fed_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
      err_msg = ConcatString("tensorArray element shape", DebugString(tensor_shape.GetDims()), " and value subshape",
                             DebugString(fed_shape.GetDims()), " failed");
      InferShapeOtherErrReport(op.GetName(), ConcatString("merge ", err_msg));
      OP_LOGE(op.GetName().c_str(), "Merge %s.", err_msg.c_str());
      return GRAPH_FAILED;
    }
  }

  TensorDesc output_desc = op.GetOutputDesc("flow_out");
  output_desc.SetShape(Shape());
  output_desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("flow_out", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update flow_out desc failed");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorArrayScatter, TensorArrayScatterInfer);

IMPLEMT_INFERFUNC(TensorArraySplit, TensorArraySplitInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(0), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Input handle must be Scalar.");
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(2), 1, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Input lengths must be 1-D.");
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(3), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Input flow_in must be Scalar.");
    return GRAPH_FAILED;
  }

  TensorDesc output_desc = op.GetOutputDesc("flow_out");
  output_desc.SetShape(Shape());
  output_desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("flow_out", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update flow_out desc failed");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorArraySplit, TensorArraySplitInfer);

IMPLEMT_INFERFUNC(TensorArraySize, TensorArraySizeInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(0), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Input handle must be Scalar.");
    return GRAPH_FAILED;
  }

  TensorDesc output_desc = op.GetOutputDesc("size");
  output_desc.SetShape(Shape());
  output_desc.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("size", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update size desc failed");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorArraySize, TensorArraySizeInfer);

IMPLEMT_INFERFUNC(MapStage, MapStageInfer) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MapStage, MapStageInfer);

IMPLEMT_INFERFUNC(MapUnstage, MapUnstageInfer) {
  return InferMapShapes(op, "dtypes", "values");
}

INFER_FUNC_REG(MapUnstage, MapUnstageInfer);

IMPLEMT_INFERFUNC(MapUnstageNoKey, MapUnstageNoKeyInfer) {
  return InferMapShapes(op, "dtypes", "values");
}

INFER_FUNC_REG(MapUnstageNoKey, MapUnstageNoKeyInfer);

IMPLEMT_INFERFUNC(MapPeek, MapPeekInfer) {
  return InferMapShapes(op, "dtypes", "values");
}

INFER_FUNC_REG(MapPeek, MapPeekInfer);

IMPLEMT_INFERFUNC(MapSize, MapSizeInfer) {
  Shape scalar_shape;
  (void)Scalar(scalar_shape);
  TensorDesc output_desc = op.GetOutputDesc("size");
  output_desc.SetDataType(DT_INT32);
  output_desc.SetShape(scalar_shape);
  if (op.UpdateOutputDesc("size", output_desc) != GRAPH_SUCCESS) {
    OpsOPUpdateErrReport(op.GetName(), "size");
    OP_LOGE(op.GetName().c_str(), "Update [size] desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MapSize, MapSizeInfer);

IMPLEMT_INFERFUNC(RandomShuffleQueue, RandomShuffleQueueInfer) {
  Shape scalar_shape;
  (void)Scalar(scalar_shape);
  TensorDesc handle_desc = op.GetOutputDesc("handle");
  handle_desc.SetShape(scalar_shape);
  handle_desc.SetDataType(DT_RESOURCE);
  if (op.UpdateOutputDesc("handle", handle_desc) != GRAPH_SUCCESS) {
    OpsOPUpdateErrReport(op.GetName(), "handle");
    OP_LOGE(op.GetName().c_str(), "update [handle] desc failed");
    return GRAPH_FAILED;
  }
  return SetAttrsToShapesAndTypes(op, "component_types", "shapes");
}

INFER_FUNC_REG(RandomShuffleQueue, RandomShuffleQueueInfer);

IMPLEMT_INFERFUNC(DynamicPartition, DynamicPartitionInfer) {
  auto data_tensor = op.get_input_desc_x();
  auto partitions_tensor = op.get_input_desc_partitions();
  auto data_shape = data_tensor.GetShape();
  auto partitions_shape = partitions_tensor.GetShape();
  const auto rank_of_data = data_shape.GetDimNum();
  const auto rank_of_partitions = partitions_shape.GetDimNum();
  std::vector<int64_t> data_dims = data_shape.GetDims();
  std::vector<int64_t> partitions_dims = partitions_shape.GetDims();
  auto num_partitions = op.get_attr_num_partitions();

  if (partitions_shape.GetDims() == ge::UNKNOWN_SHAPE) {
    OP_LOGW(op.GetName().c_str(), "input partition is unknown_shape !");
    Shape unknown_shape(ge::UNKNOWN_SHAPE);
    for (int i = 0; i < num_partitions; ++i) {
      TensorDesc y_tensor = op.GetDynamicOutputDesc("y", i);
      y_tensor.SetShape(unknown_shape);
      op.UpdateDynamicOutputDesc("y", i, y_tensor);
    }
    return GRAPH_SUCCESS;
  }
  auto result = WithRankAtLeast(data_tensor, rank_of_partitions, data_shape, op.GetName().c_str());
  if (result != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "rank of x must >= rank of partitions !");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> dims;
  dims.reserve(std::max(rank_of_partitions, rank_of_data));
  dims.resize(rank_of_partitions);
  for (uint8_t i = 0; i < rank_of_partitions; i++) {
    if (Merge(data_dims[i], partitions_dims[i], dims[i]) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "shape of x must start with shape of partitions !");
      return GRAPH_FAILED;
    }
  }
  vector<int64_t> output_dim0({ge::UNKNOWN_DIM});
  Shape output_unknown_dim0 = Shape(output_dim0);
  Shape data_suffix_shape;
  result = SubShape(data_shape, rank_of_partitions, rank_of_data, 1, data_suffix_shape, op.GetName().c_str());
  if (result != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get suffix shape of x error ! ");
    return GRAPH_FAILED;
  }
  Shape output_shape;
  result = Concatenate(output_unknown_dim0, data_suffix_shape, output_shape);
  if (result != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "generate output_shape error! ");
    return GRAPH_FAILED;
  }
  for (int64_t i = 0; i < num_partitions; ++i) {
    TensorDesc y_tensor = op.GetDynamicOutputDesc("y", i);
    y_tensor.SetDataType(data_tensor.GetDataType());
    y_tensor.SetShape(output_shape);
    if (op.UpdateDynamicOutputDesc("y", i, y_tensor) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "update y_%ld desc failed", i);
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DynamicPartition, DynamicPartitionInfer);

IMPLEMT_INFERFUNC(DynamicStitch, DynamicStitchInfer) {
  auto num_incides = op.get_attr_N();
  if (num_incides < 1) {
    OP_LOGE(op.GetName().c_str(), "Attr N < 1");
    return GRAPH_FAILED;
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  // unknown shape support
  std::vector<std::string> input_infer_depends;
  for (int32_t i = 0; i < num_incides; ++i) {
    input_infer_depends.push_back(std::string("indices") + std::to_string(i));
  }
  op_desc->SetOpInferDepends(input_infer_depends);

  bool all_indices_constant = true;
  int32_t max_index = 0;
  GeShape last_suffix_shape(ge::UNKNOWN_RANK);

  for (int64_t i = 0; i < num_incides; ++i) {
    auto indices_tensor_name = "indices" + std::to_string(i);
    auto data_tensor_name = "x" + std::to_string(i);

    Tensor unused_tensor;
    if (op.GetInputConstData(indices_tensor_name, unused_tensor) != GRAPH_SUCCESS) {
      OP_LOGW(op.GetName().c_str(), "try get indices %ld failed !", i);
      all_indices_constant = false;
    }

    auto indices_desc = op_desc->MutableInputDesc(indices_tensor_name);
    auto data_desc = op_desc->MutableInputDesc(data_tensor_name);

    const auto indices_shape = indices_desc->GetShape();
    const auto data_shape = data_desc->GetShape();

    auto indices_dims = indices_shape.GetDims();
    auto data_dims = data_shape.GetDims();
    if (indices_dims == UNKNOWN_RANK || data_dims == UNKNOWN_RANK) {
      continue;
    }

    const auto rank_of_indices = indices_shape.GetDimNum();
    const auto rank_of_data = data_shape.GetDimNum();
    if (rank_of_indices > rank_of_data) {
      OP_LOGE(op.GetName().c_str(), "shape of x must start with shape of indices !");
      return GRAPH_FAILED;
    }

    std::vector<int64_t> dims;
    dims.reserve(std::max(rank_of_indices, rank_of_data));
    dims.resize(rank_of_indices);

    for (uint8_t i = 0; i < rank_of_indices; ++i) {
      if (data_dims[i] != UNKNOWN_DIM && indices_dims[i] != UNKNOWN_DIM &&
          (Merge(data_dims[i], indices_dims[i], dims[i]) != GRAPH_SUCCESS)) {
        OP_LOGE(op.GetName().c_str(), "shape of x must start with shape of indices !");
        return GRAPH_FAILED;
      }
    }

    GeShape data_suffix_shape;
    if (SubShape(data_shape, rank_of_indices, rank_of_data, 1, data_suffix_shape, op.GetName().c_str()) !=
        GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "get suffix shape of x error !");
      return GRAPH_FAILED;
    }
    if (Merge(last_suffix_shape, data_suffix_shape, last_suffix_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "Failed to merge suffixShape and data_suffix_shape !");
      return GRAPH_FAILED;
    }

    Tensor indices_data_tensor;
    if (op.GetInputConstData(indices_tensor_name, indices_data_tensor) == GRAPH_SUCCESS) {
      const int32_t* indices_data = reinterpret_cast<const int32_t*>(indices_data_tensor.GetData());
      int64_t count = indices_shape.GetShapeSize();
      for (int32_t i = 0; i < count; ++i) {
        if (static_cast<int64_t>(indices_data[i]) > max_index) {
          max_index = static_cast<int64_t>(indices_data[i]);
        }
      }
    }
  }
  auto output_dim0 = all_indices_constant ? (max_index + 1) : (ge::UNKNOWN_DIM);
  GeShape output_shape_prefix({output_dim0});
  GeShape output_shape;
  if (Concatenate(output_shape_prefix, last_suffix_shape, output_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "generate output_shape error! ");
    return GRAPH_FAILED;
  }

  GeTensorDescPtr y_desc = op_desc->MutableOutputDesc("y");
  GeShape y_shape(output_shape.GetDims());
  DataType y_data_type = op_desc->MutableInputDesc("x0")->GetDataType();

  (void)FillOpDesc(y_desc, y_shape, y_data_type);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DynamicStitch, DynamicStitchInfer);

IMPLEMT_INFERFUNC(ParallelDynamicStitch, ParallelDynamicStitchInfer) {
  auto num_incides = op.get_attr_N();
  bool all_indices_constant = true;
  int32_t max_index = 0;
  Shape last_suffix_shape(ge::UNKNOWN_RANK);

  if (num_incides < 1) {
    OP_LOGE(op.GetName().c_str(), "Attr N < 1");
    return GRAPH_FAILED;
  }
  for (int64_t i = 0; i < num_incides; ++i) {
    auto indices_tensor_name = "indices" + std::to_string(i);
    TensorDesc unused_tensor;
    if ((op.TryGetInputDesc(indices_tensor_name, unused_tensor)) == GRAPH_FAILED) {
      OP_LOGW(op.GetName().c_str(), "try get indices %ld failed !", i);
      all_indices_constant = false;
    }
    const TensorDesc indices_tensor = op.GetDynamicInputDesc("indices", i);
    const TensorDesc data_tensor = op.GetDynamicInputDesc("x", i);
    const auto indices_shape = indices_tensor.GetShape();
    const auto data_shape = data_tensor.GetShape();
    const auto rank_of_indices = indices_shape.GetDimNum();
    const auto rank_of_data = data_shape.GetDimNum();
    std::vector<int64_t> indices_dims = indices_shape.GetDims();
    std::vector<int64_t> data_dims = data_shape.GetDims();
    std::vector<int64_t> dims;
    dims.reserve(std::max(rank_of_indices, rank_of_data));
    dims.resize(rank_of_indices);

    for (uint8_t i = 0; i < rank_of_indices; ++i) {
      if ((Merge(data_dims[i], indices_dims[i], dims[i])) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "shape of x must start with shape of indices !");
        return GRAPH_FAILED;
      }
    }
    Shape data_suffix_shape;
    auto result = SubShape(data_shape, rank_of_indices, rank_of_data, 1, data_suffix_shape, op.GetName().c_str());
    if (result != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "get suffix shape of x error !");
      return GRAPH_FAILED;
    }
    result = Merge(last_suffix_shape, data_suffix_shape, last_suffix_shape, op.GetName().c_str());
    if (result != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "Failed to merge suffixShape and data_suffix_shape !");
      return GRAPH_FAILED;
    }
    if (op.TryGetInputDesc(indices_tensor_name, unused_tensor) == GRAPH_SUCCESS) {
      Tensor indices_data_tensor;
      result = op.GetInputConstData(indices_tensor_name, indices_data_tensor);
      if (result != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "Failed to get indices tensor data !");
        return GRAPH_FAILED;
      }
      const int32_t* indices_data = reinterpret_cast<const int32_t*>(indices_data_tensor.GetData());
      int64_t count = indices_data_tensor.GetTensorDesc().GetShape().GetShapeSize();
      for (int32_t i = 0; i < count; ++i) {
        if (static_cast<int64_t>(indices_data[i]) > max_index) {
          max_index = static_cast<int64_t>(indices_data[i]);
        }
      }
    }
  }
  auto output_dim0 = all_indices_constant ? (max_index + 1) : (ge::UNKNOWN_DIM);
  Shape output_shapePrefix({output_dim0});
  Shape output_shape;
  auto result = Concatenate(output_shapePrefix, last_suffix_shape, output_shape);
  if (result != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "generate output_shape error! ");
    return GRAPH_FAILED;
  }

  TensorDesc y_desc = op.GetOutputDesc("y");
  auto data_tensor = op.GetDynamicInputDesc("x", 0);
  y_desc.SetDataType(data_tensor.GetDataType());
  y_desc.SetShape(output_shape);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update y desc failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ParallelDynamicStitch, ParallelDynamicStitchInfer);

IMPLEMT_INFERFUNC(PaddingFIFOQueue, PaddingFIFOQueueInfer) {
  TensorDesc output_desc = op.GetOutputDesc("handle");
  output_desc.SetShape(Shape());
  output_desc.SetDataType(DT_RESOURCE);
  if (op.UpdateOutputDesc("handle", output_desc) != GRAPH_SUCCESS) {
    OpsOPUpdateErrReport(op.GetName(), "handle");
    OP_LOGE(op.GetName().c_str(), "update [handle] desc failed.");
    return GRAPH_FAILED;
  }
  return SetAttrsToShapesAndTypes(op, "component_types", "shapes");
}

INFER_FUNC_REG(PaddingFIFOQueue, PaddingFIFOQueueInfer);

IMPLEMT_INFERFUNC(PriorityQueue, PriorityQueueInfer) {
  TensorDesc output_desc = op.GetOutputDesc("handle");
  output_desc.SetShape(Shape());
  output_desc.SetDataType(DT_RESOURCE);
  if (op.UpdateOutputDesc("handle", output_desc) != GRAPH_SUCCESS) {
    OpsOPUpdateErrReport(op.GetName(), "handle");
    OP_LOGE(op.GetName().c_str(), "update [handle] desc failed.");
    return GRAPH_FAILED;
  }
  return SetAttrsToShapesAndTypes(op, "component_types", "shapes");
}

INFER_FUNC_REG(PriorityQueue, PriorityQueueInfer);

IMPLEMT_INFERFUNC(QueueClose, QueueCloseInfer) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(QueueClose, QueueCloseInfer);

IMPLEMT_INFERFUNC(OrderedMapStage, OrderedMapStageInfer) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(OrderedMapStage, OrderedMapStageInfer);

IMPLEMT_INFERFUNC(OrderedMapSize, OrderedMapSizeInfer) {
  Shape scalar_shape;
  (void)Scalar(scalar_shape);

  TensorDesc output_tensor = op.GetOutputDesc("size");
  output_tensor.SetShape(scalar_shape);
  output_tensor.SetDataType(DT_INT32);
  return op.UpdateOutputDesc("size", output_tensor);
}

INFER_FUNC_REG(OrderedMapSize, OrderedMapSizeInfer);

IMPLEMT_INFERFUNC(OrderedMapClear, OrderedMapClearInfer) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(OrderedMapClear, OrderedMapClearInfer);

IMPLEMT_INFERFUNC(OrderedMapIncompleteSize, OrderedMapIncompleteSizeInfer) {
  Shape scalar_shape;
  (void)Scalar(scalar_shape);

  TensorDesc output_tensor = op.GetOutputDesc("size");
  output_tensor.SetShape(scalar_shape);
  output_tensor.SetDataType(DT_INT32);
  return op.UpdateOutputDesc("size", output_tensor);
}

INFER_FUNC_REG(OrderedMapIncompleteSize, OrderedMapIncompleteSizeInfer);

IMPLEMT_INFERFUNC(OrderedMapPeek, OrderedMapPeekInfer) {
  Shape unknown_shape(ge::UNKNOWN_SHAPE);
  std::vector<ge::DataType> list_type;
  if (op.GetAttr("dtypes", list_type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get Attr dtyoes error.");
    return GRAPH_FAILED;
  }
  size_t outputs_size = list_type.size();
  graphStatus status;
  for (size_t i = 0; i < outputs_size; ++i) {
    TensorDesc output_tensor = op.GetDynamicOutputDesc("values", i);
    output_tensor.SetShape(unknown_shape);
    output_tensor.SetDataType(list_type[i]);
    status = op.UpdateDynamicOutputDesc("values", i, output_tensor);
    if (status != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "Update output shape and type error.");
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(OrderedMapPeek, OrderedMapPeekInfer);

IMPLEMT_INFERFUNC(OrderedMapUnstageNoKey, OrderedMapUnstageNoKeyInfer) {
  Shape unknown_shape(ge::UNKNOWN_SHAPE);
  std::vector<ge::DataType> list_type;
  if (op.GetAttr("dtypes", list_type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get Attr dtyoes error.");
    return GRAPH_FAILED;
  }
  size_t outputs_size = list_type.size();
  graphStatus status;
  for (size_t i = 0; i < outputs_size; ++i) {
    TensorDesc output_tensor = op.GetDynamicOutputDesc("values", i);
    output_tensor.SetShape(unknown_shape);
    output_tensor.SetDataType(list_type[i]);
    status = op.UpdateDynamicOutputDesc("values", i, output_tensor);
    if (status != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "Update output shape and type error.");
      return GRAPH_FAILED;
    }
  }
  TensorDesc tensordesc_output = op.GetOutputDesc("key");
  tensordesc_output.SetDataType(DT_INT64);
  (void)op.UpdateOutputDesc("key", tensordesc_output);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(OrderedMapUnstageNoKey, OrderedMapUnstageNoKeyInfer);

IMPLEMT_INFERFUNC(OrderedMapUnstage, OrderedMapUnstageInfer) {
  Shape unknown_shape(ge::UNKNOWN_SHAPE);
  std::vector<ge::DataType> list_type;
  if (op.GetAttr("dtypes", list_type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get Attr dtyoes error.");
    return GRAPH_FAILED;
  }
  size_t outputs_size = list_type.size();
  graphStatus status;
  for (size_t i = 0; i < outputs_size; ++i) {
    TensorDesc output_tensor = op.GetDynamicOutputDesc("values", i);
    output_tensor.SetShape(unknown_shape);
    output_tensor.SetDataType(list_type[i]);
    status = op.UpdateDynamicOutputDesc("values", i, output_tensor);
    if (status != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "Update output shape and type error.");
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(OrderedMapUnstage, OrderedMapUnstageInfer);

IMPLEMT_INFERFUNC(Barrier, BarrierInfer) {
  std::vector<ge::DataType> component_types;
  if (op.GetAttr("component_types", component_types) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to get attr component_types");
    return GRAPH_FAILED;
  }
  if (component_types.size() < 1) {
    OP_LOGE(op.GetName().c_str(), "The length of component_types should not be less than 1.");
    return GRAPH_FAILED;
  }
  Shape out_shape;
  if (Vector(2, out_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to create vector.");
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDesc("handle");
  output_desc.SetShape(out_shape);
  output_desc.SetDataType(DT_STRING_REF);
  op.UpdateOutputDesc("handle", output_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Barrier, BarrierInfer);

IMPLEMT_INFERFUNC(BarrierInsertMany, BarrierInsertManyInfer) {
  TensorDesc keys_desc = op.GetInputDesc("keys");
  TensorDesc values_desc = op.GetInputDesc("values");
  Shape handle_shape, keys_shape, values_shape;
  if (WithRank(op.GetInputDesc("handle"), 1, handle_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "handle's rank should be 1.");
    return GRAPH_FAILED;
  }
  int64_t unused_dim;
  if (WithValue(handle_shape.GetDim(0), 2, unused_dim, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "handle's dim 0 should be 2.");
    return GRAPH_FAILED;
  }
  if (WithRank(keys_desc, 1, keys_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "keys' rank should be 1.");
    return GRAPH_FAILED;
  }
  if (WithRankAtLeast(values_desc, 1, values_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "values' rank should be at least 1.");
    return GRAPH_FAILED;
  }
  graphStatus status;
  Shape values_dim_zero_shape;
  Vector(values_shape.GetDim(0), values_dim_zero_shape);
  status = Merge(keys_shape, values_dim_zero_shape, handle_shape, op.GetName().c_str());
  if (status != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Can't merge keys_shape and the vector with length of vectors.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BarrierInsertMany, BarrierInsertManyInfer);

IMPLEMT_INFERFUNC(BarrierTakeMany, BarrierTakeManyInfer) {
  std::vector<ge::DataType> component_types;
  if (op.GetAttr("component_types", component_types) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to get attr component_types");
    return GRAPH_FAILED;
  }
  if (component_types.size() < 1) {
    OP_LOGE(op.GetName().c_str(), "The length of component_types should not be less than 1.");
    return GRAPH_FAILED;
  }
  bool component_types_flag = true;
  for (size_t i = 1; i < component_types.size(); ++i) {
    if (component_types[i] != component_types[0]) {
      component_types_flag = false;
    }
  }
  if (component_types_flag == false) {
    OP_LOGE(op.GetName().c_str(), "All menbers of component_types should be the same.");
    return GRAPH_FAILED;
  }
  graphStatus status;
  TensorDesc output_desc;
  // update ordinary outputs as unknown shape
  vector<string> outputs_name = {"indices", "keys"};
  vector<DataType> outputs_type = {DT_INT64, DT_STRING};
  for (size_t i = 0; i < outputs_name.size(); ++i) {
    output_desc = op.GetOutputDesc(outputs_name[i]);
    output_desc.SetDataType(outputs_type[i]);
    output_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
    status = op.UpdateOutputDesc(outputs_name[i], output_desc);
    if (status != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "Failed to update %s desc.", outputs_name[i].c_str());
      return GRAPH_FAILED;
    }
  }
  // update dynamic outputs as unknown shape
  int size = op.GetDynamicOutputNum("values");
  for (int i = 0; i < size; ++i) {
    output_desc = op.GetDynamicOutputDesc("values", i);
    output_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
    output_desc.SetDataType(component_types[0]);
    status = op.UpdateDynamicOutputDesc("values", i, output_desc);
    if (status != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "Failed to update output desc.");
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BarrierTakeMany, BarrierTakeManyInfer);

IMPLEMT_INFERFUNC(BarrierClose, BarrierCloseInfer) {
  Shape shape;
  int64_t unused_dim;
  for (size_t i = 0; i < op.GetInputsSize(); ++i) {
    if (WithRank(op.GetInputDesc(i), 1, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "The rank of input %d should be 1.", i);
      return GRAPH_FAILED;
    }
    if (WithValue(shape.GetDim(0), 2, unused_dim, op.GetName().c_str()) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "dim[0] of handle should be 2.");
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BarrierClose, BarrierCloseInfer);

IMPLEMT_INFERFUNC(BarrierReadySize, BarrierReadySizeInfer) {
  Shape shape;
  int64_t unused_dim;
  for (size_t i = 0; i < op.GetInputsSize(); ++i) {
    if (WithRank(op.GetInputDesc(i), 1, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "The rank of input %d should be 1.", i);
      return GRAPH_FAILED;
    }
    if (WithValue(shape.GetDim(0), 2, unused_dim, op.GetName().c_str()) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "dim[0] of handle should be 2.");
      return GRAPH_FAILED;
    }
  }
  TensorDesc output_desc = op.GetOutputDesc("size");
  output_desc.SetShape(Shape());
  output_desc.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("size", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to update size desc.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BarrierReadySize, BarrierReadySizeInfer);

IMPLEMT_INFERFUNC(BarrierIncompleteSize, BarrierIncompleteSizeInfer) {
  Shape shape;
  int64_t unused_dim;
  for (size_t i = 0; i < op.GetInputsSize(); ++i) {
    if (WithRank(op.GetInputDesc(i), 1, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "The rank of input %d should be 1.", i);
      return GRAPH_FAILED;
    }
    if (WithValue(shape.GetDim(0), 2, unused_dim, op.GetName().c_str()) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "dim[0] of handle should be 2.");
      return GRAPH_FAILED;
    }
  }
  TensorDesc output_desc = op.GetOutputDesc("size");
  output_desc.SetShape(Shape());
  output_desc.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("size", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to update size desc.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BarrierIncompleteSize, BarrierIncompleteSizeInfer);

IMPLEMT_INFERFUNC(RecordInput, RecordInputInfer) {
  graphStatus status;
  TensorDesc output_desc = op.GetOutputDesc("records");
  output_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
  output_desc.SetDataType(DT_STRING);
  status = op.UpdateOutputDesc("records", output_desc);
  if (status != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to update output desc.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(RecordInput, RecordInputInfer);

IMPLEMT_INFERFUNC(ConditionalAccumulator, ConditionalAccumulatorInfer) {
  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr dtype failed");
    return GRAPH_FAILED;
  }
  Shape output_shape;
  if (Vector(2, output_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "create output shape failed");
    return GRAPH_FAILED;
  }
  TensorDesc handle_desc = op.get_output_desc_handle();
  handle_desc.SetShape(output_shape);
  handle_desc.SetDataType(DT_STRING_REF);
  if (op.UpdateOutputDesc("handle", handle_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update handle desc failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ConditionalAccumulator, ConditionalAccumulatorInfer);

IMPLEMT_INFERFUNC(AccumulatorApplyGradient, AccumulatorApplyGradientInfer) {
  Shape unused_shape;
  auto local_step_tensor = op.get_input_desc_local_step();
  if (WithRank(local_step_tensor, 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input local_step must be 0-D");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AccumulatorApplyGradient, AccumulatorApplyGradientInfer);

IMPLEMT_INFERFUNC(AccumulatorNumAccumulated, AccumulatorNumAccumulatedInfer) {
  Shape scalar_shape;
  if (Scalar(scalar_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "create scalar shape failed");
    return GRAPH_FAILED;
  }
  TensorDesc y_desc = op.get_output_desc_y();
  y_desc.SetShape(scalar_shape);
  y_desc.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update y desc failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AccumulatorNumAccumulated, AccumulatorNumAccumulatedInfer);

IMPLEMT_INFERFUNC(AccumulatorSetGlobalStep, AccumulatorSetGlobalStepInfer) {
  Shape unused_shape;
  auto new_global_step_tensor = op.get_input_desc_new_global_step();
  if (WithRank(new_global_step_tensor, 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input new_global_step must be 0-D");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AccumulatorSetGlobalStep, AccumulatorSetGlobalStepInfer);

IMPLEMT_INFERFUNC(AccumulatorTakeGradient, AccumulatorTakeGradientInfer) {
  Shape unused_shape;
  auto num_required_tensor = op.get_input_desc_num_required();
  if (WithRank(num_required_tensor, 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input num_required must be 0-D");
    return GRAPH_FAILED;
  }
  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr dtype failed");
    return GRAPH_FAILED;
  }
  TensorDesc y_desc = op.get_output_desc_y();
  y_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
  y_desc.SetDataType(dtype);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update y desc failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AccumulatorTakeGradient, AccumulatorTakeGradientInfer);

IMPLEMT_INFERFUNC(SparseConditionalAccumulator, SparseConditionalAccumulatorInfer) {
  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr dtype failed");
    return GRAPH_FAILED;
  }
  Shape output_shape;
  if (Vector(2, output_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "create output shape failed");
    return GRAPH_FAILED;
  }
  TensorDesc handle_desc = op.get_output_desc_handle();
  handle_desc.SetShape(output_shape);
  handle_desc.SetDataType(DT_STRING_REF);
  if (op.UpdateOutputDesc("handle", handle_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update handle desc failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseConditionalAccumulator, SparseConditionalAccumulatorInfer);

IMPLEMT_INFERFUNC(SparseAccumulatorApplyGradient, SparseAccumulatorApplyGradientInfer) {
  Shape unused_shape;
  auto local_step_tensor = op.get_input_desc_local_step();
  if (WithRank(local_step_tensor, 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input local_step must be 0-D");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseAccumulatorApplyGradient, SparseAccumulatorApplyGradientInfer);

IMPLEMT_INFERFUNC(SparseAccumulatorTakeGradient, SparseAccumulatorTakeGradientInfer) {
  Shape unused_shape;
  auto num_required_tensor = op.get_input_desc_num_required();
  if (WithRank(num_required_tensor, 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input num_required must be 0-D");
    return GRAPH_FAILED;
  }

  TensorDesc indices_desc = op.get_output_desc_indices();
  indices_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
  indices_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("indices", indices_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update indices desc failed");
    return GRAPH_FAILED;
  }

  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr dtype failed");
    return GRAPH_FAILED;
  }
  TensorDesc values_desc = op.get_output_desc_values();
  values_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
  values_desc.SetDataType(dtype);
  if (op.UpdateOutputDesc("values", values_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update values desc failed");
    return GRAPH_FAILED;
  }

  TensorDesc shape_desc = op.get_output_desc_shape();
  shape_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
  shape_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("shape", shape_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update shape desc failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SparseAccumulatorTakeGradient, SparseAccumulatorTakeGradientInfer);

IMPLEMT_INFERFUNC(ResourceConditionalAccumulator, ResourceConditionalAccumulatorInfer) {
  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr dtype failed");
    return GRAPH_FAILED;
  }
  Shape vector_shape;
  // Set Output as Vector(2) of DT_RESOURCE
  if (Vector(2, vector_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "create output shape failed");
    return GRAPH_FAILED;
  }
  auto handle_desc = op.GetOutputDesc("handle");
  handle_desc.SetShape(vector_shape);
  handle_desc.SetDataType(DT_RESOURCE);


  if (op.UpdateOutputDesc("handle", handle_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update handle desc failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ResourceConditionalAccumulator, ResourceConditionalAccumulatorInfer);

IMPLEMT_INFERFUNC(ResourceAccumulatorApplyGradient, ResourceAccumulatorApplyGradientInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc("local_step"), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input local_step must be 0-D");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ResourceAccumulatorApplyGradient, ResourceAccumulatorApplyGradientInfer);

IMPLEMT_INFERFUNC(ResourceAccumulatorNumAccumulated, ResourceAccumulatorNumAccumulatedInfer) {
  Shape scalar_shape;
  if (Scalar(scalar_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "create scalar shape failed");
    return GRAPH_FAILED;
  }
  auto num_accumulated_sesc = op.GetOutputDesc("num_accumulated");
  num_accumulated_sesc.SetShape(scalar_shape);
  num_accumulated_sesc.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("num_accumulated", num_accumulated_sesc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update num_accumulated desc failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ResourceAccumulatorNumAccumulated, ResourceAccumulatorNumAccumulatedInfer);

IMPLEMT_INFERFUNC(ResourceAccumulatorSetGlobalStep, ResourceAccumulatorSetGlobalStepInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc("new_global_step"), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input new_global_step must be 0-D");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ResourceAccumulatorSetGlobalStep, ResourceAccumulatorSetGlobalStepInfer);

IMPLEMT_INFERFUNC(ResourceAccumulatorTakeGradient, ResourceAccumulatorTakeGradientInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc("num_required"), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input num_required must be 0-D");
    return GRAPH_FAILED;
  }
  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr dtype failed");
    return GRAPH_FAILED;
  }
  auto average_desc = op.GetOutputDesc("average");
  average_desc.SetShape(Shape(ge::UNKNOWN_SHAPE));
  average_desc.SetDataType(dtype);
  if (op.UpdateOutputDesc("average", average_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update average desc failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ResourceAccumulatorTakeGradient, ResourceAccumulatorTakeGradientInfer);

IMPLEMT_INFERFUNC(OutfeedEnqueueOp, OutfeedEnqueueInfer) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(OutfeedEnqueueOp, OutfeedEnqueueInfer);

int32_t Split(const std::string &s, std::vector<std::string> &result, const char *delchar) {
  if (s.empty()) { return 0; }
  result.clear();
  char *buffer = new (std::nothrow) char[s.size() + 1];
  if (buffer == nullptr) {
    return -1;
  }
  buffer[s.size()] = '\0';
  errno_t e = strcpy_s(buffer, s.size() + 1, s.c_str());
  if (e != EOK) {
    delete[] buffer;
    return -1;
  }
  char *p_tmp = nullptr;
  char *p = strtok_s(buffer, delchar, &p_tmp);
  if (p != nullptr) {
    do { result.emplace_back(p); } while ((p = strtok_s(nullptr, delchar, &p_tmp)));
  }
  delete[] buffer;
  return 0;
}

IMPLEMT_INFERFUNC(DynamicGetNext, DynamicGetNextInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc("x"), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Input x must be 0-D");
    return GRAPH_FAILED;
  }

  std::vector<ge::DataType> output_types;
  if (op.GetAttr("output_types", output_types) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Op get attr output_types failed");
    return GRAPH_FAILED;
  }

  std::vector<std::vector<int64_t>> output_shapes;
  if (op.GetAttr("output_shapes", output_shapes) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Op get attr output_shapes failed");
    return GRAPH_FAILED;
  }

  std::string dynamic_graph_execute_mode;
  if (op.GetAttr("_dynamic_graph_execute_mode", dynamic_graph_execute_mode) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Op get attr _dynamic_graph_execute_mode failed");
    return GRAPH_FAILED;
  }

  std::string getnext_inputs_shape_range;
  if (op.GetAttr("_getnext_inputs_shape_range", getnext_inputs_shape_range) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Op get attr _getnext_inputs_shape_range failed");
    return GRAPH_FAILED;
  }

  if (dynamic_graph_execute_mode != "lazy_recompile" && dynamic_graph_execute_mode != "dynamic_execute") {
    OP_LOGE(op.GetName().c_str(), "Op get attr _dynamic_graph_execute_mode : [%s] is invailed",
            dynamic_graph_execute_mode.c_str());
    return GRAPH_FAILED;
  }

  std::vector<std::vector<std::pair<int64_t, int64_t>>> shape_ranges;
  if (dynamic_graph_execute_mode == "dynamic_execute") {
    std::string regex_format = std::string("(\\[(((\\d+|(\\d+~\\d+)),{1})*)(\\d+|(\\d+~\\d+))\\],{1})*") +
                               std::string("\\[(((\\d+|(\\d+~\\d+)),{1})*)(\\d+|(\\d+~\\d+))\\]");
    std::regex r(regex_format, std::regex_constants::ECMAScript);
    if (!regex_match(getnext_inputs_shape_range, r)) {
      OP_LOGE(op.GetName().c_str(),
              "The format of _getnext_inputs_shape_range is incorrect, and the parsing failed");
      return GRAPH_FAILED;
    }
    std::vector<std::string> shape_range_strings;
    int32_t res = Split(getnext_inputs_shape_range,
                        shape_range_strings, "]"); // shape_range_strings = [128,3~5,2~128,1],[128]
    if (res != 0) {
      OP_LOGE(op.GetName().c_str(),
              "Split _getnext_inputs_shape_range failed");
      return GRAPH_FAILED;
    }

    for (auto shape_range_string : shape_range_strings) {
      if (shape_range_string.find("[") == 0) {
        shape_range_string.erase(0, 1);
      } else if (shape_range_string.find(",[") == 0) {
        shape_range_string.erase(0, 2);
      } else {
        OP_LOGE(op.GetName().c_str(),
                "The format of _getnext_inputs_shape_range is incorrect, and the parsing failed");
        return GRAPH_FAILED;
      }
      // shape_range_string = 128,3~5,2~128,1
      std::vector<std::string> single_shape_ranges;
      std::vector<std::pair<int64_t, int64_t>> pair_shape_ranges;
      res = Split(shape_range_string, single_shape_ranges, ",");
      if (res != 0) {
        OP_LOGE(op.GetName().c_str(),
                "Split _getnext_inputs_shape_range failed");
        return GRAPH_FAILED;
      }
      for (auto single_shape_range : single_shape_ranges) {
        std::vector<std::string> tmp;
        res = Split(single_shape_range, tmp, "~");
        if (res != 0) {
          OP_LOGE(op.GetName().c_str(),
                  "Split _getnext_inputs_shape_range failed");
          return GRAPH_FAILED;
        }
        if (tmp.size() > 2 || tmp.size() <= 0) {
          OP_LOGE(op.GetName().c_str(),
                  "The format of _getnext_inputs_shape_range is incorrect, and the parsing failed");
          return GRAPH_FAILED;
        }
        pair_shape_ranges.push_back(tmp.size() == 1 ? \
                      std::pair<int64_t, int64_t>{ atoi(tmp[0].c_str()), atoi(tmp[0].c_str()) } : \
                      std::pair<int64_t, int64_t>{ atoi(tmp[0].c_str()), atoi(tmp[1].c_str()) });
      }
      shape_ranges.push_back(pair_shape_ranges);
    }

    if (shape_ranges.size() != output_shapes.size()) {
      OP_LOGE(op.GetName().c_str(), "The _getnext_inputs_shape_range and output_shapes should be the same length.");
      return GRAPH_FAILED;
    }
    for (int i = 0; i < shape_ranges.size(); i++) {
      if (shape_ranges[i].size() != output_shapes[i].size()) {
        OP_LOGE(op.GetName().c_str(),
                "The _getnext_inputs_shape_range and output_shapes at [%d] should be the same length.", i);
        return GRAPH_FAILED;
      }
    }
  }

  if (output_types.size() != output_shapes.size()) {
    OP_LOGE(op.GetName().c_str(), "The output_types and output_shapes should be the same length.");
    return GRAPH_FAILED;
  }

  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  graphStatus output_status;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  for (int i = 0; i < output_shapes.size(); i++) {
    GeTensorDescPtr y_desc = op_desc->MutableOutputDesc("y");
    TensorDesc output_desc = op.GetDynamicOutputDesc("y", i);
    Shape shape(output_shapes[i]);
    output_desc.SetShape(Shape(output_shapes[i]));
    output_desc.SetDataType(output_types[i]);
    if (RankKnown(shape)) {
      auto dims = shape.GetDims();
      bool shape_fully_defined = true;
      for (const int64_t& dim : dims) {
        if (dim == UNKNOWN_DIM) {
          shape_fully_defined = false;
          break;
        }
      }
      if (!shape_fully_defined) {
        std::vector<std::pair<int64_t, int64_t>> shape_range;
        for (int j = 0; j < dims.size(); j++) {
          int64_t dim = dims[j];
          if (dynamic_graph_execute_mode == "lazy_recompile") {
            shape_range.push_back(dim == UNKNOWN_DIM ? std::pair<int64_t, int64_t>{1, -1} : \
                                                       std::pair<int64_t, int64_t>{dim, dim});
          } else {
            shape_range.push_back(shape_ranges[i][j]);
          }
        }
        output_desc.SetShapeRange(shape_range);
      }
    }
    output_status = op.UpdateDynamicOutputDesc("y", i, output_desc);
    if (output_status != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "Update [%s] [%d] failed", "y", i);
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DynamicGetNext, DynamicGetNextInfer);

// --------------------------------LruCache-------------------------------------
IMPLEMT_COMMON_INFERFUNC(LruCacheInferShape) {
  int64_t cache_size;
  if (op.GetAttr("cache_size", cache_size) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr cache_size error.");
    return GRAPH_FAILED;
  }
  if (cache_size <= 0) {
    OP_LOGE(op.GetName().c_str(), "cache_size should be >0 , real value is %d.", cache_size);
    return GRAPH_PARAM_INVALID;
  }

  float load_factor;
  if (op.GetAttr("load_factor", load_factor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr load_factor error.");
    return GRAPH_FAILED;
  }
  if (load_factor <= 0.0 || load_factor > 1.0) {
    OP_LOGE(op.GetName().c_str(), "load_factor should be in  (0, 1.0] , real value is %d.", load_factor);
    return GRAPH_PARAM_INVALID;
  }
  Shape scalar_shape;
  (void)Scalar(scalar_shape);
  TensorDesc output_desc = op.GetOutputDesc("cache");
  DataType output_type = DT_RESOURCE;
  output_desc.SetShape(scalar_shape);
  output_desc.SetDataType(output_type);
  graphStatus output_status = op.UpdateOutputDesc("cache", output_desc);
  if (output_status != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update handle failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LruCache, LruCacheInferShape);
// ---------------------LruCache END-------------------------------------

// --------------------------------CacheAdd-------------------------------------
IMPLEMT_COMMON_INFERFUNC(CacheAddInferShape) {
  OpDescPtr op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr ids = op_desc->MutableInputDesc(1);

  GeShape ids_shape;
  if (WithRank(ids, 1, ids_shape) != GRAPH_SUCCESS) {
    ShapeErrReport(0, op.GetName(), DebugString(ids->GetShape().GetDims()), "1D");
    OP_LOGE(op.GetName().c_str(), "input ids must be 1-D");
    return GRAPH_FAILED;
  }
  DataType output_type = DT_INT64;
  GeTensorDescPtr swap_in_id_desc = op_desc->MutableOutputDesc(0);
  GeTensorDescPtr swap_in_idx_desc = op_desc->MutableOutputDesc(1);
  GeTensorDescPtr swap_out_id_desc = op_desc->MutableOutputDesc(2);
  GeTensorDescPtr swap_out_idx_desc = op_desc->MutableOutputDesc(3);
  swap_in_id_desc->SetShape(GeShape({UNKNOWN_DIM}));
  swap_in_id_desc->SetOriginShape(GeShape({UNKNOWN_DIM}));
  swap_in_id_desc->SetDataType(ids->GetDataType());

  swap_in_idx_desc->SetShape(GeShape({UNKNOWN_DIM}));
  swap_in_idx_desc->SetOriginShape(GeShape({UNKNOWN_DIM}));
  swap_in_idx_desc->SetDataType(output_type);

  swap_out_id_desc->SetShape(GeShape({UNKNOWN_DIM}));
  swap_out_id_desc->SetOriginShape(GeShape({UNKNOWN_DIM}));
  swap_out_id_desc->SetDataType(ids->GetDataType());

  swap_out_idx_desc->SetShape(GeShape({UNKNOWN_DIM}));
  swap_out_idx_desc->SetOriginShape(GeShape({UNKNOWN_DIM}));
  swap_out_idx_desc->SetDataType(output_type);

  if (ids_shape.GetShapeSize() != UNKNOWN_DIM) {
    std::vector<std::pair<int64_t, int64_t>> range;
    int64_t max_dim = ids_shape.GetDim(0);
    range.emplace_back(std::make_pair(1, max_dim));
    swap_in_id_desc->SetShapeRange(range);
    swap_in_idx_desc->SetShapeRange(range);
    swap_out_id_desc->SetShapeRange(range);
    swap_out_idx_desc->SetShapeRange(range);
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(CacheAdd, CacheAddInferShape);
// ---------------------CacheAdd END-------------------------------------

// --------------------------------CacheRemoteIndexToLocal--------------------------------
IMPLEMT_COMMON_INFERFUNC(CacheRemoteIndexToLocalInferShape) {
  TensorDesc desc = op.GetInputDesc(1);
  TensorDesc local_idx_desc = op.GetOutputDesc(0);
  Shape ids_shape = desc.GetShape();
  DataType output_type = DT_INT64;
  std::vector<std::pair<int64_t,int64_t>> range;
  auto status = desc.GetShapeRange(range);
  if (status != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  local_idx_desc.SetShapeRange(range);
  local_idx_desc.SetShape(ids_shape);
  local_idx_desc.SetDataType(output_type);
  return op.UpdateOutputDesc("local_idx", local_idx_desc);
}

COMMON_INFER_FUNC_REG(CacheRemoteIndexToLocal, CacheRemoteIndexToLocalInferShape);
// ---------------------CacheRemoteIndexToLocal END-------------------------------------

}  // namespace ge
