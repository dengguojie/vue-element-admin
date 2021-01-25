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
 * \file list_ops.cc
 * \brief
 */
#include "inc/list_ops.h"
#include "graph/operator.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "util/error_util.h"
#include "util/common_shape_fns.h"
#include "util/lookup_ops_shape_fns.h"

namespace ge {
namespace {
graphStatus MakeShapeFromShapeTensorTreatScalarAsUnknownShape(
    const Tensor& tensor, Shape& out, const char* op_name) {
  TensorDesc shape_data_desc = tensor.GetTensorDesc();
  Shape shape_data_shape = shape_data_desc.GetShape();
  std::vector<int64_t> dims = shape_data_shape.GetDims();
  DataType data_type = shape_data_desc.GetDataType();

  size_t rank_size = 1;
  if (!((dims.size() <= rank_size) || (dims == ge::UNKNOWN_SHAPE))) {
    OP_LOGE(op_name, "Shape's rank must be at most [%zu], but it is [%u]",
            rank_size, dims.size());
    return GRAPH_FAILED;
  }

  if (dims.size() == 0) {
    if (data_type == DT_INT32) {
      const int32_t* shape_data = reinterpret_cast<const int32_t*>(tensor.GetData());
      if (shape_data[0] != -1) {
        OP_LOGE(op_name, "If rank is 0, the value must be -1, but got [%d].",
                shape_data[0]);
        return GRAPH_FAILED;
      }
    } else if (data_type == DT_INT64) {
      const int64_t* shape_data = reinterpret_cast<const int64_t*>(tensor.GetData());
      if (shape_data[0] != -1) {
        OP_LOGE(op_name, "If rank is 0, the value must be -1, but got [%ld]",
                shape_data[0]);
        return GRAPH_FAILED;
      }
    } else {
      OP_LOGE(op_name, "Data type invalid, should be DT_INT32 or DT_INT64,"
        "but got [%d].", data_type);
      return GRAPH_FAILED;
    }
    out = Shape(ge::UNKNOWN_RANK);
    return GRAPH_SUCCESS;
  }

  if (MakeShapeFromShapeTensor(tensor, out, op_name) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "MakeShapeFromShapeTensor failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}
} // namespace

IMPLEMT_INFERFUNC(EmptyTensorList, EmptyTensorListInfer) {
  const char *op_name = op.GetName().c_str();
  Shape shape;
  if (Scalar(shape) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Create Scalar failed.");
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDesc(0);
  output_desc.SetShape(shape);
  output_desc.SetDataType(DT_VARIANT);
  DataType element_dtype;
  if (op.GetAttr("element_dtype", element_dtype) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Get attr element_dtype failed.");
    return GRAPH_FAILED;
  }
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  // unknown shape support
  op_desc->SetOpInferDepends({"element_shape"});
  Tensor input_tensor;
  Shape shape_handle;
  if (op.GetInputConstData("element_shape", input_tensor) != GRAPH_SUCCESS) {
    OP_LOGW(op_name, "Get const data of input failed, return unknown shape.");
    shape_handle = Shape(ge::UNKNOWN_RANK);
  }
  else {
    if (MakeShapeFromShapeTensorTreatScalarAsUnknownShape(
          input_tensor, shape_handle, op_name) != GRAPH_SUCCESS) {
      OP_LOGE(op_name,
              "Do makeShapeFromShapeTensorTreatScalarAsUnknownShape failed.");
      return GRAPH_FAILED;
    }
  }
  ShapeAndType shape_and_type(shape_handle, element_dtype);
  std::vector<ShapeAndType> handle_shapes_and_types;
  handle_shapes_and_types.reserve(1);
  handle_shapes_and_types.emplace_back(shape_and_type);
  std::vector<std::vector<ShapeAndType>> shapes_and_types(2);
  shapes_and_types[0] = handle_shapes_and_types;
  auto context = op.GetInferenceContext();
  if (context == nullptr) {
    OP_LOGE(op_name, "Get context failed, it is null.");
    return GRAPH_FAILED;
  }
  context->SetOutputHandleShapesAndTypes(shapes_and_types);
  if (op.UpdateOutputDesc("handle", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Update handle desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(EmptyTensorList, EmptyTensorListInfer);

IMPLEMT_INFERFUNC(TensorListPushBack, TensorListPushBackInfer) {
  const char *op_name = op.GetName().c_str();
  Shape shape;
  if (Scalar(shape) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Create Scalar failed.");
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDesc(0);
  output_desc.SetShape(shape);
  output_desc.SetDataType(DT_VARIANT);

  DataType element_dtype;
  if (op.GetAttr("element_dtype", element_dtype) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Get attr element_dtype failed.");
    return GRAPH_FAILED;
  }

  Shape element_shape(UNKNOWN_RANK);
  auto p_context = op.GetInferenceContext();
  if (p_context == nullptr) {
    OP_LOGE(op_name, "Get context failed, it is null.");
    return GRAPH_FAILED;
  }
  const auto& shapes_and_types = p_context->GetInputHandleShapesAndTypes();
  if (shapes_and_types.size() == 0) {
    OP_LOGE(op_name, "Get shapes_and_types failed, it must not be equal 0.");
    return GRAPH_FAILED;
  }
  auto handle_data = shapes_and_types[0];
  if ((!shapes_and_types.empty()) && (handle_data.size() > 1)) {
    OP_LOGE(op_name, "Trying to push to list with wrong variant data.");
    return GRAPH_FAILED;
  }
  if ((!shapes_and_types.empty()) && (handle_data.size() == 1)) {
    const ShapeAndType& list_shape_type = handle_data[0];
    if (list_shape_type.GetDataType() != element_dtype) {
      OP_LOGE(op_name,
              "List has type [%d], but trying to push element with type [%d].",
              list_shape_type.GetDataType(), element_dtype);
      return GRAPH_FAILED;
    }
    Shape ignored;
    if (Merge(element_shape, list_shape_type.GetShape(), ignored, op_name)
        != GRAPH_SUCCESS) {
      OP_LOGE(op_name, "Merge element_shape and list_shape_type failed.");
      return GRAPH_FAILED;
    }
    element_shape = list_shape_type.GetShape();
  }

  std::vector<std::vector<ShapeAndType>> key_value_vec;
  std::vector<ShapeAndType> key_value;
  ShapeAndType key(element_shape, element_dtype);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  p_context->SetOutputHandleShapesAndTypes(std::move(key_value_vec));

  if (op.UpdateOutputDesc("output_handle", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Update output handle desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(TensorListPushBack, TensorListPushBackInfer);

IMPLEMT_INFERFUNC(TensorListPopBack, TensorListPopBackInfer) {
  const char *op_name = op.GetName().c_str();
  Shape shape;
  if (Scalar(shape) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Create Scalar failed.");
    return GRAPH_FAILED;
  }
  TensorDesc output0_desc = op.GetOutputDesc(0);
  output0_desc.SetShape(shape);
  output0_desc.SetDataType(DT_VARIANT);

  DataType element_dtype;
  if (op.GetAttr("element_dtype", element_dtype) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Get attr element_dtype failed.");
    return GRAPH_FAILED;
  }

  Shape tensor_shape(UNKNOWN_RANK);
  auto p_context = op.GetInferenceContext();
  if (p_context == nullptr) {
    OP_LOGE(op_name, "Get context failed, it is null.");
    return GRAPH_FAILED;
  }
  const auto& shapes_and_types = p_context->GetInputHandleShapesAndTypes();
  if (shapes_and_types.size() == 0) {
    OP_LOGE(op_name, "Get shapes_and_types failed, it must not be equal 0.");
    return GRAPH_FAILED;
  }
  auto handle_data = shapes_and_types[0];
  if ((!shapes_and_types.empty()) && (handle_data.size() > 1)){
    OP_LOGE(op_name, "Trying to read from list with invalid variant data.");
    return GRAPH_FAILED;
  }
  if ((!shapes_and_types.empty()) && (handle_data.size() == 1)){
    const ShapeAndType& list_shape_type = handle_data[0];
    if (list_shape_type.GetDataType() != element_dtype) {
      OP_LOGE(op_name,
              "List has type [%d], but trying to push element with type [%d].",
              list_shape_type.GetDataType(), element_dtype);
      return GRAPH_FAILED;
    }
    Shape ignored;
    if (Merge(tensor_shape, list_shape_type.GetShape(), ignored, op_name)
        != GRAPH_SUCCESS) {
      OP_LOGE(op_name, "Merge tensor_shape and list_shape_type failed.");
      return GRAPH_FAILED;
    }
    p_context->SetOutputHandleShapesAndTypes(shapes_and_types);
    tensor_shape = list_shape_type.GetShape();
  }
  TensorDesc output1_desc = op.GetOutputDesc(1);
  output1_desc.SetShape(tensor_shape);
  output1_desc.SetDataType(element_dtype);

  if (op.UpdateOutputDesc("output_handle", output0_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Update output handle desc failed.");
    return GRAPH_FAILED;
  }
  if (op.UpdateOutputDesc("tensor", output1_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Update output tensor desc failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(TensorListPopBack, TensorListPopBackInfer);

IMPLEMT_INFERFUNC(TensorListLength, TensorListLengthInfer) {
  Shape scalar_shape;
  (void)Scalar(scalar_shape);
  TensorDesc output_desc = op.GetOutputDesc("length");
  output_desc.SetShape(scalar_shape);
  output_desc.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("length", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Update length desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(TensorListLength, TensorListLengthInfer);

IMPLEMT_INFERFUNC(TensorListElementShape, TensorListElementShapeInfer) {
  const char *op_name = op.GetName().c_str();
  TensorDesc output_desc = op.GetOutputDesc(0);
  DataType type;
  if (op.GetAttr("shape_type", type) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Get attr shape_type failed.");
    return GRAPH_FAILED;
  }
  auto p_context = op.GetInferenceContext();
  if (p_context == nullptr) {
    OP_LOGE(op_name, "Get context failed, it is null.");
    return GRAPH_FAILED;
  }
  const auto& shapes_and_types = p_context->GetInputHandleShapesAndTypes();
  if (shapes_and_types.size() == 0) {
    OP_LOGE(op_name, "Get shapes_and_types failed, it must not be equal 0.");
    return GRAPH_FAILED;
  }
  auto handle_data = shapes_and_types[0];
  if (shapes_and_types.empty()) {
    output_desc.SetShape(Shape({UNKNOWN_DIM}));
    output_desc.SetDataType(type);
  } else {
    const ShapeAndType& list_shape_type = handle_data[0];
    if (list_shape_type.GetShape().GetDims() == UNKNOWN_RANK) {
      output_desc.SetShape(Shape(UNKNOWN_RANK));
    } else {
      std::vector<int64_t> dims;
      dims.push_back(list_shape_type.GetShape().GetDimNum());
      output_desc.SetShape(Shape(dims));
    }
    output_desc.SetDataType(type);
  }

  if (op.UpdateOutputDesc("element_shape", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Update element shape desc failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(TensorListElementShape, TensorListElementShapeInfer);

IMPLEMT_INFERFUNC(TensorListReserve, TensorListReserveInfer) {
  const char *op_name = op.GetName().c_str();
  Shape shape;
  if (Scalar(shape) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Create Scalar failed.");
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDesc(0);
  output_desc.SetShape(shape);
  output_desc.SetDataType(DT_VARIANT);

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  // unknown shape support
  op_desc->SetOpInferDepends({"element_shape"});

  Tensor input_tensor;
  Shape element_shape;
  if (op.GetInputConstData("element_shape", input_tensor) != GRAPH_SUCCESS) {
    OP_LOGW(op_name, "Get const data of input failed, return unknown shape.");
    element_shape = Shape(ge::UNKNOWN_RANK);
  } else {
    if (MakeShapeFromShapeTensorTreatScalarAsUnknownShape(
          input_tensor, element_shape, op_name) != GRAPH_SUCCESS) {
      OP_LOGE(op_name,
              "Do makeShapeFromShapeTensorTreatScalarAsUnknownShape failed.");
      return GRAPH_FAILED;
    }
  }

  DataType element_dtype;
  if (op.GetAttr("element_dtype", element_dtype) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Get attr element_dtype failed.");
    return GRAPH_FAILED;
  }

  ShapeAndType shape_and_type(element_shape, element_dtype);
  std::vector<ShapeAndType> handle_shapes_and_types;
  handle_shapes_and_types.reserve(1);
  handle_shapes_and_types.emplace_back(shape_and_type);
  std::vector<std::vector<ShapeAndType>> shapes_and_types(2);
  shapes_and_types[0] = handle_shapes_and_types;
  auto context = op.GetInferenceContext();
  if (context == nullptr) {
    OP_LOGE(op_name, "Get context failed, it is null.");
    return GRAPH_FAILED;
  }
  context->SetOutputHandleShapesAndTypes(shapes_and_types);

  if (op.UpdateOutputDesc("handle", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Update handle desc failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(TensorListReserve, TensorListReserveInfer);

IMPLEMT_INFERFUNC(TensorListGetItem, TensorListGetItemInfer) {
  const char *op_name = op.GetName().c_str();
  DataType element_dtype;
  if (op.GetAttr("element_dtype", element_dtype) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Get attr element_dtype failed.");
    return GRAPH_FAILED;
  }

  Shape element_shape(UNKNOWN_RANK);
  auto p_context = op.GetInferenceContext();
  if (p_context == nullptr) {
    OP_LOGE(op_name, "Get context failed, it is null.");
    return GRAPH_FAILED;
  }
  const auto& shapes_and_types = p_context->GetInputHandleShapesAndTypes();
  if (shapes_and_types.size() == 0) {
    OP_LOGE(op_name, "Get shapes_and_types failed, it must not be equal 0.");
    return GRAPH_FAILED;
  }
  auto handle_data = shapes_and_types[0];
  if (!shapes_and_types.empty()) {
    const ShapeAndType& list_shape_type = handle_data[0];
    element_shape = list_shape_type.GetShape();
    if (list_shape_type.GetDataType() != element_dtype) {
      OP_LOGE(op_name,
              "Expected list with element dtype [%d], but got [%d].",
              element_dtype, list_shape_type.GetDataType());
      return GRAPH_FAILED;
    }
  }
  Tensor input_tensor;
  Shape element_shape_input(UNKNOWN_RANK);
  if (op.GetInputConstData("element_shape", input_tensor) != GRAPH_SUCCESS) {
    OP_LOGW(op_name, "Get const data of input failed, return unknown shape.");
  } else {
    if (MakeShapeFromShapeTensorTreatScalarAsUnknownShape(
          input_tensor, element_shape_input, op_name) != GRAPH_SUCCESS) {
      OP_LOGE(op_name,
              "Do makeShapeFromShapeTensorTreatScalarAsUnknownShape failed.");
      return GRAPH_FAILED;
    }
  }

  if (Merge(element_shape, element_shape_input, element_shape, op_name)
      != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Merge element_shape and list_shape_type failed.");
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDesc(0);
  output_desc.SetShape(element_shape);
  output_desc.SetDataType(element_dtype);
  if (op.UpdateOutputDesc("item", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Update item desc failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(TensorListGetItem, TensorListGetItemInfer);

IMPLEMT_INFERFUNC(TensorListSetItem, TensorListSetItemInfer) {
  const char *op_name = op.GetName().c_str();
  Shape shape;
  if (Scalar(shape) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Create Scalar failed.");
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDesc(0);
  output_desc.SetShape(shape);
  output_desc.SetDataType(DT_VARIANT);

  DataType element_dtype;
  if (op.GetAttr("element_dtype", element_dtype) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Get attr element_dtype failed.");
    return GRAPH_FAILED;
  }

  Shape element_shape(UNKNOWN_RANK);
  auto p_context = op.GetInferenceContext();
  if (p_context == nullptr) {
    OP_LOGE(op_name, "Get context failed, it is null.");
    return GRAPH_FAILED;
  }
  const auto& shapes_and_types = p_context->GetInputHandleShapesAndTypes();
  if (shapes_and_types.size() == 0) {
    OP_LOGE(op_name, "Get shapes_and_types failed, it must not be equal 0.");
    return GRAPH_FAILED;
  }
  auto handle_data = shapes_and_types[0];
  if (shapes_and_types.empty()) {
    std::vector<std::vector<ShapeAndType>> key_value_vec;
    std::vector<ShapeAndType> key_value;
    ShapeAndType key(element_shape, element_dtype);
    key_value.emplace_back(key);
    key_value_vec.emplace_back(key_value);
    p_context->SetOutputHandleShapesAndTypes(std::move(key_value_vec));

    return GRAPH_SUCCESS;
  }

  const ShapeAndType& list_shape_type = handle_data[0];
  Shape item_shape = op.GetInputDesc(2).GetShape();
  if (Merge(item_shape, list_shape_type.GetShape(), item_shape, op_name)
      != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Merge item_shape and list_shape_type failed.");
    return GRAPH_FAILED;
  }
  p_context->SetOutputHandleShapesAndTypes(shapes_and_types);

  if (op.UpdateOutputDesc("output_handle", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Update handle desc failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorListSetItem, TensorListSetItemInfer);

}  // namespace ge
