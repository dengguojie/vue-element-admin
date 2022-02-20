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
graphStatus SetShapeAndType(Operator& op, Shape element_shape, DataType element_dtype) {
  const char *op_name = op.GetName().c_str();
  std::vector<std::pair<int64_t, int64_t>> value_shape_range;
  op.GetInputDesc(0).GetShapeRange(value_shape_range);
  ShapeAndRange feed_shape_and_range = {element_shape, value_shape_range, element_dtype};
  if (SetShapeAndRange(op, feed_shape_and_range) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "SetShapeAndType: SetShapeAndRange failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus GetShapeAndType(Operator& op, ShapeAndType& out, bool& is_shape_and_type_empty, InferenceContextPtr infer_context) {
  const char *op_name = op.GetName().c_str();
  ShapeAndRange shape_and_range_out;
  std::vector<AscendString> marks;
  is_shape_and_type_empty = false;
  
  infer_context->GetMarks(marks);
  if (marks.empty()) {
    is_shape_and_type_empty = true;
    return GRAPH_FAILED;
  }
  bool geted = false;
  if (GetShapeAndRange(op, shape_and_range_out, geted, infer_context) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "GetShapeAndType: GetShapeAndRange failed.");
    return GRAPH_FAILED;
  }
  if (!geted) {
    OP_LOGE(op_name, "GetShapeAndType: GetShapeAndRange failed, marks is empty.");
    return GRAPH_FAILED; 
  }
  out.SetShape(shape_and_range_out.shape_);
  out.SetType(shape_and_range_out.shape_type_);
  return GRAPH_SUCCESS;
}

graphStatus TensorListConcatShapeInference(
    Operator &op, Shape element_shape, const char *op_name) {
  DataType element_dtype;
  if (op.GetAttr("element_dtype", element_dtype) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Get attr element_dtype failed.");
    return GRAPH_FAILED;
  }

  auto p_context = op.GetInferenceContext();
  if (p_context == nullptr) {
    OP_LOGE(op_name, "Get context failed, it is null.");
    return GRAPH_FAILED;
  }
  const auto& shapes_and_types = p_context->GetInputHandleShapesAndTypes();
  if (shapes_and_types.size() == 0) {
    OP_LOGE(op_name, "Get shapes_and_types failed, it's size is 0.");
    return GRAPH_FAILED;
  }
  auto handle_data = shapes_and_types[0];
  if (handle_data.size() > 1) {
    OP_LOGE(op_name, "Trying to read from list with wrong variant data.");
    return GRAPH_FAILED;
  }
  if (handle_data.size() == 1) {
    const ShapeAndType& list_shape_type = handle_data[0];
    if (list_shape_type.GetDataType() != element_dtype) {
      OP_LOGE(op_name,
              "List has type [%d], but expected type [%d].",
              list_shape_type.GetDataType(), element_dtype);
      return GRAPH_FAILED;
    }
    Shape merged;
    if (Merge(element_shape, list_shape_type.GetShape(), merged, op_name)
        != GRAPH_SUCCESS) {
      OP_LOGE(op_name, "Merge element_shape and list_shape_type failed.");
      return GRAPH_FAILED;
    }
    element_shape = merged;
  }
  TensorDesc output_tensor_desc = op.GetOutputDesc(0);
  TensorDesc output_length_desc = op.GetOutputDesc(1);
  if(RankKnown(element_shape)) {
    Shape result;
    if(SubShape(element_shape, 1, std::numeric_limits<int64_t>::max(),
                1, result, op_name) != GRAPH_SUCCESS) {
      OP_LOGE(op_name, "SubShape of element_shape failed.");
      return GRAPH_FAILED;
    }
    if (Concatenate(Shape({UNKNOWN_DIM}), result, result) != GRAPH_SUCCESS) {
      OP_LOGE(op_name, "Generate output_shape failed.");
      return GRAPH_FAILED;
    }
    output_tensor_desc.SetShape(result);
  } else {
    output_tensor_desc.SetShape(Shape(ge::UNKNOWN_RANK));
  }
  output_tensor_desc.SetDataType(element_dtype);
  if (op.UpdateOutputDesc("tensor", output_tensor_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Update output tensor desc failed.");
    return GRAPH_FAILED;
  }

  output_length_desc.SetShape(Shape({UNKNOWN_DIM}));
  output_length_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("lengths", output_length_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Update output lengths desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
graphStatus MergePrefix(const char *op_name, const Shape &s, const Shape &prefix,
                        Shape &s_out, Shape &prefix_out) {
  if (!RankKnown(prefix) || !RankKnown(s)) {
    s_out = s;
    prefix_out = prefix;
    return GRAPH_SUCCESS;
  }
  const int64_t rank = prefix.GetDimNum();

  // Merge the prefix dims and create the new output shapes.
  const int64_t rank_s = s.GetDimNum();
  std::vector<int64_t> dims;
  dims.reserve(std::max(rank, rank_s));
  dims.resize(rank);
  for (int64_t i = 0; i < rank; ++i) {
    int64_t dim_s = s.GetDim(i);
    int64_t dim_prefix = prefix.GetDim(i);
    if (Merge(dim_s, dim_prefix, dims[i])
        != GRAPH_SUCCESS) {
      OP_LOGE(op_name, "Merge two shapes failed.");
      return GRAPH_FAILED;
    }
  }
  prefix_out = Shape(dims);
  for (int64_t i = rank; i < rank_s; ++i) {
    dims.push_back(s.GetDim(i));
  }
  s_out = Shape(dims);
  return GRAPH_SUCCESS;
}

graphStatus MakeShapeFromShapeTensorTreatScalarAsUnknownShape(
    const Tensor& tensor, Shape& out, const char* op_name) {
  TensorDesc shape_data_desc = tensor.GetTensorDesc();
  Shape shape_data_shape = shape_data_desc.GetShape();
  std::vector<int64_t> dims = shape_data_shape.GetDims();
  DataType data_type = shape_data_desc.GetDataType();

  size_t rank_size = 1;
  if (!((dims.size() <= rank_size) || (dims == ge::UNKNOWN_SHAPE))) {
    OP_LOGE(op_name, "Shape's rank must be at most [%zu], but it is [%lu]",
            rank_size, dims.size());
    return GRAPH_FAILED;
  }

  if (dims.size() == 0) {
    if (data_type == DT_INT32) {
      const int32_t* shape_data = reinterpret_cast<const int32_t*>(tensor.GetData());
      if (tensor.GetSize() / sizeof(int32_t) > 0 && shape_data[0] != -1) {
        OP_LOGE(op_name, "If rank is 0, the value must be -1, but got [%d].",
                shape_data[0]);
        return GRAPH_FAILED;
      }
    } else if (data_type == DT_INT64) {
      const int64_t* shape_data = reinterpret_cast<const int64_t*>(tensor.GetData());
      if (tensor.GetSize() / sizeof(int64_t) > 0 && shape_data[0] != -1) {
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

  std::vector<AscendString> marks = {op_name};
  context->SetMarks(marks);
  OP_LOGI(op_name, "SetShapeAndType: shape = %s, dtype = %d.",
    DebugString(element_shape.GetDims()).c_str(), element_dtype);
  if (SetShapeAndType(op, element_shape, element_dtype) != GRAPH_SUCCESS) {
      OP_LOGE(op_name, "SetShapeAndType failed.");
  }

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

  bool is_shape_and_type_empty = false;
  std::vector<ShapeAndType> handle_data;
  const auto& shapes_and_types = p_context->GetInputHandleShapesAndTypes();
  if (shapes_and_types.size() == 0) {
    OP_LOGI(op_name, "Get shapes_and_types failed, it must not be equal 0.");
    ShapeAndType shape_and_type;
    if (GetShapeAndType(op, shape_and_type, is_shape_and_type_empty, p_context) != GRAPH_SUCCESS) {
      OP_LOGE(op_name, "GetShapeAndType get shape and type failed.");
      return GRAPH_FAILED;
    }
    handle_data.emplace_back(shape_and_type);
    OP_LOGI(op_name, "GetShapeAndType: shape = %s, dtype = %d.",
      DebugString(shape_and_type.GetShape().GetDims()).c_str(), shape_and_type.GetDataType());
  } else {
    handle_data = shapes_and_types[0];
    is_shape_and_type_empty = (shapes_and_types.empty()) ? true : false;
  }

  if (!is_shape_and_type_empty) {
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
  bool is_shape_and_type_empty = false;
  std::vector<ShapeAndType> handle_data;
  const auto& shapes_and_types = p_context->GetInputHandleShapesAndTypes();
  if (shapes_and_types.size() == 0) {
    OP_LOGI(op_name, "Get shapes_and_types failed, it must not be equal 0");
    ShapeAndType shape_and_type;
    if (GetShapeAndType(op, shape_and_type, is_shape_and_type_empty, p_context) != GRAPH_SUCCESS) {
      OP_LOGE(op_name, "GetShapeAndType get shape and type failed.");
      return GRAPH_FAILED;
    }
    handle_data.emplace_back(shape_and_type);
    OP_LOGI(op_name, "GetShapeAndType: shape = %s, dtype = %d.",
      DebugString(shape_and_type.GetShape().GetDims()).c_str(), shape_and_type.GetDataType());
  } else {
    handle_data = shapes_and_types[0];
    is_shape_and_type_empty = (shapes_and_types.empty()) ? true : false;
  }
  if (is_shape_and_type_empty) {
    std::vector<std::vector<ShapeAndType>> key_value_vec;
    std::vector<ShapeAndType> key_value;
    ShapeAndType key(element_shape, element_dtype);
    key_value.emplace_back(key);
    key_value_vec.emplace_back(key_value);
    p_context->SetOutputHandleShapesAndTypes(std::move(key_value_vec));

    OP_LOGI(op_name, "shape = %s, dtype = %d.", DebugString(element_shape.GetDims()).c_str(), element_dtype);
    if (SetShapeAndType(op, element_shape, element_dtype) != GRAPH_SUCCESS) {
        OP_LOGE(op_name, "shapes_and_types is empty, SetShapeAndType failed.");
    }
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

  OP_LOGI(op_name, "SetShapeAndType: shape = %s, dtype = %d.",
    DebugString(list_shape_type.GetShape().GetDims()).c_str(), element_dtype);
  if (SetShapeAndType(op, list_shape_type.GetShape(), element_dtype) != GRAPH_SUCCESS) {
      OP_LOGE(op_name, "SetShapeAndType failed.");
  }

  if (op.UpdateOutputDesc("output_handle", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Update handle desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorListSetItem, TensorListSetItemInfer);

IMPLEMT_INFERFUNC(TensorListPushBackBatch, TensorListPushBackBatchInfer) {
  const char *op_name = op.GetName().c_str();
  Shape input_handles;
  if (WithRank(op.GetInputDesc(0), 1, input_handles, op_name) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Input input_handles must be 1-D.");
    return GRAPH_FAILED;
  }
  Shape tensor;
  if (WithRankAtLeast(op.GetInputDesc(1), 1, tensor, op_name) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Input tensor must be at least 1-D.");
    return GRAPH_FAILED;
  }
  if (MergePrefix(op_name, tensor, input_handles, tensor, input_handles)
      != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Merge two input tensor failed.");
    return GRAPH_FAILED;
  }

  TensorDesc output_desc = op.GetOutputDesc(0);
  output_desc.SetShape(input_handles);
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

  if (op.UpdateOutputDesc("output_handles", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Update output handle desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(TensorListPushBackBatch, TensorListPushBackBatchInfer);

IMPLEMT_INFERFUNC(TensorListStack, TensorListStackInfer) {
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
  bool is_shape_and_type_empty = false;
  std::vector<ShapeAndType> handle_data;
  const auto& shapes_and_types = p_context->GetInputHandleShapesAndTypes();
  if (shapes_and_types.size() == 0) {
    OP_LOGI(op_name, "Get shapes_and_types failed, it must not be equal 0.");
    ShapeAndType shape_and_type;
    if (GetShapeAndType(op, shape_and_type, is_shape_and_type_empty, p_context) != GRAPH_SUCCESS) {
      OP_LOGE(op_name, "GetShapeAndType get shape and type failed.");
      return GRAPH_FAILED;
    }
    handle_data.emplace_back(shape_and_type);
    OP_LOGI(op_name, "GetShapeAndType: shape = %s, dtype = %d.",
      DebugString(shape_and_type.GetShape().GetDims()).c_str(), shape_and_type.GetDataType());
  } else {
    handle_data = shapes_and_types[0];
    is_shape_and_type_empty = (shapes_and_types.empty()) ? true : false;
  }

  if (!is_shape_and_type_empty && (handle_data.size() > 1)) {
      OP_LOGE(op_name, "Trying to read from list with wrong variant data.");
      return GRAPH_FAILED;
  }
  if (!is_shape_and_type_empty && (handle_data.size() == 1)) {
    const ShapeAndType& list_shape_type = handle_data[0];
    if (list_shape_type.GetDataType() != element_dtype) {
      OP_LOGE(op_name, "List has type [%d], but expected type [%d].", list_shape_type.GetDataType(), element_dtype);
      return GRAPH_FAILED;
    }
    Shape ignored;
    if (Merge(element_shape, list_shape_type.GetShape(), ignored, op_name) != GRAPH_SUCCESS) {
      OP_LOGE(op_name, "Merge element_shape and list_shape_type failed.");
      return GRAPH_FAILED;
    }
    element_shape = list_shape_type.GetShape();
  }
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  // unknown shape support
  op_desc->SetOpInferDepends({"element_shape"});
  Tensor input_tensor;
  Shape element_shape_input(UNKNOWN_RANK);
  if (op.GetInputConstData("element_shape", input_tensor) != GRAPH_SUCCESS) {
    OP_LOGW(op_name, "Get const data of input failed, return unknown shape.");
  } else {
    if (MakeShapeFromShapeTensorTreatScalarAsUnknownShape(
          input_tensor, element_shape_input, op_name) != GRAPH_SUCCESS) {
      OP_LOGE(op_name, "Do makeShapeFromShapeTensorTreatScalarAsUnknownShape failed.");
      return GRAPH_FAILED;
    }
  }

  if (Merge(element_shape, element_shape_input, element_shape, op_name) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Merge element_shape and element_shape_input failed.");
    return GRAPH_FAILED;
  }
  int expected_num_elements = -1;
  if (op.GetAttr("num_elements", expected_num_elements) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Get attr num_elements failed.");
    return GRAPH_FAILED;
  }
  Shape num_elements;
  if (expected_num_elements == -1) {
    num_elements = Shape({UNKNOWN_DIM});
  } else {
    num_elements = Shape({expected_num_elements});
  }

  Shape result;
  if (Concatenate(num_elements, element_shape, result) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Generate output_shape failed.");
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDesc(0);
  output_desc.SetShape(result);
  output_desc.SetDataType(element_dtype);
  if (op.UpdateOutputDesc("tensor", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Update output tensor desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(TensorListStack, TensorListStackInfer);

IMPLEMT_INFERFUNC(TensorListConcatV2, TensorListConcatV2Infer) {
  const char *op_name = op.GetName().c_str();
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  // unknown shape support
  op_desc->SetOpInferDepends({"element_shape"});
  Tensor input_tensor;
  Shape element_shape(UNKNOWN_RANK);
  if (op.GetInputConstData("element_shape", input_tensor) != GRAPH_SUCCESS) {
    OP_LOGW(op_name, "Get const data of input failed, return unknown shape.");
  } else {
    if (MakeShapeFromShapeTensorTreatScalarAsUnknownShape(
          input_tensor, element_shape, op_name) != GRAPH_SUCCESS) {
      OP_LOGE(op_name,
              "Do makeShapeFromShapeTensorTreatScalarAsUnknownShape failed.");
      return GRAPH_FAILED;
    }
  }
  if (TensorListConcatShapeInference(op, element_shape, op_name)
      != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Get tensorlist concat shape failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(TensorListConcatV2, TensorListConcatV2Infer);

IMPLEMT_INFERFUNC(TensorListSplit, TensorListSplitInfer) {
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

  Shape tensor_shape = op.GetInputDesc(0).GetShape();
  Shape ignored;
  if (WithRankAtLeast(op.GetInputDesc(0), 1, ignored, op_name) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Input tensor shape must be at least 1-D.");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(2), 1, ignored, op_name) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Input length shape must be 1D.");
    return GRAPH_FAILED;
  }

  Shape element_shape_from_tensor_shape;
  if(SubShape(tensor_shape, 1, std::numeric_limits<int64_t>::max(),
              1, element_shape_from_tensor_shape, op_name) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "SubShape of input tensor failed.");
    return GRAPH_FAILED;
  }
  if (Concatenate(Shape({UNKNOWN_DIM}), element_shape_from_tensor_shape,
                  element_shape_from_tensor_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Generate element shape from tensor shape failed.");
    return GRAPH_FAILED;
  }
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  // unknown shape support
  op_desc->SetOpInferDepends({"element_shape"});
  Tensor input_tensor;
  Shape element_shape(UNKNOWN_RANK);
  if (op.GetInputConstData("element_shape", input_tensor) != GRAPH_SUCCESS) {
    OP_LOGW(op_name, "Get const data of input failed, return unknown shape.");
  } else {
    if (MakeShapeFromShapeTensorTreatScalarAsUnknownShape(
          input_tensor, element_shape, op_name) != GRAPH_SUCCESS) {
      OP_LOGE(op_name,
              "Do makeShapeFromShapeTensorTreatScalarAsUnknownShape failed.");
      return GRAPH_FAILED;
    }
  }

  if (Merge(element_shape_from_tensor_shape, element_shape,
            element_shape_from_tensor_shape, op_name)
      != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Merge element_shape and element_shape_from_tensor_shape failed.");
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

  if (op.UpdateOutputDesc("output_handle", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Update output output_handle desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(TensorListSplit, TensorListSplitInfer);

IMPLEMT_INFERFUNC(TensorListFromTensor, TensorListFromTensorInfer) {
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

  Shape tensor_shape = op.GetInputDesc(0).GetShape();
  Shape tensor_shape_except_first_dim;

  if(SubShape(tensor_shape, 1, std::numeric_limits<int64_t>::max(),
              1, tensor_shape_except_first_dim, op_name) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "SubShape of input tensor failed.");
    return GRAPH_FAILED;
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  // unknown shape support
  op_desc->SetOpInferDepends({"element_shape"});
  Tensor input_tensor;
  Shape element_shape(UNKNOWN_RANK);
  if (op.GetInputConstData("element_shape", input_tensor) != GRAPH_SUCCESS) {
    OP_LOGW(op_name, "Get const data of input failed, return unknown shape.");
  } else {
    if (MakeShapeFromShapeTensorTreatScalarAsUnknownShape(
          input_tensor, element_shape, op_name) != GRAPH_SUCCESS) {
      OP_LOGE(op_name,
              "Do makeShapeFromShapeTensorTreatScalarAsUnknownShape failed.");
      return GRAPH_FAILED;
    }
  }

  if (Merge(tensor_shape_except_first_dim, element_shape,
            tensor_shape_except_first_dim, op_name)
      != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Merge element_shape and tensor_shape_except_first_dim failed.");
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

  std::vector<AscendString> marks = {op_name};
  context->SetMarks(marks);

  OP_LOGI(op_name, "SetShapeAndType: shape = %s, dtype = %d.",
    DebugString(element_shape.GetDims()).c_str(), element_dtype);
  if (SetShapeAndType(op, element_shape, element_dtype) != GRAPH_SUCCESS) {
      OP_LOGE(op_name, "SetShapeAndType failed.");
  }

  if (op.UpdateOutputDesc("output_handle", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Update output output_handle desc failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(TensorListFromTensor, TensorListFromTensorInfer);

IMPLEMT_INFERFUNC(TensorListResize, TensorListResizeInfer) {
  const char *op_name = op.GetName().c_str();
  Shape unused;
  if (WithRank(op.GetInputDesc(1), 0, unused, op_name) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Input size shape must be 0D.");
    return GRAPH_FAILED;
  }
  Shape shape;
  if (Scalar(shape) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Create Scalar failed.");
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDesc(0);
  output_desc.SetShape(shape);
  output_desc.SetDataType(DT_VARIANT);

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
    p_context->SetOutputHandleShapesAndTypes(shapes_and_types);
  }

  if (op.UpdateOutputDesc("output_handle", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Update output output_handle desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(TensorListResize, TensorListResizeInfer);

IMPLEMT_INFERFUNC(TensorListGather, TensorListGatherInfer) {
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

  Shape out;
  if (Concatenate(op.GetInputDesc(1).GetShape(), element_shape,
                  out) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Generate element shape from tensor shape failed.");
    return GRAPH_FAILED;
  }

  TensorDesc output_desc = op.GetOutputDesc(0);
  output_desc.SetShape(out);
  output_desc.SetDataType(element_dtype);
  if (op.UpdateOutputDesc("values", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Update values desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(TensorListGather, TensorListGatherInfer);

IMPLEMT_INFERFUNC(TensorListScatterV2, TensorListScatterV2Infer) {
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
  Shape element_shape;
  if (op.GetInputConstData("element_shape", input_tensor) != GRAPH_SUCCESS) {
    OP_LOGW(op_name, "Get const data of input failed, return unknown shape.");
    element_shape = Shape(ge::UNKNOWN_RANK);
  }
  else {
    if (MakeShapeFromShapeTensorTreatScalarAsUnknownShape(
          input_tensor, element_shape, op_name) != GRAPH_SUCCESS) {
      OP_LOGE(op_name,
              "Do makeShapeFromShapeTensorTreatScalarAsUnknownShape failed.");
      return GRAPH_FAILED;
    }
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
  if (op.UpdateOutputDesc("output_handle", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Update output_handle desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(TensorListScatterV2, TensorListScatterV2Infer);

IMPLEMT_INFERFUNC(TensorListScatterIntoExistingList, TensorListScatterIntoExistingListInfer) {
  const char *op_name = op.GetName().c_str();
  Shape ignored;
  if (WithRankAtLeast(op.GetInputDesc(1), 1, ignored, op_name) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Input tensor must be at least 1-D.");
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(2), 1, ignored, op_name) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Input indices must be 1-D.");
    return GRAPH_FAILED;
  }

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
  if ((!shapes_and_types.empty()) && (handle_data.size() != 1)) {
    OP_LOGE(op_name, "Expected length of shape_and_types is 1, but got [%zu].",
            handle_data.size());
    return GRAPH_FAILED;
  }
  if ((!shapes_and_types.empty()) && (handle_data.size() == 1)) {
    const ShapeAndType& list_shape_type = handle_data[0];
    if (list_shape_type.GetDataType() != element_dtype) {
      OP_LOGE(op_name,
              "Expected List type is [%d], but got element with type [%d].",
              element_dtype, list_shape_type.GetDataType());
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

  Shape shape;
  if (Scalar(shape) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Create Scalar failed.");
    return GRAPH_FAILED;
  }
  TensorDesc output_desc = op.GetOutputDesc(0);
  output_desc.SetShape(shape);
  output_desc.SetDataType(DT_VARIANT);
  if (op.UpdateOutputDesc("output_handle", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Update output handle desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(TensorListScatterIntoExistingList, TensorListScatterIntoExistingListInfer);


IMPLEMT_INFERFUNC(TensorListConcatLists, TensorListConcatListsInfer) {
  const char *op_name = op.GetName().c_str();
  auto input_a = op.GetInputDesc(0).GetShape();
  auto input_b = op.GetInputDesc(1).GetShape();

  if (Merge(input_a, input_b, input_a, op_name)
      != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Merge input_a and input_b failed.");
    return GRAPH_FAILED;
  }

  TensorDesc output_desc = op.GetOutputDesc(0);
  output_desc.SetShape(input_a);
  output_desc.SetDataType(DT_VARIANT);
  if (op.UpdateOutputDesc("output", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Update output desc failed.");
    return GRAPH_FAILED;
  }

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
  auto handle_data_a = shapes_and_types[0];
  auto handle_data_b = shapes_and_types[1];

  if (handle_data_a.empty() && handle_data_b.empty()) {
    std::vector<std::vector<ShapeAndType>> key_value_vec;
    std::vector<ShapeAndType> key_value;
    ShapeAndType key(element_shape, element_dtype);
    key_value.emplace_back(key);
    key_value_vec.emplace_back(key_value);
    p_context->SetOutputHandleShapesAndTypes(std::move(key_value_vec));
    return GRAPH_SUCCESS;
  }

  const ShapeAndType& list_shape_type_a =
    handle_data_a.empty() ? handle_data_b[0] : handle_data_a[0];
  if (list_shape_type_a.GetDataType() != element_dtype) {
    OP_LOGE(op_name,
            "Expected input_a type is [%d], but got element with type [%d].",
            element_dtype, list_shape_type_a.GetDataType());
    return GRAPH_FAILED;
  }
  const ShapeAndType& list_shape_type_b =
        handle_data_b.empty() ? handle_data_a[0] : handle_data_b[0];
  if (list_shape_type_b.GetDataType() != element_dtype) {
    OP_LOGE(op_name,
            "Expected input_b type is [%d], but got element with type [%d].",
            element_dtype, list_shape_type_b.GetDataType());
    return GRAPH_FAILED;
  }
  Shape out;
  if (Merge(list_shape_type_a.GetShape(), list_shape_type_b.GetShape(),
      out, op_name)
      != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Merge two list failed.");
    return GRAPH_FAILED;
  }
  std::vector<std::vector<ShapeAndType>> key_value_vec;
  std::vector<ShapeAndType> key_value;
  ShapeAndType key(out, element_dtype);
  key_value.emplace_back(key);
  key_value_vec.emplace_back(key_value);
  p_context->SetOutputHandleShapesAndTypes(std::move(key_value_vec));

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(TensorListConcatLists, TensorListConcatListsInfer);

}  // namespace ge
