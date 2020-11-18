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
 * \file transformation_ops.cpp
 * \brief
 */
#ifdef CHECK_FORMAT
#undef CHECK_FORMAT
#endif

#define CHECK_FORMAT(format)                                                     \
  {                                                                              \
    if (ge::FORMAT_RESERVED == format) {                                         \
      OP_LOGE(op.GetName().c_str(), "get format failed:%s:%d", #format, format); \
      return GRAPH_FAILED;                                                       \
    }                                                                            \
  }

#include "inc/transformation_ops.h"
#include <string>
#include <vector>
#include "util/util.h"
#include "common_shape_fns.h"
#include "op_log.h"
#include "util/error_util.h"
#include "register/infer_data_slice_registry.h"
#include "graph/common_error_codes.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
// ----------------Bitcast Op Start-------------------
graphStatus CalcAndUpdateShape(const Operator& op, vector<int64_t>& dim_vec, ge::DataType ori_data_type,
                               ge::DataType dst_data_type) {
  if (dim_vec.size() == 0) {
    OP_LOGE(op.GetName().c_str(), "input_desc shape size is zero.");
    return GRAPH_FAILED;
  }
  int64_t ori_data_size = GetSizeByDataType(ori_data_type);
  int64_t dst_data_size = GetSizeByDataType(dst_data_type);
  if (ori_data_size == dst_data_size) {
    return GRAPH_SUCCESS;
  } else if (ori_data_size > dst_data_size) {
    if (ori_data_size % dst_data_size != 0) {
      OP_LOGE(op.GetName().c_str(), "ori_data_size is not divisible by dst_data_size..");
      return GRAPH_FAILED;
    }
    dim_vec.push_back(ori_data_size / dst_data_size);
    return GRAPH_SUCCESS;
  } else {
    if (dst_data_size % ori_data_size != 0) {
      OP_LOGE(op.GetName().c_str(), "dst_data_size is not divisible by ori_data_size.");
      return GRAPH_FAILED;
    }

    if (dim_vec[dim_vec.size() - 1] != (dst_data_size / ori_data_size)) {
      OP_LOGE(op.GetName().c_str(), "The last dim is not equal to dst_data_size / ori_data_size.");
      return GRAPH_FAILED;
    }
    dim_vec.pop_back();
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Bitcast, BitcastVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(BitcastInfer) {
  auto input_desc = op.GetInputDesc("x");
  auto input_shape_dims = input_desc.GetShape().GetDims();
  auto input_type = input_desc.GetDataType();

  Operator::OpType out_type;
  if (op.GetAttr("type", out_type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Bitcast: get attr type failed");
    return GRAPH_FAILED;
  }
  if (input_type >= ge::DT_UNDEFINED || out_type >= ge::DT_UNDEFINED) {
    OP_LOGE(op.GetName().c_str(), "Bitcast: input_type[%d] or out_type[%d]  is not valid.", input_type, out_type);
    return GRAPH_FAILED;
  }

  // calculate dest shape
  vector<int64_t> output_shape(input_shape_dims);
  if (CalcAndUpdateShape(op, output_shape, input_type, out_type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Bitcast: calculate and update shape failed");
    return GRAPH_FAILED;
  }

  TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(Shape(output_shape));
  output_desc.SetDataType(out_type);

  graphStatus output_status = op.UpdateOutputDesc("y", output_desc);
  if (output_status != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update output_desc failed");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(Bitcast, BitcastVerify);
COMMON_INFER_FUNC_REG(Bitcast, BitcastInfer);
// ----------------Bitcast Op End-------------------

// ----------------DepthwiseWeight6DTo4D Op-------------------
bool SixToFourInferShapeAndType(const ge::Operator& op, ge::TensorDesc& vOutputDesc) {
  int64_t columnNum;
  if (op.GetAttr("channel_size", columnNum) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  auto inputDesc = op.GetInputDesc("x");
  int64_t dimSize = inputDesc.GetShape().GetDimNum();

  vector<int64_t> outShape;
  if (dimSize == 6) {
    outShape.push_back(inputDesc.GetShape().GetDim(1));
    outShape.push_back(inputDesc.GetShape().GetDim(2));
    outShape.push_back(columnNum);
    outShape.push_back(inputDesc.GetShape().GetDim(3));
    ge::Shape outputShape = ge::Shape(outShape);
    vOutputDesc.SetShape(outputShape);
    vOutputDesc.SetDataType(op.GetInputDesc("x").GetDataType());
  } else {
    OP_LOGE(op.GetName().c_str(), "please make sure that the dim of inputshape is 6!");
    return false;
  }
  return true;
}

// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(DepthwiseWeight6DTo4D, DepthwiseWeight6DTo4DVerify) {
  OP_LOGI(op.GetName().c_str(), "enter op_proto verifier function!!!");
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DepthwiseWeight6DTo4DInferShape) {
  OP_LOGI(op.GetName().c_str(), "enter op_proto inferfunction!!!");
  ge::TensorDesc output_desc;
  output_desc.SetShape(op.GetInputDesc("x").GetShape());
  output_desc.SetDataType(op.GetInputDesc("x").GetDataType());
  if (!SixToFourInferShapeAndType(op, output_desc)) {
    return GRAPH_FAILED;
  }
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DepthwiseWeight6DTo4D, DepthwiseWeight6DTo4DInferShape);
// Registered verify function
VERIFY_FUNC_REG(DepthwiseWeight6DTo4D, DepthwiseWeight6DTo4DVerify);

// ----------------DepthwiseWeight4DTo6D Op-------------------
// transfer shape and dtype
static bool FourToSixInferShapeAndType(const ge::Operator& op, ge::TensorDesc& output_desc) {
  auto inputDesc = op.GetInputDesc("x");
  int64_t dimNum = inputDesc.GetShape().GetDimNum();
  if (dimNum == 4) {
    int64_t columnNum = inputDesc.GetShape().GetDim(2);
    int64_t c0Value = 16;
    int64_t c1Value = (columnNum + c0Value - 1) / c0Value;
    // Infer the output dimension from the corresponding input dimension
    vector<int64_t> outShape;
    outShape.push_back(c1Value);
    outShape.push_back(inputDesc.GetShape().GetDim(0));
    outShape.push_back(inputDesc.GetShape().GetDim(1));
    outShape.push_back(inputDesc.GetShape().GetDim(3));
    outShape.push_back(c0Value);
    outShape.push_back(c0Value);
    ge::Shape outputShape = ge::Shape(outShape);
    output_desc.SetShape(outputShape);
    output_desc.SetDataType(op.GetInputDesc("x").GetDataType());
  } else {
    OP_LOGE(op.GetName().c_str(), "please make sure that the dim of inputshape is 4!");
    return false;
  }
  return true;
}

// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(DepthwiseWeight4DTo6D, DepthwiseWeight4DTo6DVerify) {
  OP_LOGI(op.GetName().c_str(), "enter op_proto verifier function!!!");
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DepthwiseWeight4DTo6DInferShape) {
  OP_LOGI(op.GetName().c_str(), "enter op_proto inferfunction!!!");
  ge::TensorDesc output_desc;
  if (!FourToSixInferShapeAndType(op, output_desc)) {
    return GRAPH_FAILED;
  }
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DepthwiseWeight4DTo6D, DepthwiseWeight4DTo6DInferShape);
// Registered verify function
VERIFY_FUNC_REG(DepthwiseWeight4DTo6D, DepthwiseWeight4DTo6DVerify);

// ----------------SpaceToBatchND Op Start-------------------
static void CalcSpaceToBatch(const Tensor& data, const DataType& dtype, std::vector<int64_t>& const_vec) {
  const uint8_t* constData = data.GetData();
  size_t size;
  if (dtype == ge::DT_INT32) {
    size = data.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_vec.push_back(*((int32_t*)constData + i));
    }
  } else {
    size = data.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_vec.push_back(*((int64_t*)constData + i));
    }
  }
}

IMPLEMT_COMMON_INFERFUNC(SpaceToBatchNDInferShape) {
  Tensor data1;
  if (op.GetInputConstData("block_shape", data1) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get constValue failed of [block_shape]");
    return GRAPH_FAILED;
  }
  DataType dtype1 = op.GetInputDesc("block_shape").GetDataType();
  std::vector<int64_t> block_shape;
  CalcSpaceToBatch(data1, dtype1, block_shape);
  Tensor data2;
  if (op.GetInputConstData("paddings", data2) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get constValue failed of [paddings]");
    return GRAPH_FAILED;
  }
  DataType dtype2 = op.GetInputDesc("paddings").GetDataType();
  std::vector<int64_t> paddings;
  CalcSpaceToBatch(data2, dtype2, paddings);

  auto tensordesc = op.GetInputDesc("x");
  auto shape = tensordesc.GetShape();
  auto dtype = tensordesc.GetDataType();
  std::vector<int64_t> shape_vector = shape.GetDims();
  if (shape_vector.size() <= block_shape.size()) {
    OP_LOGE(op.GetName().c_str(),
            "DimSize of x is not greater than size \
                                   of block_shape.");
    return GRAPH_FAILED;
  }

  TensorDesc td = op.GetOutputDesc("y");
  std::vector<int64_t> y_shape;
  int64_t block_shape_size = shape_vector[0];
  for (size_t i = 0; i < block_shape.size(); i++) {
    block_shape_size = block_shape_size * block_shape[i];
  }
  y_shape.push_back(block_shape_size);
  for (size_t i = 1; i <= block_shape.size(); i++) {
    y_shape.push_back((shape_vector[i] + paddings[2 * i - 2] + paddings[2 * i - 1]) / block_shape[i - 1]);
  }
  for (size_t i = block_shape.size() + 1; i < shape_vector.size(); i++) {
    y_shape.push_back(shape_vector[i]);
  }

  Shape outShape(y_shape);
  td.SetShape(outShape);
  td.SetDataType(dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SpaceToBatchND, SpaceToBatchNDVerify) {
  if (!CheckTwoInputDtypeSame(op, "block_shape", "paddings")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SpaceToBatchND, SpaceToBatchNDInferShape);
VERIFY_FUNC_REG(SpaceToBatchND, SpaceToBatchNDVerify);
// ----------------SpaceToBatchND Op End-------------------

// ----------------SpaceToBatchNDD Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(SpaceToBatchNDDInferShape) {
  auto tensordesc = op.GetInputDesc("x");
  auto dtype = tensordesc.GetDataType();
  auto shape = tensordesc.GetShape();
  int64_t shape_2 = shape.GetDim(2);
  int64_t shape_3 = shape.GetDim(3);
  std::vector<int64_t> shape_vector = shape.GetDims();
  std::vector<int64_t> block_shape;
  if (op.GetAttr("block_shape", block_shape) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "block_shape");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue block_shape failed!");
    return GRAPH_FAILED;
  }
  if (block_shape.size() != 2) {
    OpsAttrValueErrReport(op.GetName(), "block_shape", "2", ConcatString(block_shape.size()));
    OP_LOGE(op.GetName().c_str(), "the shape of block_shape should be 2 !");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> paddings;
  if (op.GetAttr("paddings", paddings) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "paddings");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue paddings failed!");
    return GRAPH_FAILED;
  }
  if ((paddings.size() != 4)) {
    OpsAttrValueErrReport(op.GetName(), "paddings", "4", ConcatString(paddings.size()));
    OP_LOGE(op.GetName().c_str(), "the shape of paddings should be 2x2 !");
    return GRAPH_FAILED;
  }
  if ((paddings[0] < 0) || (paddings[1] < 0) || (paddings[2] < 0) || (paddings[3] < 0)) {
    OP_LOGE(op.GetName().c_str(), "the value of paddings should be greater or equal to 0");
    return GRAPH_FAILED;
  }
  if ((shape_2 + paddings[0] + paddings[1]) % (block_shape[0]) != 0) {
    OP_LOGE(op.GetName().c_str(), "paddings height should be exactly divisible by block height");
    return GRAPH_FAILED;
  }
  if ((shape_3 + paddings[2] + paddings[3]) % (block_shape[1]) != 0) {
    OP_LOGE(op.GetName().c_str(), "paddings width should be exactly divisible by block width");
    return GRAPH_FAILED;
  }
  if (shape_vector.size() <= block_shape.size()) {
    OP_LOGE(op.GetName().c_str(),
            "DimSize of x is not greater than size \
                                   of block_shape.");
    return GRAPH_FAILED;
  }

  TensorDesc td = op.GetOutputDesc("y");
  std::vector<int64_t> y_shape;
  int64_t block_shape_size = shape_vector[0];
  for (size_t i = 0; i < block_shape.size(); i++) {
    block_shape_size = block_shape_size * block_shape[i];
  }
  y_shape.push_back(block_shape_size);
  for (size_t i = 1; i <= block_shape.size(); i++) {
    y_shape.push_back((shape_vector[i] + paddings[2 * i - 2] + paddings[2 * i - 1]) / block_shape[i - 1]);
  }
  for (size_t i = block_shape.size() + 1; i < shape_vector.size(); i++) {
    y_shape.push_back(shape_vector[i]);
  }

  Shape outShape(y_shape);
  td.SetShape(outShape);
  td.SetDataType(dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SpaceToBatchNDD, SpaceToBatchNDDInferShape);
// ----------------SpaceToBatchNDD Op End-------------------

// ----------------BatchToSpaceND Op Start-------------------
static void CalcBatchToSpace(const Tensor& data, const DataType& dtype, std::vector<int64_t>& const_vec) {
  const uint8_t* constData = data.GetData();
  size_t size;
  if (dtype == ge::DT_INT32) {
    size = data.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_vec.push_back(*((int32_t*)constData + i));
    }
  } else {
    size = data.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_vec.push_back(*((int64_t*)constData + i));
    }
  }
}

IMPLEMT_COMMON_INFERFUNC(BatchToSpaceNDInferShape) {
  Tensor data1;
  if (op.GetInputConstData("block_shape", data1) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get constValue failed of [block_shape]");
    return GRAPH_FAILED;
  }
  DataType dtype1 = op.GetInputDesc("block_shape").GetDataType();
  std::vector<int64_t> block_shape;
  CalcBatchToSpace(data1, dtype1, block_shape);
  Tensor data2;
  if (op.GetInputConstData("crops", data2) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get constValue failed of [crops]");
    return GRAPH_FAILED;
  }
  DataType dtype2 = op.GetInputDesc("crops").GetDataType();
  std::vector<int64_t> crops;
  CalcBatchToSpace(data2, dtype2, crops);

  auto tensordesc = op.GetInputDesc("x");
  auto shape = tensordesc.GetShape();
  auto dtype = tensordesc.GetDataType();
  std::vector<int64_t> shape_vector = shape.GetDims();

  if (shape_vector.size() <= block_shape.size()) {
    OP_LOGE(op.GetName().c_str(),
            "DimSize of x is not greater than size \
                                   of block_shape.");
    return GRAPH_FAILED;
  }
  TensorDesc td = op.GetOutputDesc("y");
  std::vector<int64_t> y_shape;
  int64_t block_shape_size = shape_vector[0];
  for (size_t i = 0; i < block_shape.size(); i++) {
    block_shape_size = block_shape_size / block_shape[i];
  }
  y_shape.push_back(block_shape_size);
  for (size_t i = 1; i <= block_shape.size(); i++) {
    y_shape.push_back(shape_vector[i] * block_shape[i - 1] - crops[2 * i - 2] - crops[2 * i - 1]);
  }
  for (size_t i = block_shape.size() + 1; i < shape_vector.size(); i++) {
    y_shape.push_back(shape_vector[i]);
  }

  Shape outShape(y_shape);
  td.SetShape(outShape);
  td.SetDataType(dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(BatchToSpaceND, BatchToSpaceNDVerify) {
  if (!CheckTwoInputDtypeSame(op, "block_shape", "crops")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BatchToSpaceND, BatchToSpaceNDInferShape);
VERIFY_FUNC_REG(BatchToSpaceND, BatchToSpaceNDVerify);
// ----------------BatchToSpaceND Op End-------------------

// ----------------BatchToSpaceNDD Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(BatchToSpaceNDDInferShape) {
  auto tensordesc = op.GetInputDesc("x");
  auto dtype = tensordesc.GetDataType();
  auto shape = tensordesc.GetShape();
  int64_t shape_0 = shape.GetDim(0);
  int64_t shape_2 = shape.GetDim(2);
  int64_t shape_3 = shape.GetDim(3);
  std::vector<int64_t> shape_vector = shape.GetDims();
  std::vector<int64_t> block_shape;
  if (op.GetAttr("block_shape", block_shape) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "block_shape");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue block_shape failed!");
    return GRAPH_FAILED;
  }
  if (block_shape.size() != DIM_SIZE2) {
    OpsAttrValueErrReport(op.GetName(), "block_shape", ConcatString(DIM_SIZE2), ConcatString(block_shape.size()));
    OP_LOGE(op.GetName().c_str(), "the shape of block_shape should be 2 !");
    return GRAPH_FAILED;
  }
  if ((block_shape[0] < 0) || (block_shape[1] < 0)) {
    OP_LOGE(op.GetName().c_str(), "the value of block_shape should be greater or equal to 0");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> crops;
  if (op.GetAttr("crops", crops) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "crops");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue crops failed!");
    return GRAPH_FAILED;
  }

  if ((crops.size() != DIM_SIZE4)) {
    OpsAttrValueErrReport(op.GetName(), "crops", ConcatString(DIM_SIZE4), ConcatString(crops.size()));
    OP_LOGE(op.GetName().c_str(), "the shape of crops should be 2x2 !");
    return GRAPH_FAILED;
  }

  if ((crops[0] < 0) || (crops[1] < 0) || (crops[2] < 0) || (crops[3] < 0)) {
    OP_LOGE(op.GetName().c_str(), "the value of crops should be greater or equal to 0");
    return GRAPH_FAILED;
  }

  if ((crops[0] + crops[1]) >= shape_2 * block_shape[0]) {
    OP_LOGE(op.GetName().c_str(),
            "crops in height dimension should less than "
            "(input height)*(block height)");
    return GRAPH_FAILED;
  }

  if ((crops[2] + crops[3]) >= shape_3 * block_shape[1]) {
    OP_LOGE(op.GetName().c_str(),
            "crops in width dimension should less than "
            "(input width)*(block width)");
    return GRAPH_FAILED;
  }

  if (shape_0 % (block_shape[0] * block_shape[1]) != 0) {
    OP_LOGE(op.GetName().c_str(), "batch size/(block height*block width) should be integer");
    return GRAPH_FAILED;
  }

  if (shape_vector.size() <= block_shape.size()) {
    OP_LOGE(op.GetName().c_str(),
            "DimSize of x is not greater than size \
                                   of block_shape.");
    return GRAPH_FAILED;
  }

  TensorDesc td = op.GetOutputDesc("y");
  std::vector<int64_t> y_shape;
  int64_t block_shape_size = shape_vector[0];
  for (size_t i = 0; i < block_shape.size(); i++) {
    block_shape_size = block_shape_size / block_shape[i];
  }
  y_shape.push_back(block_shape_size);
  for (size_t i = 1; i <= block_shape.size(); i++) {
    y_shape.push_back(shape_vector[i] * block_shape[i - 1] - crops[2 * i - 2] - crops[2 * i - 1]);
  }
  for (size_t i = block_shape.size() + 1; i < shape_vector.size(); i++) {
    y_shape.push_back(shape_vector[i]);
  }

  Shape outShape(y_shape);
  td.SetShape(outShape);
  td.SetDataType(dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BatchToSpaceNDD, BatchToSpaceNDDInferShape);
// ----------------BatchToSpaceNDD Op End-------------------

IMPLEMT_VERIFIER(Flatten, FlattenVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(FlattenInferShape) {
  Shape x_shape = op.GetInputDesc("x").GetShape();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  std::vector<int64_t> xVector = x_shape.GetDims();
  std::vector<int64_t> yVector;
  int64_t num = 1;
  for (size_t i = 0; i < xVector.size() - 1; ++i) {
    num = num * xVector[i + 1];
  }
  yVector.push_back(xVector[0]);
  yVector.push_back(num);
  Shape outShape(yVector);
  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(outShape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(Flatten, FlattenVerify);
COMMON_INFER_FUNC_REG(Flatten, FlattenInferShape);

int64_t GetMax(const std::vector<int64_t>& vec) {
  int64_t resValue = vec[0];

  for (auto v : vec) {
    if (resValue < v) {
      resValue = v;
    }
  }

  return resValue;
}

int64_t GetMin(const std::vector<int64_t>& vec) {
  int64_t resValue = vec[0];

  for (auto v : vec) {
    if (resValue > v) {
      resValue = v;
    }
  }

  return resValue;
}

// ----------------Transpose Op Begin-------------------
static graphStatus TransposeCommonInferShape(const std::vector<int64_t>& perm_list, Operator& op) {
  vector<int64_t> out_vec;
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  std::vector<std::pair<int64_t, int64_t>> out_range;

  Shape shape = op.GetInputDesc("x").GetShape();
  op.GetInputDesc("x").GetShapeRange(shape_range);

  size_t dim_num = shape.GetDimNum();
  // do perm operation when shape is not -2
  if (shape.GetDims() != UNKNOWN_RANK) {
    size_t perm_list_size = perm_list.size();
    if (perm_list.empty() || (perm_list_size != dim_num)) {
      OpsAttrValueErrReport(op.GetName(), "perm", ConcatString(dim_num), ConcatString(perm_list_size));
      OP_LOGE(op.GetName().c_str(), "perm is empty or perm size is not match shape size");
      return GRAPH_FAILED;
    }
    for (size_t i = 0; i < dim_num; ++i) {
      if ((size_t)perm_list[i] >= dim_num || (size_t)perm_list[i] < 0) {
        OP_LOGE(op.GetName().c_str(), "value of perm is wrong");
        return GRAPH_FAILED;
      }
    }

    // for shape is -1 case
    if (shape_range.size() == dim_num) {
      for (size_t i = 0; i < dim_num; ++i) {
        out_range.push_back(shape_range[perm_list[i]]);
      }
    }
    // for shape is static case
    else {
      out_range = shape_range;
    }

    for (size_t i = 0; i < dim_num; ++i) {
      out_vec.push_back(shape.GetDim(perm_list[i]));
    }

  }
  // for shape is -2 case
  else {
    for (size_t i = 0; i < perm_list.size(); ++i) {
      out_vec.push_back(-1);
      out_range.push_back(std::make_pair(1, -1));
    }
  }

  Shape out_shape(out_vec);
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(out_shape);
  tensordesc_output.SetOriginShape(out_shape);
  tensordesc_output.SetDataType(op.GetInputDesc("x").GetDataType());
  tensordesc_output.SetShapeRange(out_range);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(TransposeInferShape)
  Tensor perm_tensor;
  // in order to switch to aicpu when aicore not support
  op_desc->SetOpInferDepends({"perm"});

  if (GRAPH_SUCCESS != op.GetInputConstData("perm", perm_tensor)) {
    OP_LOGI("GetInputConstData perm failed. Set unkonwn shape.");
    Shape shape = op.GetInputDesc("x").GetShape();
    std::vector<std::pair<int64_t, int64_t>> shape_range;
    std::vector<std::pair<int64_t, int64_t>> out_range;
    int64_t max_range_value = 0;
    int64_t min_range_value = 0;
    int64_t max_shape_value = 0;
    int64_t min_shape_value = 0;
    size_t dim_num = shape.GetDimNum();

    vector<int64_t> out_vec;
    if (shape.GetDims() != UNKNOWN_RANK) {
      for (size_t i = 0; i < dim_num; ++i) {
        out_vec.push_back(-1);
      }
    } else {
      out_vec.push_back(-2);
    }

    // for shape is -1 case
    std::vector<int64_t> range_list;
    op.GetInputDesc("x").GetShapeRange(shape_range);
    if (shape_range.size() == dim_num && shape.GetDims() != UNKNOWN_RANK) {
      for (size_t i = 0; i < dim_num; ++i) {
        range_list.push_back(shape_range[i].first);
        range_list.push_back(shape_range[i].second);
      }
      max_range_value = GetMax(range_list);
      min_range_value = GetMin(range_list);
      for (size_t i = 0; i < dim_num; ++i) {
        out_range.push_back(std::make_pair(min_range_value, max_range_value));
      }
    }
    // for shape is static
    else if (shape.GetDims() != UNKNOWN_RANK && shape.GetDims() != UNKNOWN_SHAPE) {
      max_shape_value = GetMax(shape.GetDims());
      min_shape_value = GetMin(shape.GetDims());
      for (size_t i = 0; i < dim_num; ++i) {
        out_range.push_back(std::make_pair(min_shape_value, max_shape_value));
      }
    }
    // for shape is -2 case, no shape range
    else {
      out_range = shape_range;
    }

    Shape out_shape(out_vec);
    TensorDesc tensordesc_output = op.GetOutputDesc("y");
    tensordesc_output.SetShape(out_shape);
    tensordesc_output.SetOriginShape(out_shape);
    tensordesc_output.SetDataType(op.GetInputDesc("x").GetDataType());
    tensordesc_output.SetShapeRange(out_range);
    (void)op.UpdateOutputDesc("y", tensordesc_output);
    return GRAPH_SUCCESS;
  }
  DataType dtype = op.GetInputDesc("perm").GetDataType();

  std::vector<int64_t> perm_list;
  if (dtype == ge::DT_INT32) {
    int32_t* const_data_ptr = (int32_t*)perm_tensor.GetData();
    size_t const_num = perm_tensor.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < const_num; ++i) {
      perm_list.push_back((int32_t)((*(const_data_ptr + i))));
    }
  } else if (dtype == ge::DT_INT64) {
    int64_t* const_data_ptr = (int64_t*)perm_tensor.GetData();
    size_t const_num = perm_tensor.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < const_num; ++i) {
      perm_list.push_back((int64_t)((*(const_data_ptr + i))));
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "transpose perm do not support this type %d", dtype);
    return GRAPH_FAILED;
  }

  if (GRAPH_SUCCESS != TransposeCommonInferShape(perm_list, op)) {
    return GRAPH_FAILED;
  }
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(Transpose, TransposeInferShape);

IMPLEMT_INFERFORMAT_FUNC(Transpose, TransposeInferFormat) {
  bool recovery_flag = false;  // if not scene that transformation between NCHW or NHWC, keep ND
  Tensor perm_tensor;
  if (GRAPH_SUCCESS != op.GetInputConstData("perm", perm_tensor)) {
    OP_LOGI("GetInputConstData perm failed. Set unkonwn format.");
    auto input_format = op.GetInputDesc("x").GetOriginFormat();
    auto output_format = op.GetOutputDesc("y").GetOriginFormat();

    TensorDesc tensordesc_output = op.GetOutputDesc("y");
    tensordesc_output.SetOriginFormat(output_format);
    tensordesc_output.SetFormat(output_format);
    (void)op.UpdateOutputDesc("y", tensordesc_output);

    TensorDesc tensordesc_input = op.GetInputDesc("x");
    tensordesc_input.SetOriginFormat(input_format);
    tensordesc_input.SetFormat(input_format);
    (void)op.UpdateInputDesc("x", tensordesc_input);

    return GRAPH_SUCCESS;
  }
  DataType dtype = perm_tensor.GetTensorDesc().GetDataType();

  std::vector<int64_t> perm_list;
  if (dtype == ge::DT_INT32) {
    int32_t* const_data_ptr = (int32_t*)perm_tensor.GetData();
    size_t const_num = perm_tensor.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < const_num; ++i) {
      perm_list.push_back((int32_t)((*(const_data_ptr + i))));
    }
  } else if (dtype == ge::DT_INT64) {
    int64_t* const_data_ptr = (int64_t*)perm_tensor.GetData();
    size_t const_num = perm_tensor.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < const_num; ++i) {
      perm_list.push_back((int64_t)((*(const_data_ptr + i))));
    }
  } else {
    recovery_flag = true;
    OP_LOGE(op.GetName().c_str(), "transpose perm do not support this type %d", dtype);
  }

  auto input_format = (recovery_flag == false) ? op.GetInputDesc("x").GetOriginFormat() : FORMAT_ND;
  auto output_format = (recovery_flag == false) ? op.GetOutputDesc("y").GetOriginFormat() : FORMAT_ND;

  if (input_format == FORMAT_ND && output_format == FORMAT_ND) {
    OP_LOGI(op.GetName().c_str(),
            "[Transpose Inferformat] only support trans between NCHW and NHWC.input format is %d, output format is %d",
            input_format, output_format);
    // Recovery ND origin format
    TensorDesc tensordesc_output = op.GetOutputDesc("y");
    tensordesc_output.SetOriginFormat(output_format);
    tensordesc_output.SetFormat(output_format);
    (void)op.UpdateOutputDesc("y", tensordesc_output);

    TensorDesc tensordesc_input = op.GetInputDesc("x");
    tensordesc_input.SetOriginFormat(input_format);
    tensordesc_input.SetFormat(input_format);
    (void)op.UpdateInputDesc("x", tensordesc_input);
    return GRAPH_SUCCESS;
  }
  vector<int64_t> NCHW_to_NHWC_order = {0, 2, 3, 1};
  vector<int64_t> NHWC_to_NCHW_order = {0, 3, 1, 2};

  if (input_format == FORMAT_ND) {
    switch (output_format) {
      case FORMAT_NCHW:
        input_format = (perm_list == NHWC_to_NCHW_order) ? FORMAT_NHWC : FORMAT_ND;
        output_format = (perm_list == NHWC_to_NCHW_order) ? output_format : FORMAT_ND;
        break;
      case FORMAT_NHWC:
        input_format = (perm_list == NCHW_to_NHWC_order) ? FORMAT_NCHW : FORMAT_ND;
        output_format = (perm_list == NCHW_to_NHWC_order) ? output_format : FORMAT_ND;
        break;
      default:
        OP_LOGI(
            op.GetName().c_str(),
            "[Transpose Inferformat] only support trans between NCHW and NHWC.input format is %d, output format is %d",
            input_format, output_format);
        output_format = FORMAT_ND;
        break;
    }
  } else {
    switch (input_format) {
      case FORMAT_NCHW:
        output_format = (perm_list == NCHW_to_NHWC_order) ? FORMAT_NHWC : FORMAT_ND;
        input_format = (perm_list == NCHW_to_NHWC_order) ? input_format : FORMAT_ND;
        break;
      case FORMAT_NHWC:
        output_format = (perm_list == NHWC_to_NCHW_order) ? FORMAT_NCHW : FORMAT_ND;
        input_format = (perm_list == NHWC_to_NCHW_order) ? input_format : FORMAT_ND;
        break;
      default:
        OP_LOGI(
            op.GetName().c_str(),
            "[Transpose Inferformat] only support trans between NCHW and NHWC.input format is %d, output format is %d",
            input_format, output_format);
        input_format = FORMAT_ND;
        break;
    }
  }

  OP_LOGD(op.GetName().c_str(), "[Transpose Inferformat] Finaly input format is %d, output format is %d", input_format,
          output_format);

  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetOriginFormat(output_format);
  tensordesc_output.SetFormat(output_format);
  (void)op.UpdateOutputDesc("y", tensordesc_output);

  TensorDesc tensordesc_input = op.GetInputDesc("x");
  tensordesc_input.SetOriginFormat(input_format);
  tensordesc_input.SetFormat(input_format);
  (void)op.UpdateInputDesc("x", tensordesc_input);

  return GRAPH_SUCCESS;
}

INFER_FORMAT_FUNC_REG(Transpose, TransposeInferFormat);
// ----------------Transpose Op End-------------------

// ----------------TransposeD Op Begin-------------------
// to adapt dynamic shape case
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(TransposeDInferShape)
  std::vector<int64_t> perm_list;
  if (ge::GRAPH_SUCCESS != op.GetAttr("perm", perm_list)) {
    OpsGetAttrErrReport(op.GetName(), "perm");
    OP_LOGE(op.GetName().c_str(), "The transpose_d op GetOpAttr ConstValue failed!");
  }
  if (GRAPH_SUCCESS != TransposeCommonInferShape(perm_list, op)) {
    return GRAPH_FAILED;
  }
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(TransposeD, TransposeDInferShape);

IMPLEMT_INFERFORMAT_FUNC(TransposeD, TransposeDInferFormat) {
  bool recovery_flag = false;  // if not scene that transformation between NCHW or NHWC, keep ND
  std::vector<int64_t> perm_list;
  if (ge::GRAPH_SUCCESS != op.GetAttr("perm", perm_list)) {
    OpsGetAttrErrReport(op.GetName(), "perm");
    OP_LOGE(op.GetName().c_str(), "The transpose_d op GetOpAttr ConstValue failed!");
    recovery_flag = true;  // if not scene that transformation between NCHW or NHWC, keep ND
  }

  auto input_format = (recovery_flag == false) ? op.GetInputDesc("x").GetOriginFormat() : FORMAT_ND;
  auto output_format = (recovery_flag == false) ? op.GetOutputDesc("y").GetOriginFormat() : FORMAT_ND;

  if (input_format == FORMAT_ND && output_format == FORMAT_ND) {
    OP_LOGI(op.GetName().c_str(),
            "[TransposeD Inferformat] only support trans between NCHW and NHWC.input format is %d, output format is %d",
            input_format, output_format);
    // Recovery ND origin format
    TensorDesc tensordesc_output = op.GetOutputDesc("y");
    tensordesc_output.SetOriginFormat(output_format);
    tensordesc_output.SetFormat(output_format);
    (void)op.UpdateOutputDesc("y", tensordesc_output);

    TensorDesc tensordesc_input = op.GetInputDesc("x");
    tensordesc_input.SetOriginFormat(input_format);
    tensordesc_input.SetFormat(input_format);
    (void)op.UpdateInputDesc("x", tensordesc_input);
    return GRAPH_SUCCESS;
  }
  vector<int64_t> NCHW_to_NHWC_order = {0, 2, 3, 1};
  vector<int64_t> NHWC_to_NCHW_order = {0, 3, 1, 2};

  if (input_format == FORMAT_ND) {
    switch (output_format) {
      case FORMAT_NCHW:
        input_format = (perm_list == NHWC_to_NCHW_order) ? FORMAT_NHWC : FORMAT_ND;
        output_format = (perm_list == NHWC_to_NCHW_order) ? output_format : FORMAT_ND;
        break;
      case FORMAT_NHWC:
        input_format = (perm_list == NCHW_to_NHWC_order) ? FORMAT_NCHW : FORMAT_ND;
        output_format = (perm_list == NCHW_to_NHWC_order) ? output_format : FORMAT_ND;
        break;
      default:
        OP_LOGI(
            op.GetName().c_str(),
            "[TransposeD Inferformat] only support trans between NCHW and NHWC.input format is %d, output format is %d",
            input_format, output_format);
        output_format = FORMAT_ND;
        break;
    }
  } else {
    switch (input_format) {
      case FORMAT_NCHW:
        output_format = (perm_list == NCHW_to_NHWC_order) ? FORMAT_NHWC : FORMAT_ND;
        input_format = (perm_list == NCHW_to_NHWC_order) ? input_format : FORMAT_ND;
        break;
      case FORMAT_NHWC:
        output_format = (perm_list == NHWC_to_NCHW_order) ? FORMAT_NCHW : FORMAT_ND;
        input_format = (perm_list == NHWC_to_NCHW_order) ? input_format : FORMAT_ND;
        break;
      default:
        OP_LOGI(
            op.GetName().c_str(),
            "[TransposeD Inferformat] only support trans between NCHW and NHWC.input format is %d, output format is %d",
            input_format, output_format);
        input_format = FORMAT_ND;
        break;
    }
  }

  OP_LOGD(op.GetName().c_str(), "[TransposeD Inferformat] Finaly input format is %d, output format is %d", input_format,
          output_format);

  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetOriginFormat(output_format);
  tensordesc_output.SetFormat(output_format);
  (void)op.UpdateOutputDesc("y", tensordesc_output);

  TensorDesc tensordesc_input = op.GetInputDesc("x");
  tensordesc_input.SetOriginFormat(input_format);
  tensordesc_input.SetFormat(input_format);
  (void)op.UpdateInputDesc("x", tensordesc_input);

  return GRAPH_SUCCESS;
}

INFER_FORMAT_FUNC_REG(TransposeD, TransposeDInferFormat);
// ----------------TransposeD Op End-------------------

// ----------------TranData Op Begin---------------------
IMPLEMT_COMMON_INFERFUNC_HELPER_BEGIN(TransDataInferShape)
  // main part of shape infer
  TensorDesc src_tensor = op.GetInputDesc("src");
  Shape src_shape = src_tensor.GetShape();
  DataType input_dtype = src_tensor.GetDataType();
  TensorDesc td = op.GetOutputDesc("dst");
  if (src_tensor.GetOriginFormat() == td.GetOriginFormat()) {
    td.SetShape(ge::Shape(src_shape));
    td.SetDataType(input_dtype);
    (void)op.UpdateOutputDesc("dst", td);
  }
IMPLEMT_COMMON_INFERFUNC_HELPER_END()

COMMON_INFER_FUNC_REG(TransData, TransDataInferShape);
// ----------------TranData Op End------------------------

// ----------------Permute Op Begin-------------------
IMPLEMT_COMMON_INFERFUNC(PermuteInferShape) {
  auto input_shape = op.GetInputDesc("x").GetShape();
  std::vector<int64_t> input_shape_dims = input_shape.GetDims();

  std::vector<int64_t> perm_list;
  if (ge::GRAPH_SUCCESS != op.GetAttr("order", perm_list)) {
    OP_LOGE(op.GetName().c_str(), "The Permute op GetOpAttr ConstValue failed!");
    OpsGetAttrErrReport(op.GetName(), "order");
    return GRAPH_FAILED;
  }
  for (size_t i = 0; i < input_shape_dims.size(); ++i) {
    if (std::find(perm_list.begin(), perm_list.end(), i) == perm_list.end()) {
      perm_list.push_back((int64_t)i);
    }
  }
  op.SetAttr("perm", perm_list);
  op.SetAttr("order", perm_list);
  return TransposeCommonInferShape(perm_list, op);
}

COMMON_INFER_FUNC_REG(Permute, PermuteInferShape);
// ----------------Permute Op End-----------------

// ------------------DepthToSpace------------------
IMPLEMT_VERIFIER(DepthToSpace, DepthToSpaceVerify) {
  auto x_shape = op.GetInputDesc("x").GetShape().GetDims();
  int64_t block_size;
  if (GRAPH_SUCCESS != op.GetAttr("block_size", block_size)) {
    OpsGetAttrErrReport(op.GetName(), "block_size");
    OP_LOGE("ERROR] GetOpAttr block_size failed!");
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_SUCCESS) {
    if (data_format != "NHWC" && data_format != "NCHW" && data_format != "NC1HWC0") {
      string expected_format_list = ConcatString("NHWC, NCHW, NC1HWC0");
      OpsInputFormatErrReport(op.GetName(), "data_format", expected_format_list, data_format);
      OP_LOGE(op.GetName().c_str(), "data_format only support 'NHWC', 'NCHW', 'NC1HWC0'.");
      return GRAPH_FAILED;
    }
  }
  auto y_depth = x_shape[3] / block_size / block_size;
  if ((y_depth != (int)y_depth) || (block_size < 2)) {
    OP_LOGE("[ERROR]the depth_to_space op do not supported the block_size!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(DepthToSpaceInfer) {
  auto x_shape = op.GetInputDesc(0).GetShape().GetDims();
  DataType x_dtype = op.GetInputDesc(0).GetDataType();
  TensorDesc y = op.GetOutputDesc(0);
  Format format = op.GetInputDesc(0).GetFormat();
  int64_t block_size;
  if (GRAPH_SUCCESS != op.GetAttr("block_size", block_size)) {
    OpsGetAttrErrReport(op.GetName(), "block_size");
    OP_LOGE("ERROR] GetOpAttr block_size failed!");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> y_shape;
  if (format == FORMAT_NHWC) {
    y_shape.push_back(x_shape[0]);
    y_shape.push_back(x_shape[1] * block_size);
    y_shape.push_back(x_shape[2] * block_size);
    y_shape.push_back(x_shape[3] / block_size / block_size);
  } else if (format == FORMAT_NCHW) {
    y_shape.push_back(x_shape[0]);
    y_shape.push_back(x_shape[1] / block_size / block_size);
    y_shape.push_back(x_shape[2] * block_size);
    y_shape.push_back(x_shape[3] * block_size);
  }
  y.SetShape(Shape(y_shape));
  y.SetDataType(x_dtype);
  (void)op.UpdateOutputDesc("y", y);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(DepthToSpace, DepthToSpaceInfer);
VERIFY_FUNC_REG(DepthToSpace, DepthToSpaceVerify);
// -------------------DepthToSpace END-----------------

// ----------------SpaceToDepth Op Start-------------------
IMPLEMT_VERIFIER(SpaceToDepth, SpaceToDepthVerify) {
  int64_t block_size;
  if (GRAPH_SUCCESS != op.GetAttr("block_size", block_size)) {
    OpsGetAttrErrReport(op.GetName(), "block_size");
    OP_LOGE("[ERROR] GetOpAttr block_size failed!");
    return GRAPH_FAILED;
  }
  if (block_size <= 1) {
    OpsAttrValueErrReport(op.GetName(), "block_size", ">1", ConcatString(block_size));
    OP_LOGE("[ERROR]the space_to_depth op block_size need >1!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SpaceToDepthInferShape) {
  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_SUCCESS) {
    if (data_format != "NHWC" && data_format != "NCHW" && data_format != "NC1HWC0") {
      string expected_format_list = ConcatString("NHWC, NCHW, NC1HWC0");
      OpsInputFormatErrReport(op.GetName(), "data_format", expected_format_list, data_format);
      OP_LOGE(op.GetName().c_str(), "data_format only support 'NHWC', 'NCHW', 'NC1HWC0'.");
      return GRAPH_FAILED;
    }
  }
  std::vector<int64_t> x_shape = op.GetInputDesc("x").GetShape().GetDims();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  Format format = op.GetInputDesc("x").GetFormat();
  int64_t block_size;
  if (GRAPH_SUCCESS != op.GetAttr("block_size", block_size)) {
    OpsGetAttrErrReport(op.GetName(), "block_size");
    OP_LOGE("ERROR] GetOpAttr block_size failed!");
    return GRAPH_FAILED;
  }
  if (block_size < 2) {
    OpsAttrValueErrReport(op.GetName(), "block_size", "greater than or equals to 2", ConcatString(block_size));
    OP_LOGE(op.GetName().c_str(), "block_size need greater than or equals to 2");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> y_shape;
  if (x_shape.size() < 4) {
    OpsAttrValueErrReport(op.GetName(), "x'shape size", "greater than or equals to 4", ConcatString(x_shape.size()));
    OP_LOGE(op.GetName().c_str(), "Input shape size must >= 4, but got %d", x_shape.size());
    return GRAPH_FAILED;
  }
  if (format == FORMAT_NCHW) {
    y_shape.push_back(x_shape[0]);
    y_shape.push_back(x_shape[1] * block_size * block_size);
    y_shape.push_back(x_shape[2] / block_size);
    y_shape.push_back(x_shape[3] / block_size);
  } else {  // without NCHW all other formats set as NHWC
    y_shape.push_back(x_shape[0]);
    y_shape.push_back(x_shape[1] / block_size);
    y_shape.push_back(x_shape[2] / block_size);
    y_shape.push_back(x_shape[3] * block_size * block_size);
  }
  TensorDesc output = op.GetOutputDesc("y");
  Shape output_shape(y_shape);
  output.SetShape(output_shape);
  output.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SpaceToDepth, SpaceToDepthInferShape);
VERIFY_FUNC_REG(SpaceToDepth, SpaceToDepthVerify);
// ----------------SpaceToDepth Op End-------------------

// ----------------SpaceToBatch Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(SpaceToBatchInferShape) {
  int64_t block_size;
  bool unknown_dim_flag = false;
  if (op.GetAttr("block_size", block_size) == ge::GRAPH_FAILED) {
    OP_LOGI(op.GetName().c_str(), "GetOpAttr ConstValue block_size failed. Set unkonwn shape.");
    unknown_dim_flag = true;
  }
  if (block_size < 2) {
    OP_LOGE(op.GetName().c_str(), "block_size need greater than or equals to 2");
    return GRAPH_FAILED;
  }

  Tensor data2;
  if (op.GetInputConstData("paddings", data2) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "Get constValue failed of [paddings]. Set unkonwn shape.");
    unknown_dim_flag = true;
  }

  auto tensor_desc = op.GetInputDesc("x");
  auto shape = tensor_desc.GetShape();
  auto dtype = tensor_desc.GetDataType();
  std::vector<int64_t> shape_vector = shape.GetDims();
  auto format = tensor_desc.GetFormat();
  TensorDesc output = op.GetOutputDesc("y");

  if (unknown_dim_flag == true) {
    if (format == FORMAT_NHWC) {
      Shape unknown_4d({UNKNOWN_DIM, UNKNOWN_DIM, UNKNOWN_DIM, shape_vector[3]});
      TensorDesc output(unknown_4d, format, dtype);
      (void)op.UpdateOutputDesc("y", output);
      return GRAPH_SUCCESS;
    }
    if (format == FORMAT_NCHW) {
      Shape unknown_4d({UNKNOWN_DIM, shape_vector[1], UNKNOWN_DIM, UNKNOWN_DIM});
      TensorDesc output(unknown_4d, format, dtype);
      (void)op.UpdateOutputDesc("y", output);
      return GRAPH_SUCCESS;
    }
    OP_LOGE(op.GetName().c_str(), "input format not support in (NHWC, NCHW).");
    return GRAPH_FAILED;
  }

  DataType dtype2 = op.GetInputDesc("paddings").GetDataType();
  std::vector<int64_t> const_vec2;
  CalcSpaceToBatch(data2, dtype2, const_vec2);

  std::vector<std::vector<int64_t>> paddings{{0, 0}, {0, 0}};
  paddings[0][0] = const_vec2[0];
  paddings[0][1] = const_vec2[1];
  paddings[1][0] = const_vec2[2];
  paddings[1][1] = const_vec2[3];
  std::vector<int64_t> y_shape;
  if (shape_vector.size() < 4) {
    OP_LOGE(op.GetName().c_str(), "Input shape size must >= 4, but got %d", shape_vector.size());
    return GRAPH_FAILED;
  }
  if (format == FORMAT_NCHW) {
    y_shape.push_back(shape_vector[0] * block_size * block_size);
    y_shape.push_back(shape_vector[1]);
    y_shape.push_back((shape_vector[2] + paddings[0][0] + paddings[0][1]) / block_size);
    y_shape.push_back((shape_vector[3] + paddings[1][0] + paddings[1][1]) / block_size);
  } else {  // without NCHW all other formats set as NHWC
    y_shape.push_back(shape_vector[0] * block_size * block_size);
    y_shape.push_back((shape_vector[1] + paddings[0][0] + paddings[0][1]) / block_size);
    y_shape.push_back((shape_vector[2] + paddings[1][0] + paddings[1][1]) / block_size);
    y_shape.push_back(shape_vector[3]);
  }
  Shape out_shape(y_shape);
  output.SetShape(out_shape);
  output.SetDataType(dtype);
  (void)op.UpdateOutputDesc("y", output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SpaceToBatch, SpaceToBatchInferShape);
// ----------------SpaceToBatch Op End-------------------

// ----------------SpaceToBatchD Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(SpaceToBatchDInferShape) {
  auto tensor_desc = op.GetInputDesc("x");
  auto dtype = tensor_desc.GetDataType();
  auto shape = tensor_desc.GetShape();
  std::vector<int64_t> shape_vector = shape.GetDims();
  auto format = tensor_desc.GetFormat();

  int64_t block_size;
  if (op.GetAttr("block_size", block_size) == ge::GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "block_size");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue block_size failed!");
    return GRAPH_FAILED;
  }
  if (block_size < 2) {
    OpsAttrValueErrReport(op.GetName(), "block_size", "greater than or equals to 2", ConcatString(block_size));
    OP_LOGE(op.GetName().c_str(), "block_size need greater than or equals to 2");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> paddings;
  if (op.GetAttr("paddings", paddings) == ge::GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "paddings");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue paddings failed!");
    return GRAPH_FAILED;
  }
  if ((paddings.size() != 4)) {
    OpsAttrValueErrReport(op.GetName(), "paddings'shape", "4", ConcatString(paddings.size()));
    OP_LOGE(op.GetName().c_str(), "the shape of paddings should be 2x2 !");
    return GRAPH_FAILED;
  }
  if ((paddings[0] < 0) || (paddings[1] < 0) || (paddings[2] < 0) || (paddings[3] < 0)) {
    OP_LOGE(op.GetName().c_str(), "the value of crops should be greater or equal to 0");
    return GRAPH_FAILED;
  }
  int64_t shape_2 = shape.GetDim(2);
  int64_t shape_3 = shape.GetDim(3);
  if ((shape_2 + paddings[0] + paddings[1]) % block_size != 0) {
    OP_LOGE(op.GetName().c_str(), "paddings height should be exactly divisible by block height");
    return GRAPH_FAILED;
  }
  if ((shape_3 + paddings[2] + paddings[3]) % block_size != 0) {
    OP_LOGE(op.GetName().c_str(), "paddings width should be exactly divisible by block width");
    return GRAPH_FAILED;
  }
  TensorDesc output = op.GetOutputDesc("y");
  std::vector<int64_t> y_shape;
  if (shape_vector.size() < 4) {
    OpsAttrValueErrReport(op.GetName(), "x'shape", "greater than or equals to 4", ConcatString(shape_vector.size()));
    OP_LOGE(op.GetName().c_str(), "Input shape size must >= 4, but got %d", shape_vector.size());
    return GRAPH_FAILED;
  }
  if (format == FORMAT_NCHW) {
    y_shape.push_back(shape_vector[0] * block_size * block_size);
    y_shape.push_back(shape_vector[1]);
    y_shape.push_back((shape_vector[2] + paddings[0] + paddings[1]) / block_size);
    y_shape.push_back((shape_vector[3] + paddings[2] + paddings[3]) / block_size);
  } else {  // without NCHW all other formats set as NHWC
    y_shape.push_back(shape_vector[0] * block_size * block_size);
    y_shape.push_back((shape_vector[1] + paddings[0] + paddings[1]) / block_size);
    y_shape.push_back((shape_vector[2] + paddings[2] + paddings[3]) / block_size);
    y_shape.push_back(shape_vector[3]);
  }
  Shape out_shape(y_shape);
  output.SetShape(out_shape);
  output.SetDataType(dtype);
  (void)op.UpdateOutputDesc("y", output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SpaceToBatchD, SpaceToBatchDInferShape);
// ----------------SpaceToBatchD Op End-------------------

// ----------------BatchToSpace Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(BatchToSpaceInferShape) {
  int64_t block_size;
  if (op.GetAttr("block_size", block_size) == ge::GRAPH_FAILED) {
    OP_LOGI(op.GetName().c_str(), "GetOpAttr ConstValue block_size failed. Set unkonwn shape.");
    Shape unknown_4d({UNKNOWN_DIM, UNKNOWN_DIM, UNKNOWN_DIM, UNKNOWN_DIM});
    CHECK_FORMAT(op.GetInputDesc("x").GetFormat());
    TensorDesc output(unknown_4d, op.GetInputDesc("x").GetFormat(), op.GetInputDesc("x").GetDataType());
    (void)op.UpdateOutputDesc("y", output);
    return GRAPH_SUCCESS;
  }

  Tensor data2;
  if (op.GetInputConstData("crops", data2) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(), "Get constValue failed of [crops]. Set unkonwn shape.");
    Shape unknown_4d({UNKNOWN_DIM, UNKNOWN_DIM, UNKNOWN_DIM, UNKNOWN_DIM});
    CHECK_FORMAT(op.GetInputDesc("x").GetFormat());
    TensorDesc output(unknown_4d, op.GetInputDesc("x").GetFormat(), op.GetInputDesc("x").GetDataType());
    (void)op.UpdateOutputDesc("y", output);
    return GRAPH_SUCCESS;
  }

  DataType dtype2 = op.GetInputDesc("crops").GetDataType();
  std::vector<int64_t> const_vec2;
  CalcBatchToSpace(data2, dtype2, const_vec2);

  auto tensor_desc = op.GetInputDesc("x");
  auto shape = tensor_desc.GetShape();
  auto dtype = tensor_desc.GetDataType();
  std::vector<int64_t> shape_vector = shape.GetDims();
  auto format = tensor_desc.GetFormat();
  TensorDesc output = op.GetOutputDesc("y");

  std::vector<std::vector<int64_t>> crops{{0, 0}, {0, 0}};
  crops[0][0] = const_vec2[0];
  crops[0][1] = const_vec2[1];
  crops[1][0] = const_vec2[2];
  crops[1][1] = const_vec2[3];
  std::vector<int64_t> y_shape;

  if (block_size < 2) {
    OP_LOGE(op.GetName().c_str(), "block_size need greater than or equals to 2");
    return GRAPH_FAILED;
  }
  if ((shape_vector[0] % (block_size * block_size)) != 0) {
    OP_LOGE(op.GetName().c_str(),
            "batch_size should be divisible by "
            "the square of block_size.");
    return GRAPH_FAILED;
  }
  if (shape_vector.size() < 4) {
    OP_LOGE(op.GetName().c_str(), "Input shape size must >= 4, but got %d", shape_vector.size());
    return GRAPH_FAILED;
  }
  if (format == FORMAT_NCHW) {
    y_shape.push_back(shape_vector[0] / block_size / block_size);
    y_shape.push_back(shape_vector[1]);
    y_shape.push_back(shape_vector[2] * block_size - crops[0][0] - crops[0][1]);
    y_shape.push_back(shape_vector[3] * block_size - crops[1][0] - crops[1][1]);
  } else {  // without NCHW all other formats set as NHWC
    y_shape.push_back(shape_vector[0] / block_size / block_size);
    y_shape.push_back(shape_vector[1] * block_size - crops[0][0] - crops[0][1]);
    y_shape.push_back(shape_vector[2] * block_size - crops[1][0] - crops[1][1]);
    y_shape.push_back(shape_vector[3]);
  }
  Shape out_shape(y_shape);
  output.SetShape(out_shape);
  output.SetDataType(dtype);
  (void)op.UpdateOutputDesc("y", output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BatchToSpace, BatchToSpaceInferShape);
// ----------------BatchToSpace Op End-------------------

// ----------------BatchToSpaceD Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(BatchToSpaceDInferShape) {
  auto tensor_desc = op.GetInputDesc("x");
  auto dtype = tensor_desc.GetDataType();
  auto shape = tensor_desc.GetShape();
  std::vector<int64_t> shape_vector = shape.GetDims();
  auto format = tensor_desc.GetFormat();

  int64_t block_size;
  if (op.GetAttr("block_size", block_size) == ge::GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "block_size");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue block_size failed!");
    return GRAPH_FAILED;
  }
  if (block_size < 2) {
    OpsAttrValueErrReport(op.GetName(), "block_size", "greater than or equals to 2", ConcatString(block_size));
    OP_LOGE(op.GetName().c_str(), "block_size need greater than or equals to 2");
    return GRAPH_FAILED;
  }
  if ((shape_vector[0] % (block_size * block_size)) != 0) {
    OP_LOGE(op.GetName().c_str(),
            "batch_size  should be divisible by "
            "the square of block_size.");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> crops;
  if (op.GetAttr("crops", crops) == ge::GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "crops");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue crops failed!");
    return GRAPH_FAILED;
  }
  if ((crops.size() != 4)) {
    OpsAttrValueErrReport(op.GetName(), "crops'shape size", "4", ConcatString(crops.size()));
    OP_LOGE(op.GetName().c_str(), "the shape of crops should be 2x2 !");
    return GRAPH_FAILED;
  }
  if ((crops[0] < 0) || (crops[1] < 0) || (crops[2] < 0) || (crops[3] < 0)) {
    OP_LOGE(op.GetName().c_str(), "the value of crops should be greater or equal to 0");
    return GRAPH_FAILED;
  }
  TensorDesc output = op.GetOutputDesc("y");
  std::vector<int64_t> y_shape;
  if (shape_vector.size() < 4) {
    OpsAttrValueErrReport(op.GetName(), "x'shape size", "greater than or equals to 4",
                          ConcatString(shape_vector.size()));
    OP_LOGE(op.GetName().c_str(), "Input shape size must >= 4, but got %d", shape_vector.size());
    return GRAPH_FAILED;
  }
  if (format == FORMAT_NCHW) {
    y_shape.push_back(shape_vector[0] / block_size / block_size);
    y_shape.push_back(shape_vector[1]);
    y_shape.push_back(shape_vector[2] * block_size - crops[0] - crops[1]);
    y_shape.push_back(shape_vector[3] * block_size - crops[2] - crops[3]);
  } else {  // without NCHW all other formats set as NHWC
    y_shape.push_back(shape_vector[0] / block_size / block_size);
    y_shape.push_back(shape_vector[1] * block_size - crops[0] - crops[1]);
    y_shape.push_back(shape_vector[2] * block_size - crops[2] - crops[3]);
    y_shape.push_back(shape_vector[3]);
  }
  Shape out_shape(y_shape);
  output.SetShape(out_shape);
  output.SetDataType(dtype);
  (void)op.UpdateOutputDesc("y", output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BatchToSpaceD, BatchToSpaceDInferShape);
// ----------------BatchToSpaceD Op End-------------------

// ----------------Unapck Op-------------------
IMPLEMT_COMMON_INFERFUNC(UnpackInferShape) {
  OP_LOGI(op.GetName().c_str(), "UnpackInferShape function start!");
  std::vector<std::pair<int64_t, int64_t>> x_range;
  std::vector<std::pair<int64_t, int64_t>> out_range;
  Shape output_shape;

  Shape shape_x = op.GetInputDesc("x").GetShape();
  op.GetInputDesc("x").GetShapeRange(x_range);
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("y");

  // check value of aixs and num
  int64_t axis{0};
  if (op.GetAttr("axis", axis) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "axis");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue axis failed!");
    return GRAPH_FAILED;
  }

  int64_t num{0};
  if (op.GetAttr("num", num) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "num");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue num failed!");
    return GRAPH_FAILED;
  }

  if (RankKnown(shape_x)) {
    int64_t x_dims = shape_x.GetDimNum();
    int64_t real_axis = (axis >= 0) ? axis : axis + x_dims;
    if (real_axis < 0 || real_axis >= x_dims) {
      OpsInputShapeDimErrReport(op.GetName(), "Axis", ConcatString(x_dims), ConcatString(0), ConcatString(real_axis));
      OP_LOGE(op.GetName().c_str(), "Axis exceeding the prescribed range.");
      return GRAPH_FAILED;
    }
    // infer output shape
    std::vector<int64_t> output_vec;
    for (int64_t i = 0; i < x_dims; i++) {
      if (i != real_axis) {
        output_vec.push_back(shape_x.GetDim(i));
        if (static_cast<int64_t>(x_range.size()) == x_dims) {
          out_range.push_back(x_range[i]);
        }
      }
    }
    output_shape = Shape(output_vec);
  } else {
    Shape unknown_shape(UNKNOWN_SHAPE);
    output_shape = unknown_shape;
  }
  for (int64_t i = 0; i < num; i++) {
    tensordesc_output.SetShape(output_shape);
    tensordesc_output.SetShapeRange(out_range);
    tensordesc_output.SetDataType(input_dtype);
    if (op.UpdateDynamicOutputDesc("y", i, tensordesc_output) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
      return GRAPH_FAILED;
    }
  }
  OP_LOGI(op.GetName().c_str(), "UnpackInferShape function End!");
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Unpack, UnpackInferShape);
// -----------------Unpack END------------------------

// ----------------ExtractImagePatches-------------------
static std::vector<int64_t> GetAttrValue(const Operator& op, const std::string& key_name) {
  std::vector<int64_t> list;
  if (op.GetAttr(key_name, list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue failed!");
  }
  return list;
}

static bool CheckListEmptyAndValue(const std::string& op_name, const std::vector<int64_t>& list,
                                   const std::string& attr_name) {
  if (list.size() < 3) {
    OP_LOGE(op_name.c_str(), "The %s dose not have enough elements(%u)!", attr_name.c_str(), list.size());
    return false;
  }
  if (list.at(0) != 1 || list.at(1) < 1 || list.at(2) < 1 || list.at(0) != 1) {
    OP_LOGE(op_name.c_str(), "The %s value is wrong !", attr_name.c_str());
    return false;
  }
  return true;
}

IMPLEMT_VERIFIER(ExtractImagePatches, ExtractImagePatchesVerify) {
  std::vector<int64_t> ksize;
  ksize = GetAttrValue(op, "ksizes");
  if (!CheckListEmptyAndValue(op.GetName(), ksize, "ksizes")) {
    return GRAPH_FAILED;
  }

  std::vector<int64_t> stride;
  stride = GetAttrValue(op, "strides");
  if (!CheckListEmptyAndValue(op.GetName(), stride, "strides")) {
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dilation;
  dilation = GetAttrValue(op, "rates");
  if (!CheckListEmptyAndValue(op.GetName(), dilation, "rates")) {
    return GRAPH_FAILED;
  }

  std::string padding;
  if (op.GetAttr("padding", padding) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get padding failed!");
    return GRAPH_FAILED;
  }

  if (padding != "SAME" && padding != "VALID") {
    OP_LOGE(op.GetName().c_str(), "Padding only supported SAME and VALID!");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ExtractImagePatchesInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter op_proto inferfunction!");

  std::vector<int64_t> ksize;
  ksize = GetAttrValue(op, "ksizes");

  std::vector<int64_t> stride;
  stride = GetAttrValue(op, "strides");

  std::vector<int64_t> dilation;
  dilation = GetAttrValue(op, "rates");

  std::string padding;
  if (op.GetAttr("padding", padding) == GRAPH_FAILED) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue padding failed!");
    return GRAPH_FAILED;
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr desc_in_ptr = op_desc->MutableInputDesc("x");
  GeTensorDescPtr desc_out_ptr = op_desc->MutableOutputDesc("y");
  auto dtype = desc_in_ptr->GetDataType();
  auto shape_in = desc_in_ptr->GetShape();

  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_FAILED || (data_format != "NHWC" && data_format != "NCHW")) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr data_format failed, data_format must be NHWC or NCHW!");
    return GRAPH_FAILED;
  }

  auto x_format = desc_in_ptr->GetOriginFormat();
  const std::map<std::string, Format> format_map{{"NHWC", FORMAT_NHWC}, {"NCHW", FORMAT_NCHW}};

  // Set ori_format as data_format if input ori_format is the default value ND.
  if (x_format == FORMAT_ND) {
    desc_in_ptr->SetOriginFormat(format_map.at(data_format));
    desc_in_ptr->SetFormat(format_map.at(data_format));
    desc_out_ptr->SetOriginFormat(format_map.at(data_format));
    desc_out_ptr->SetFormat(format_map.at(data_format));
  }

  x_format = desc_in_ptr->GetOriginFormat();
  if (x_format != FORMAT_NHWC && x_format != FORMAT_NCHW) {
    OP_LOGE(op.GetName().c_str(), "Attr x_format only support NHWC or NCHW");
    return GRAPH_FAILED;
  }

  std::map<char, int> idx_map{{'N', 0}, {'H', 1}, {'W', 2}, {'C', 3}};
  if (x_format == FORMAT_NCHW) {
    idx_map = {{'N', 0}, {'C', 1}, {'H', 2}, {'W', 3}};
  }

  int64_t in_n = shape_in.GetDim(idx_map['N']);
  int64_t in_h = shape_in.GetDim(idx_map['H']);
  int64_t in_w = shape_in.GetDim(idx_map['W']);
  int64_t in_c = shape_in.GetDim(idx_map['C']);

  int64_t filter_h = ksize.at(idx_map['H']);
  int64_t filter_w = ksize.at(idx_map['W']);
  int64_t stride_h = stride.at(idx_map['H']);
  int64_t stride_w = stride.at(idx_map['W']);
  int64_t dilation_h = dilation.at(idx_map['H']);
  int64_t dilation_w = dilation.at(idx_map['W']);

  int64_t effective_filter_h = (filter_h - 1) * dilation_h + 1;
  int64_t effective_filter_w = (filter_w - 1) * dilation_w + 1;
  int64_t out_h{0};
  int64_t out_w{0};
  int64_t out_c{0};
  if (padding == "VALID") {
    out_h = (in_h - effective_filter_h + stride_h) / stride_h;
    out_w = (in_w - effective_filter_w + stride_w) / stride_w;
  } else if (padding == "SAME") {
    out_h = (in_h + stride_h - 1) / stride_h;
    out_w = (in_w + stride_w - 1) / stride_w;
  }
  out_c = in_c * filter_h * filter_w;
  std::vector<int64_t> out_dim{in_n, out_h, out_w, out_c};
  if(x_format == FORMAT_NCHW) {
    out_dim = {in_n, out_c, out_h, out_w};
  }

  desc_out_ptr->SetShape(ge::GeShape(out_dim));
  desc_out_ptr->SetDataType(dtype);
  return GRAPH_SUCCESS;
}

static void InferHExtractImagePatches(int64_t kernel, int64_t dilation, int64_t stride, int64_t origin_input,
                                      const vector<int64_t>& output_slice, vector<int64_t>& input_slice) {
  int64_t slice_start = output_slice[0] * stride;
  if (slice_start < 0) {
    slice_start = 0;
  }
  int64_t slice_end = output_slice[1] * stride + dilation * (kernel - 1);
  if (slice_end >= origin_input) {
    slice_end = origin_input - 1;
  }
  input_slice = {slice_start, slice_end};
}
/*!
 * @brief provide ExtractImagePatches operator slice data
 * @param ExtractImagePatches Operator type.
 * @param ExtractImagePatchesInferDataSlice slice data function
 * @return Status The processing flow result.
 */
IMPLEMT_INFER_DATA_SLICE(ExtractImagePatches, ExtractImagePatchesInferDataSlice) {
  OP_LOGI(op.GetName().c_str(), "Enter ExtractImagePatches InferDataSlice");

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_in_ptr = op_desc->MutableInputDesc("x");
  GeTensorDescPtr tensor_out_ptr = op_desc->MutableOutputDesc("y");
  auto shape_in = tensor_in_ptr->GetShape();

  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_FAILED || (data_format != "NHWC" && data_format != "NCHW")) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr data_format failed, data_format must be NHWC or NCHW!");
    return GRAPH_FAILED;
  }

  auto images_format = tensor_in_ptr->GetOriginFormat();
  const std::map<std::string, Format> format_map{{"NHWC", FORMAT_NHWC}, {"NCHW", FORMAT_NCHW}};

  // Set ori_format as data_format if input ori_format is the default value ND.
  if (images_format == FORMAT_ND) {
    tensor_in_ptr->SetOriginFormat(format_map.at(data_format));
    tensor_in_ptr->SetFormat(format_map.at(data_format));
  }

  images_format = tensor_in_ptr->GetOriginFormat();
  if (images_format != FORMAT_NHWC && images_format != FORMAT_NCHW) {
    OP_LOGE(op.GetName().c_str(), "Attr x_format only support NHWC or NCHW");
    return GRAPH_FAILED;
  }

  std::map<char, int> idx_map{{'N', 0}, {'H', 1}, {'W', 2}, {'C', 3}};
  if (images_format == FORMAT_NCHW) {
    idx_map = {{'N', 0}, {'C', 1}, {'H', 2}, {'W', 3}};
  }

  std::vector<int64_t> kernel_size;
  kernel_size = GetAttrValue(op, "ksizes");

  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");

  std::vector<int64_t> dilations;
  dilations = GetAttrValue(op, "rates");

  int64_t images_h = shape_in.GetDim(idx_map['H']);
  int64_t ksize_h = kernel_size.at(idx_map['H']);
  int64_t stride_h = strides.at(idx_map['H']);
  int64_t dilation_h = dilations.at(idx_map['h']);

  vector<vector<int64_t>> y_data_slice = {{}, {}, {}, {}, {}};
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}};
  if (!AttrUtils::GetListListInt(tensor_out_ptr, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op.GetName().c_str(), "No data slice, not need infer input");
    return GRAPH_FAILED;
  }

  bool need_infer = false;
  bool have_slice = false;
  for (unsigned idx = 0; idx < y_data_slice.size(); idx++) {
    if (y_data_slice[idx].size() > 0) {
      have_slice = true;
      if (idx == 2) {
        need_infer = true;
        vector<int64_t> slice_data_h;
        InferHExtractImagePatches(ksize_h, dilation_h, stride_h, images_h, y_data_slice[idx], slice_data_h);
        OP_LOGD(op.GetName().c_str(),
                "ExtractImagePatches h axis slice ori_scope is [%d, %d], calced output scope is [%d, %d]",
                slice_data_h[0], slice_data_h[1], y_data_slice[idx][0], y_data_slice[idx][1]);
        x_data_slice[idx] = slice_data_h;
      }
    }
  }
  if (!have_slice) {
    return GRAPH_FAILED;
  }
  if (!need_infer) {
    return NO_OVERLAP_DIM;
  } else {
    if (!AttrUtils::SetListListInt(tensor_in_ptr, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
  OP_LOGI(op.GetName().c_str(), "Calc ExtractImagePatches InferDataSlice end!");
}

COMMON_INFER_FUNC_REG(ExtractImagePatches, ExtractImagePatchesInferShape);
VERIFY_FUNC_REG(ExtractImagePatches, ExtractImagePatchesVerify);
INFER_DATA_SLICE_FUNC_REG(ExtractImagePatches, ExtractImagePatchesInferDataSlice);
// ----------------ExtractImagePatches END-------------------

// ----------------ExtractVolumePatches-------------------
static std::vector<int64_t> GetAttrValueVolume(const Operator& op, const std::string& key_name) {
  std::vector<int64_t> list;
  if (op.GetAttr(key_name, list) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue failed!");
  }
  return list;
}

static bool CheckListEmptyAndValueVolume(const std::string& op_name, const std::vector<int64_t>& list,
                                         const std::string& attr_name) {
  if (list.size() < 5) {
    OP_LOGE(op_name.c_str(), "The %s dose not have enough elements(%u)!", attr_name.c_str(), list.size());
    return false;
  }
  if (list.at(0) != 1 || list.at(1) < 1 || list.at(2) < 1 || list.at(3) < 1 || list.at(4) != 1) {
    OP_LOGE(op_name.c_str(), "The %s value is wrong !", attr_name.c_str());
    return false;
  }
  return true;
}

IMPLEMT_VERIFIER(ExtractVolumePatches, ExtractVolumePatchesVerify) {
  std::vector<int64_t> ksize;
  ksize = GetAttrValueVolume(op, "ksizes");
  if (!CheckListEmptyAndValueVolume(op.GetName(), ksize, "ksizes")) {
    return GRAPH_FAILED;
  }

  std::vector<int64_t> stride;
  stride = GetAttrValueVolume(op, "strides");
  if (!CheckListEmptyAndValueVolume(op.GetName(), stride, "strides")) {
    return GRAPH_FAILED;
  }

  std::string padding;
  if (op.GetAttr("padding", padding) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get padding failed!");
    return GRAPH_FAILED;
  }

  if (padding != "SAME" && padding != "VALID") {
    OP_LOGE(op.GetName().c_str(), "Padding only supported SAME and VALID!");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ExtractVolumePatchesInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter op_proto inferfunction!");

  std::vector<int64_t> ksize;
  ksize = GetAttrValueVolume(op, "ksizes");
  std::vector<int64_t> stride;
  stride = GetAttrValueVolume(op, "strides");

  std::string padding;
  if (op.GetAttr("padding", padding) == GRAPH_FAILED) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue padding failed!");
    return GRAPH_FAILED;
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto desc_in_ptr = op_desc->MutableInputDesc("x");
  auto shape_in = desc_in_ptr->GetShape();
  auto dtype = desc_in_ptr->GetDataType();

  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_FAILED || (data_format != "NDHWC" && data_format != "NCDHW")) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr data_format failed, data_format must be NDHWC or NCDHW!");
    return GRAPH_FAILED;
  }

  auto x_format = desc_in_ptr->GetOriginFormat();
  const std::map<std::string, Format> format_map{{"NDHWC", FORMAT_NDHWC}, {"NCDHW", FORMAT_NCDHW}};

  // Set ori_format as data_format if input ori_format is the default value ND.
  if (x_format == FORMAT_ND) {
    desc_in_ptr->SetOriginFormat(format_map.at(data_format));
    desc_in_ptr->SetFormat(format_map.at(data_format));
  }

  x_format = desc_in_ptr->GetOriginFormat();

  if (x_format != FORMAT_NDHWC && x_format != FORMAT_NCDHW) {
    OP_LOGE(op.GetName().c_str(), "Attr x_format only support NDHWC or NCDHW");
    return GRAPH_FAILED;
  }

  std::map<char, int> idx_map{{'N', 0}, {'D', 1}, {'H', 2},{'W', 3},{'C', 4}};
  if (x_format == FORMAT_NCDHW) {
    idx_map = {{'N', 0}, {'C', 1}, {'D', 2}, {'H', 3}, {'W', 4}};
  }

  int64_t in_n = shape_in.GetDim(idx_map['N']);
  int64_t in_d = shape_in.GetDim(idx_map['D']);
  int64_t in_h = shape_in.GetDim(idx_map['H']);
  int64_t in_w = shape_in.GetDim(idx_map['W']);
  int64_t in_c = shape_in.GetDim(idx_map['C']);

  int64_t filter_d = ksize.at(idx_map['D']);
  int64_t filter_h = ksize.at(idx_map['H']);
  int64_t filter_w = ksize.at(idx_map['W']);
  int64_t stride_d = stride.at(idx_map['D']);
  int64_t stride_h = stride.at(idx_map['H']);
  int64_t stride_w = stride.at(idx_map['W']);

  int64_t out_d{0};
  int64_t out_h{0};
  int64_t out_w{0};
  if (padding == "VALID") {
    out_d = (in_d - filter_d + stride_d) / stride_d;
    out_h = (in_h - filter_h + stride_h) / stride_h;
    out_w = (in_w - filter_w + stride_w) / stride_w;
  } else if (padding == "SAME") {
    out_d = (in_d + stride_d - 1) / stride_d;
    out_h = (in_h + stride_h - 1) / stride_h;
    out_w = (in_w + stride_w - 1) / stride_w;
  }
  int64_t out_c = in_c * filter_d * filter_h * filter_w;

  std::vector<int64_t> out_dim = {in_n, out_d, out_h, out_w, out_c};
  if (x_format == FORMAT_NCDHW) {
    out_dim = {in_n, out_c, out_d, out_h, out_w};
  }
  auto des_out_ptr = op_desc->MutableOutputDesc("y");
  des_out_ptr->SetShape(ge::GeShape(out_dim));
  des_out_ptr->SetDataType(dtype);

  return GRAPH_SUCCESS;
}

static void InferHDExtractVolumePatches(int64_t kernel, int64_t stride, int64_t origin_input,
                                        const vector<int64_t>& output_slice, vector<int64_t>& input_slice) {
  int64_t slice_start = output_slice[0] * stride;
  if (slice_start < 0) {
    slice_start = 0;
  }

  int64_t slice_end = output_slice[1] * stride + (kernel - 1);
  if (slice_end >= origin_input) {
    slice_end = origin_input - 1;
  }
  input_slice = {slice_start, slice_end};
}
/*!
 * @brief provide ExtractVolumePatches operator slice data
 * @param ExtractVolumePatches Operator type.
 * @param ExtractVolumePatchesInferDataSlice slice data function
 * @return Status The processing flow result.
 */
IMPLEMT_INFER_DATA_SLICE(ExtractVolumePatches, ExtractVolumePatchesInferDataSlice) {
  OP_LOGI(op.GetName().c_str(), "Enter ExtractVolumePatches InferDataSlice");

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_in_ptr = op_desc->MutableInputDesc("x");
  GeTensorDescPtr tensor_out_ptr = op_desc->MutableOutputDesc("y");
  auto shape_in = tensor_in_ptr->GetShape();

  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_FAILED || (data_format != "NDHWC" && data_format != "NCDHW")) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr data_format failed, data_format must be NDHWC or NCDHW!");
    return GRAPH_FAILED;
  }

  auto x_format = tensor_in_ptr->GetOriginFormat();
  const std::map<std::string, Format> format_map{{"NDHWC", FORMAT_NDHWC}, {"NCDHW", FORMAT_NCDHW}};

  // Set ori_format as data_format if input ori_format is the default value ND.
  if (x_format == FORMAT_ND) {
    tensor_in_ptr->SetOriginFormat(format_map.at(data_format));
    tensor_in_ptr->SetFormat(format_map.at(data_format));
  }

  x_format = tensor_in_ptr->GetOriginFormat();
  if (x_format != FORMAT_NDHWC && x_format != FORMAT_NCDHW) {
    OP_LOGE(op.GetName().c_str(), "Input x format only support NDHWC or NCDHW");
    return GRAPH_FAILED;
  }

  std::map<char, int> idx_map{{'N', 0}, {'D', 1}, {'H', 2}, {'W', 3}, {'C', 4}};
  if (x_format == FORMAT_NCDHW) {
    idx_map = {{'N', 0}, {'C', 1}, {'D', 2}, {'H', 3}, {'W', 4}};
  }


  std::vector<int64_t> kernel_size;
  kernel_size = GetAttrValue(op, "ksizes");

  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");

  int64_t input_d = shape_in.GetDim(idx_map['D']);
  int64_t input_h = shape_in.GetDim(idx_map['H']);
  int64_t filter_d = kernel_size.at(idx_map['D']);
  int64_t filter_h = kernel_size.at(idx_map['H']);
  int64_t stride_d = strides.at(idx_map['D']);
  int64_t stride_h = strides.at(idx_map['H']);

  vector<vector<int64_t>> y_data_slice = {{}, {}, {}, {}, {}, {}};
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}, {}};
  if (!AttrUtils::GetListListInt(tensor_out_ptr, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op.GetName().c_str(), "No data slice, not need infer input");
    return GRAPH_FAILED;
  }

  bool need_infer = false;
  bool have_slice = false;
  for (unsigned idx = 0; idx < y_data_slice.size(); idx++) {
    if (y_data_slice[idx].size() > 0) {
      have_slice = true;
      if (idx == 1) {
        need_infer = true;
        vector<int64_t> slice_data_d;
        InferHDExtractVolumePatches(filter_d, stride_d, input_d, y_data_slice[idx], slice_data_d);
        OP_LOGD(op.GetName().c_str(),
                "ExtractVolumePatches d axis slice ori_scope is [%d, %d], calced output scope is [%d, %d]",
                slice_data_d[0], slice_data_d[1], y_data_slice[idx][0], y_data_slice[idx][1]);
        x_data_slice[idx] = slice_data_d;
      } else if(idx == 3) {
        need_infer = true;
        vector<int64_t> slice_data_h;
        InferHDExtractVolumePatches(filter_h, stride_h, input_h, y_data_slice[idx], slice_data_h);
        OP_LOGD(op.GetName().c_str(),
                "ExtractVolumePatches h axis slice ori_scope is [%d, %d], calced output scope is [%d, %d]",
                slice_data_h[0], slice_data_h[1], y_data_slice[idx][0], y_data_slice[idx][1]);
        x_data_slice[idx] = slice_data_h;
      }
    }
  }

  if (!have_slice) {
    return GRAPH_FAILED;
  }
  if (!need_infer) {
    return NO_OVERLAP_DIM;
  } else {
    if (!AttrUtils::SetListListInt(tensor_in_ptr, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
  OP_LOGI(op.GetName().c_str(), "Calc ExtractVolumePatches InferDataSlice end!");
}

COMMON_INFER_FUNC_REG(ExtractVolumePatches, ExtractVolumePatchesInferShape);
VERIFY_FUNC_REG(ExtractVolumePatches, ExtractVolumePatchesVerify);
INFER_DATA_SLICE_FUNC_REG(ExtractVolumePatches, ExtractVolumePatchesInferDataSlice);
// ----------------ExtractVolumePatches END-------------------

// -----------------------ConfusionTranspose---------------------
IMPLEMT_COMMON_INFERFUNC(ConfusionTransposeInferShape) {
  Shape input_shape = op.GetInputDesc("x").GetShape();
  Shape shape = op.GetInputDesc("shape").GetShape();
  std::vector<int64_t> perm_list;
  if (GRAPH_SUCCESS != op.GetAttr("perm", perm_list)) {
    OP_LOGE("GetOpAttr perm failed!");
    return GRAPH_FAILED;
  }
  bool transpose_first;
  if (GRAPH_SUCCESS != op.GetAttr("transpose_first", transpose_first)) {
    OP_LOGE("GetOpAttr transpose_first failed!");
    return GRAPH_FAILED;
  }

  size_t dim_num = shape.GetDimNum();
  std::vector<int64_t> out_vec;

  if (transpose_first == true) {
    for (size_t i = 0; i < dim_num; ++i) {
      out_vec.push_back(shape.GetDim(i));
    }
  } else {
    for (size_t i = 0; i < dim_num; ++i) {
      out_vec.push_back(shape.GetDim(perm_list[i]));
    }
  }

  Shape out_shape(out_vec);
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(out_shape);
  tensordesc_output.SetDataType(op.GetInputDesc("x").GetDataType());
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(ConfusionTranspose, ConfusionTransposeInferShape);

IMPLEMT_COMMON_INFERFUNC(ConfusionTransposeDInferShape) {
  Shape input_shape = op.GetInputDesc("x").GetShape();
  std::vector<int64_t> perm_list;
  if (GRAPH_SUCCESS != op.GetAttr("perm", perm_list)) {
    OpsGetAttrErrReport(op.GetName(), "perm");
    OP_LOGE("GetOpAttr perm failed!");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> shape_list;
  if (GRAPH_SUCCESS != op.GetAttr("shape", shape_list)) {
    OpsGetAttrErrReport(op.GetName(), "shape");
    OP_LOGE("GetOpAttr shape failed!");
    return GRAPH_FAILED;
  }
  bool transpose_first;
  if (GRAPH_SUCCESS != op.GetAttr("transpose_first", transpose_first)) {
    OpsGetAttrErrReport(op.GetName(), "transpose_first");
    OP_LOGE("GetOpAttr transpose_first failed!");
    return GRAPH_FAILED;
  }

  size_t dim_num = shape_list.size();
  std::vector<int64_t> out_vec;

  if (transpose_first == true) {
    for (size_t i = 0; i < dim_num; ++i) {
      out_vec.push_back(shape_list[i]);
    }
  } else {
    for (size_t i = 0; i < dim_num; ++i) {
      out_vec.push_back(shape_list[perm_list[i]]);
    }
  }

  Shape out_shape(out_vec);
  TensorDesc tensordesc_output = op.GetOutputDesc("y");
  tensordesc_output.SetShape(out_shape);
  tensordesc_output.SetDataType(op.GetInputDesc("x").GetDataType());
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(ConfusionTransposeD, ConfusionTransposeDInferShape);

// -----------------FlattenV2 Op-------------------------
IMPLEMT_VERIFIER(FlattenV2, FlattenV2Verify) {
  int64_t axis = 0;
  int64_t endAxis = 0;
  TensorDesc xDesc = op.GetInputDesc("x");
  if (GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    axis = 1;
  }
  if (GRAPH_SUCCESS != op.GetAttr("end_axis", endAxis)) {
    endAxis = -1;
  }

  int64_t realDimCnt = xDesc.GetRealDimCnt();

  if (axis < 0) {
    axis += realDimCnt;
  }
  if (endAxis < 0) {
    endAxis += realDimCnt;
  }

  if (axis < 0 || axis >= realDimCnt) {
    OP_LOGE("[ERROR] axis out of range!");
    return GRAPH_FAILED;
  }
  if (endAxis < 0 || endAxis >= realDimCnt) {
    OP_LOGE("[ERROR] end_axis out of range!");
    return GRAPH_FAILED;
  }
  if (axis > endAxis) {
    OP_LOGE("[ERROR] axis after end_axis!");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(FlattenV2InferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter FlattenV2 proto inferfunction!");
  TensorDesc xDesc = op.GetInputDesc("x");
  auto xShapeDim = xDesc.GetShape().GetDims();
  auto xDtype = xDesc.GetDataType();

  int64_t axis = 0;
  int64_t endAxis = 0;
  int64_t realDimCnt = xDesc.GetRealDimCnt();

  if (GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    axis = 1;
  }
  if (GRAPH_SUCCESS != op.GetAttr("end_axis", endAxis)) {
    endAxis = -1;
  }

  if (axis < 0) {
    axis += realDimCnt;
  }
  if (endAxis < 0) {
    endAxis += realDimCnt;
  }

  std::vector<int64_t> yShapeDim;

  for (int64_t i = 0; i < axis; i++) {
    yShapeDim.push_back(xShapeDim[i]);
  }

  int64_t dimVal = 1;
  for (int64_t i = axis; i < endAxis + 1; i++) {
    dimVal = dimVal * xShapeDim[i];
  }
  yShapeDim.push_back(dimVal);

  for (int64_t i = endAxis + 1; i < realDimCnt; i++) {
    yShapeDim.push_back(xShapeDim[i]);
  }

  Shape yShape(yShapeDim);
  TensorDesc yDesc = op.GetOutputDesc("y");
  yDesc.SetShape(yShape);
  yDesc.SetDataType(xDtype);
  op.UpdateOutputDesc("y", yDesc);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FlattenV2, FlattenV2InferShape);
VERIFY_FUNC_REG(FlattenV2, FlattenV2Verify);
// -----------------FlattenV2 END-------------------------
}  // namespace ge
