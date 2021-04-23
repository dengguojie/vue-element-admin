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
#include <cmath>
#include <algorithm>
#include "util/util.h"
#include "common_shape_fns.h"
#include "op_log.h"
#include "util/error_util.h"
#include "register/infer_data_slice_registry.h"
#include "graph/common_error_codes.h"
#include "graph/debug/ge_attr_define.h"

  // namespace ge
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
IMPLEMT_COMMON_INFERFUNC(SpaceToBatchNDInferShape) {
  const vector<string> depend_names = {"block_shape", "paddings"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  auto node = NodeUtils::GetNodeFromOperator(op);
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_dims = input_desc->MutableShape().GetDims();
  auto input_dtype = input_desc->GetDataType();

  auto output_desc = op_info->MutableOutputDesc("y");
  output_desc->SetDataType(input_dtype);

  // get const node block_shape
  bool block_done = false;
  std::vector<int64_t> block_shape;
  GeTensorPtr block_tensor = nullptr;
  if (GRAPH_SUCCESS == NodeUtils::GetInputConstData(node, "block_shape", block_tensor)) {
    auto const_desc = op_info->MutableInputDesc("block_shape");
    auto const_dtype = const_desc->GetDataType();
    if (GetConstValue(op, block_tensor, const_dtype, block_shape)) {
      block_done = true;
    } else {
      OP_LOGW(op.GetName().c_str(), "Get Const block_shape value failed.");
    }
  }

  // get const node padding
  bool padding_done = false;
  std::vector<int64_t> paddings;
  GeTensorPtr paddings_tensor = nullptr;
  if (GRAPH_SUCCESS == NodeUtils::GetInputConstData(node, "paddings", paddings_tensor)) {
    auto const_desc = op_info->MutableInputDesc("paddings");
    auto const_dtype = const_desc->GetDataType();
    if (GetConstValue(op, paddings_tensor, const_dtype, paddings)) {
      padding_done = true;
    } else {
      OP_LOGW(op.GetName().c_str(), "Get Const paddings value failed.");
    }
  }

  // if block_shape and paddings are const node, verfify const sizes
  if (!IsUnknownRankShape(input_dims) && padding_done && block_done) {
    if (block_shape.size() < 0 || paddings.size() < 0 || paddings.size() != 2 * block_shape.size()) {
      OP_LOGE(op.GetName().c_str(),
              "block_shape and paddings size must be greater than 0 and paddings size must be twice as "
              "block_shape size, but got bloack_shape size [%d] and paddings size [%d]",
              block_shape.size(), paddings.size());
      return GRAPH_FAILED;
    }
    if (input_dims.size() <= block_shape.size()) {
      OP_LOGE(
          op.GetName().c_str(),
          "input_shape size must be greater than block_shape size, "
          "but got input_shape size [%d], block_shape size [%d]",
          input_dims.size(), block_shape.size());
      return GRAPH_FAILED;
    }
  }

  // not dynamic case, only set shape
  if (!IsUnknown(input_dims) && padding_done && block_done) {
    std::vector<int64_t> output_dims;
    int64_t first_dim = input_dims[0];
    for (size_t i = 0; i < block_shape.size(); i++) {
      first_dim = first_dim * block_shape[i];
    }
    output_dims.push_back(first_dim);
    for (size_t i = 1; i <= block_shape.size(); i++) {
      output_dims.push_back((input_dims[i] + paddings[2 * i - 2] + paddings[2 * i - 1]) / block_shape[i - 1]);
    }
    for (size_t i = block_shape.size() + 1; i < input_dims.size(); i++) {
      output_dims.push_back(input_dims[i]);
    }
    output_desc->SetShape(GeShape(output_dims));
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -2, output is -2
  if (IsUnknownRankShape(input_dims)) {
    output_desc->SetShape(GeShape(input_dims));
    OP_LOGW(op.GetName().c_str(), "input shape is UnknownRank, set output is UnknownRank.");
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -1, output is -1
  std::vector<std::pair<int64_t, int64_t>> input_range;
  input_desc->GetShapeRange(input_range);
  MakeUpShapeRange(input_dims, input_range);

  auto block_desc = op_info->MutableInputDesc("block_shape");
  std::vector<int64_t> block_dims = block_desc->MutableShape().GetDims();
  std::vector<std::pair<int64_t, int64_t>> block_range;
  if (IsUnknownRankShape(block_dims)) {
    block_dims = {-1};
  } else {
    block_desc->GetShapeRange(block_range);
  }
  MakeUpShapeRange(block_dims, block_range);

  auto paddings_desc = op_info->MutableInputDesc("paddings");
  std::vector<int64_t> paddings_dims = paddings_desc->MutableShape().GetDims();
  std::vector<std::pair<int64_t, int64_t>> paddings_range;
  if (IsUnknownRankShape(paddings_dims)) {
    paddings_dims = {-1, -1};
  } else {
    paddings_desc->GetShapeRange(paddings_range);
  }
  MakeUpShapeRange(paddings_dims, paddings_range);

  // the max length of block_shape
  auto block_size_max = std::min(paddings_range[0].second, block_range[0].second);
  block_size_max = block_size_max == -1 ? std::max(paddings_range[0].second, block_range[0].second) : block_size_max;
  block_size_max = block_size_max == -1 ? static_cast<int64_t>(input_dims.size()) - 1
                                        : std::min(block_size_max, static_cast<int64_t>(input_dims.size()) - 1);
  // the total ele of block_shape
  int64_t block_total = 1;
  if (block_done) {
    for (size_t i = 0; i < block_shape.size(); i++) {
      block_total = block_total * block_shape[i];
    }
  }

  // infer output shape and range
  std::vector<int64_t> output_dims;
  std::vector<std::pair<int64_t, int64_t>> output_range;
  int64_t dim = (input_dims[0] == -1 || !block_done) ? -1 : input_dims[0] * block_total;
  int64_t range_min = !block_done ? input_range[0].first : input_range[0].first * block_total;
  int64_t range_max = (input_range[0].second == -1 || !block_done) ? -1 : input_range[0].second * block_total;
  output_dims.push_back(dim);
  output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  for (size_t i = 1; i <= block_size_max; i++) {
    dim = (input_dims[i] == -1 || !block_done || !padding_done)
              ? -1
              : (input_dims[i] + paddings[2 * i - 2] + paddings[2 * i - 1]) / block_shape[i - 1];
    range_min = !padding_done ? input_range[i].first : input_range[i].first + paddings[2 * i - 2] + paddings[2 * i - 1];
    range_max = (input_range[i].second == -1 || !padding_done)
                    ? -1
                    : input_range[i].second + paddings[2 * i - 2] + paddings[2 * i - 1];
    range_min = !block_done ? 1 : std::ceil(static_cast<float>(range_min) / static_cast<float>(block_shape[i - 1]));
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = (range_max == -1 || !block_done) ? range_max : range_max / block_shape[i - 1];
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  }
  for (size_t i = block_size_max + 1; i < input_dims.size(); i++) {
    output_dims.push_back(input_dims[i]);
    output_range.push_back(input_range[i]);
  }

  output_desc->SetShape(GeShape(output_dims));
  output_desc->SetShapeRange(output_range);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SpaceToBatchND, SpaceToBatchNDVerify) {
  if (!CheckTwoInputDtypeSame(op, "block_shape", "paddings")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SpaceToBatchND, SpaceToBatchNDInferShape);
VERIFY_FUNC_REG(SpaceToBatchND, SpaceToBatchNDVerify);
// ----------------SpaceToBatchND Op End-------------------

// ----------------SpaceToBatchNDD Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(SpaceToBatchNDDInferShape) {
  auto node = NodeUtils::GetNodeFromOperator(op);
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_dims = input_desc->MutableShape().GetDims();
  auto input_dtype = input_desc->GetDataType();

  auto output_desc = op_info->MutableOutputDesc("y");
  output_desc->SetDataType(input_dtype);

  // get attr block_shape
  std::vector<int64_t> block_shape;
  if (op.GetAttr("block_shape", block_shape) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("block_shape");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get attr paddings
  std::vector<int64_t> paddings;
  if (op.GetAttr("paddings", paddings) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("paddings");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // not dynamic case, only set shape
  if (!IsUnknown(input_dims)) {
    std::vector<int64_t> output_dims;
    int64_t first_dim = input_dims[0];
    for (size_t i = 0; i < block_shape.size(); i++) {
      first_dim = first_dim * block_shape[i];
    }
    output_dims.push_back(first_dim);
    for (size_t i = 1; i <= block_shape.size(); i++) {
      output_dims.push_back((input_dims[i] + paddings[2 * i - 2] + paddings[2 * i - 1]) / block_shape[i - 1]);
    }
    for (size_t i = block_shape.size() + 1; i < input_dims.size(); i++) {
      output_dims.push_back(input_dims[i]);
    }
    output_desc->SetShape(GeShape(output_dims));
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -2, output is -2
  if (IsUnknownRankShape(input_dims)) {
    output_desc->SetShape(GeShape(input_dims));
    OP_LOGW(op.GetName().c_str(), "input shape is UnknownRank, set output is UnknownRank.");
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -1, output is -1
  std::vector<std::pair<int64_t, int64_t>> input_range;
  input_desc->GetShapeRange(input_range);
  MakeUpShapeRange(input_dims, input_range);

  // the total ele of block_shape
  int64_t block_total = 1;
  for (size_t i = 0; i < block_shape.size(); i++) {
    block_total = block_total * block_shape[i];
  }

  // infer output shape and range
  std::vector<int64_t> output_dims;
  std::vector<std::pair<int64_t, int64_t>> output_range;
  int64_t dim = input_dims[0] == -1 ? -1 : input_dims[0] * block_total;
  int64_t range_min = input_range[0].first * block_total;
  int64_t range_max = input_range[0].second == -1 ? -1 : input_range[0].second * block_total;
  output_dims.push_back(dim);
  output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  for (size_t i = 1; i <= block_shape.size(); i++) {
    dim = input_dims[i] == -1 ? -1 : (input_dims[i] + paddings[2 * i - 2] + paddings[2 * i - 1]) / block_shape[i - 1];
    range_min = std::ceil(static_cast<float>(input_range[i].first + paddings[2 * i - 2] + paddings[2 * i - 1]) /
                          static_cast<float>(block_shape[i - 1]));
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = input_range[i].second == -1
                    ? -1
                    : (input_range[i].second + paddings[2 * i - 2] + paddings[2 * i - 1]) / block_shape[i - 1];
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  }
  for (size_t i = block_shape.size() + 1; i < input_dims.size(); i++) {
    output_dims.push_back(input_dims[i]);
    output_range.push_back(input_range[i]);
  }

  output_desc->SetShape(GeShape(output_dims));
  output_desc->SetShapeRange(output_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SpaceToBatchNDD, SpaceToBatchNDDInferShape);
// ----------------SpaceToBatchNDD Op End-------------------

// ----------------BatchToSpaceND Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(BatchToSpaceNDInferShape) {
  const vector<string> depend_names = {"block_shape", "crops"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  auto node = NodeUtils::GetNodeFromOperator(op);
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_dims = input_desc->MutableShape().GetDims();
  auto input_dtype = input_desc->GetDataType();

  auto output_desc = op_info->MutableOutputDesc("y");
  output_desc->SetDataType(input_dtype);

  // get const node block_shape
  bool block_done = false;
  std::vector<int64_t> block_shape;
  GeTensorPtr block_tensor = nullptr;
  if (GRAPH_SUCCESS == NodeUtils::GetInputConstData(node, "block_shape", block_tensor)) {
    auto const_desc = op_info->MutableInputDesc("block_shape");
    auto const_dtype = const_desc->GetDataType();
    if (GetConstValue(op, block_tensor, const_dtype, block_shape)) {
      block_done = true;
    } else {
      OP_LOGW(op.GetName().c_str(), "Get Const block_shape value failed.");
    }
  }

  // get const node crops
  bool crops_done = false;
  std::vector<int64_t> crops;
  GeTensorPtr crops_tensor = nullptr;
  if (GRAPH_SUCCESS == NodeUtils::GetInputConstData(node, "crops", crops_tensor)) {
    auto const_desc = op_info->MutableInputDesc("crops");
    auto const_dtype = const_desc->GetDataType();
    if (GetConstValue(op, crops_tensor, const_dtype, crops)) {
      crops_done = true;
    } else {
      OP_LOGW(op.GetName().c_str(), "Get Const crops value failed.");
    }
  }

  // if block_shape and crops are const node, verfify const sizes
  if (!IsUnknownRankShape(input_dims) && crops_done && block_done) {
    if (block_shape.size() < 0 || crops.size() < 0 || crops.size() != 2 * block_shape.size()) {
      OP_LOGE(op.GetName().c_str(),
              "block_shape and crops size must be greater than 0 and crops size must be twice as "
              "block_shape size, but got bloack_shape size [%d] and crops size [%d]",
              block_shape.size(), crops.size());
      return GRAPH_FAILED;
    }
    if (input_dims.size() <= block_shape.size()) {
      OP_LOGE(
          op.GetName().c_str(),
          "input_shape size must be greater than block_shape size, "
          "but got input_shape size [%d], block_shape size [%d]",
          input_dims.size(), block_shape.size());
      return GRAPH_FAILED;
    }
  }

  // not dynamic case, only set shape
  if (!IsUnknown(input_dims) && crops_done && block_done) {
    std::vector<int64_t> output_dims;
    int64_t first_dim = input_dims[0];
    for (size_t i = 0; i < block_shape.size(); i++) {
      first_dim = first_dim / block_shape[i];
    }
    output_dims.push_back(first_dim);
    for (size_t i = 1; i <= block_shape.size(); i++) {
      output_dims.push_back(input_dims[i] * block_shape[i - 1] - crops[2 * i - 2] - crops[2 * i - 1]);
    }
    for (size_t i = block_shape.size() + 1; i < input_dims.size(); i++) {
      output_dims.push_back(input_dims[i]);
    }
    output_desc->SetShape(GeShape(output_dims));
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -2, output is -2
  if (IsUnknownRankShape(input_dims)) {
    output_desc->SetShape(GeShape(input_dims));
    OP_LOGW(op.GetName().c_str(), "input shape is UnknownRank, set output is UnknownRank.");
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -1, output is -1
  std::vector<std::pair<int64_t, int64_t>> input_range;
  input_desc->GetShapeRange(input_range);
  MakeUpShapeRange(input_dims, input_range);

  auto block_desc = op_info->MutableInputDesc("block_shape");
  std::vector<int64_t> block_dims = block_desc->MutableShape().GetDims();
  std::vector<std::pair<int64_t, int64_t>> block_range;
  if (IsUnknownRankShape(block_dims)) {
    block_dims = {-1};
  } else {
    block_desc->GetShapeRange(block_range);
  }
  MakeUpShapeRange(block_dims, block_range);

  auto crops_desc = op_info->MutableInputDesc("crops");
  std::vector<int64_t> crops_dims = crops_desc->MutableShape().GetDims();
  std::vector<std::pair<int64_t, int64_t>> crops_range;
  if (IsUnknownRankShape(crops_dims)) {
    crops_dims = {-1, -1};
  } else {
    crops_desc->GetShapeRange(crops_range);
  }
  MakeUpShapeRange(crops_dims, crops_range);

  // the max length of block_shape
  auto block_size_max = std::min(crops_range[0].second, block_range[0].second);
  block_size_max = block_size_max == -1 ? std::max(crops_range[0].second, block_range[0].second) : block_size_max;
  block_size_max = block_size_max == -1 ? static_cast<int64_t>(input_dims.size()) - 1
                                        : std::min(block_size_max, static_cast<int64_t>(input_dims.size()) - 1);
  // the total ele of block_shape
  int64_t block_total = 1;
  if (block_done) {
    for (size_t i = 0; i < block_shape.size(); i++) {
      block_total = block_total * block_shape[i];
    }
  }

  // infer output shape and range
  std::vector<int64_t> output_dims;
  std::vector<std::pair<int64_t, int64_t>> output_range;
  int64_t dim = (input_dims[0] == -1 || !block_done) ? -1 : input_dims[0] / block_total;
  int64_t range_min = !block_done ? 1 : std::max(int64_t(input_range[0].first / block_total), int64_t(1));
  int64_t range_max =
      (input_range[0].second == -1 || !block_done) ? input_range[0].second : input_range[0].second / block_total;
  output_dims.push_back(dim);
  output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));

  auto block_shape_min = block_shape;
  auto block_shape_max = block_shape;
  if (!block_done) {
    if (input_dims[0] != -1) {
      block_done = true;
      block_shape_min.clear();
      block_shape_max.clear();
      for (auto i = 0; i < block_size_max; i++) {
        block_shape_min.push_back(1);
        block_shape_max.push_back(input_dims[0]);
      }
    } else if (input_dims[0] == -1 && input_range[0].second != -1) {
      block_done = true;
      block_shape_min.clear();
      block_shape_max.clear();
      for (auto i = 0; i < block_size_max; i++) {
        block_shape_min.push_back(1);
        block_shape_max.push_back(input_range[0].second);
      }
    }
  }

  for (size_t i = 1; i <= block_size_max; i++) {
    dim = (input_dims[i] == -1 || !block_done || !crops_done)
              ? -1
              : input_dims[i] * block_shape[i - 1] - crops[2 * i - 2] - crops[2 * i - 1];
    range_min = !block_done ? input_range[i].first : input_range[i].first * block_shape_min[i - 1];
    range_max = (input_range[i].second == -1 || !block_done) ? -1 : input_range[i].second * block_shape_max[i - 1];
    range_min = !crops_done ? 1 : range_min - crops[2 * i - 2] - crops[2 * i - 1];
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = (range_max == -1 || !crops_done) ? range_max : range_max - crops[2 * i - 2] - crops[2 * i - 1];
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  }
  for (size_t i = block_size_max + 1; i < input_dims.size(); i++) {
    output_dims.push_back(input_dims[i]);
    output_range.push_back(input_range[i]);
  }

  output_desc->SetShape(GeShape(output_dims));
  output_desc->SetShapeRange(output_range);
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
  auto node = NodeUtils::GetNodeFromOperator(op);
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_dims = input_desc->MutableShape().GetDims();
  auto input_dtype = input_desc->GetDataType();

  auto output_desc = op_info->MutableOutputDesc("y");
  output_desc->SetDataType(input_dtype);

  // get attr block_shape
  std::vector<int64_t> block_shape;
  if (op.GetAttr("block_shape", block_shape) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("block_shape");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get attr crops
  std::vector<int64_t> crops;
  if (op.GetAttr("crops", crops) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("crops");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // not dynamic case, only set shape
  if (!IsUnknown(input_dims)) {
    std::vector<int64_t> output_dims;
    int64_t first_dim = input_dims[0];
    for (size_t i = 0; i < block_shape.size(); i++) {
      first_dim = first_dim / block_shape[i];
    }
    output_dims.push_back(first_dim);
    for (size_t i = 1; i <= block_shape.size(); i++) {
      output_dims.push_back(input_dims[i] * block_shape[i - 1] - crops[2 * i - 2] - crops[2 * i - 1]);
    }
    for (size_t i = block_shape.size() + 1; i < input_dims.size(); i++) {
      output_dims.push_back(input_dims[i]);
    }
    output_desc->SetShape(GeShape(output_dims));
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -2, output is -2
  if (IsUnknownRankShape(input_dims)) {
    output_desc->SetShape(GeShape(input_dims));
    OP_LOGW(op.GetName().c_str(), "input shape is UnknownRank, set output is UnknownRank.");
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -1, output is -1
  std::vector<std::pair<int64_t, int64_t>> input_range;
  input_desc->GetShapeRange(input_range);
  MakeUpShapeRange(input_dims, input_range);

  // the total ele of block_shape
  int64_t block_total = 1;
  for (size_t i = 0; i < block_shape.size(); i++) {
    block_total = block_total * block_shape[i];
  }

  // infer output shape and range
  std::vector<int64_t> output_dims;
  std::vector<std::pair<int64_t, int64_t>> output_range;
  int64_t dim = input_dims[0] == -1 ? -1 : input_dims[0] / block_total;
  int64_t range_min = std::max(int64_t(input_range[0].first / block_total), int64_t(1));
  int64_t range_max = input_range[0].second == -1 ? -1 : input_range[0].second / block_total;
  output_dims.push_back(dim);
  output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));

  for (size_t i = 1; i <= block_shape.size(); i++) {
    dim = input_dims[i] == -1 ? -1 : input_dims[i] * block_shape[i - 1] - crops[2 * i - 2] - crops[2 * i - 1];
    range_min = input_range[i].first * block_shape[i - 1] - crops[2 * i - 2] - crops[2 * i - 1];
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = input_range[i].second == -1
                    ? -1
                    : input_range[i].second * block_shape[i - 1] - crops[2 * i - 2] - crops[2 * i - 1];
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  }
  for (size_t i = block_shape.size() + 1; i < input_dims.size(); i++) {
    output_dims.push_back(input_dims[i]);
    output_range.push_back(input_range[i]);
  }

  output_desc->SetShape(GeShape(output_dims));
  output_desc->SetShapeRange(output_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BatchToSpaceNDD, BatchToSpaceNDDInferShape);
// ----------------BatchToSpaceNDD Op End-------------------

// ----------------Flatten Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(FlattenInferShape) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_info->MutableInputDesc("x");
  auto x_shape = x_desc->MutableShape();
  auto input_dtype = x_desc->GetDataType();
  auto x_vector = x_shape.GetDims();

  auto y_desc = op_info->MutableOutputDesc("y");
  y_desc->SetDataType(input_dtype);

  //------------not dynamic case, only set shape-------------------------------------------
  if (!IsUnknown(x_vector)) {
    std::vector<int64_t> yVector;
    yVector.push_back(x_vector[0]);
    int64_t shape_num = 1;
    for (size_t i = 0; i < x_vector.size() - 1; ++i) {
      shape_num = shape_num * x_vector[i + 1];
    }
    yVector.push_back(shape_num);
    y_desc->SetShape(GeShape(yVector));
    return GRAPH_SUCCESS;
  }

  //--------------dynamic case, input shape is -2, output is -2-------------------------
  if (IsUnknownRankShape(x_vector)) {
    y_desc->SetShape(GeShape(x_vector));
    OP_LOGW(op.GetName().c_str(), "input shape is UnknownRank, set output is UnknownRank.");
    return GRAPH_SUCCESS;
  }

  //---------------dynamic case, input shape is -1, output is -1--------------------
  if (!IsUnknownRankShape(x_vector) and x_vector.size() == 1) {
    std::vector<std::pair<int64_t, int64_t>> x_range;
    x_desc->GetShapeRange(x_range);
    y_desc->SetShape(GeShape(x_vector));
    y_desc->SetShapeRange(x_range);
    return GRAPH_SUCCESS;
  }

  //---------------dynamic case, shape range > 1---------------------
  //----------------------shape-----------------------
  std::vector<int64_t> y_vector;
  y_vector.push_back(x_vector[0]);
  y_vector.push_back(-1);
  //----------------------range-----------------------
  std::vector<std::pair<int64_t, int64_t>> x_range;
  x_desc->GetShapeRange(x_range);
  std::vector<std::pair<int64_t, int64_t>> y_range;
  y_range.push_back(x_range[0]);

  int64_t range_min = 1;
  std::vector<int64_t> range_max_vector;
  for (size_t i = 0; i < x_vector.size() - 1; ++i) {
    range_min = range_min * x_range[i + 1].first;
    range_max_vector.push_back(x_range[i + 1].second);
  }
  if (IsUnKnownShape(range_max_vector)) {
    y_range.push_back(std::pair<int64_t, int64_t>(range_min, -1));
  } else {
    int64_t range_max = 1;
    for (size_t i = 0; i < range_max_vector.size(); i++) {
      range_max = range_max * range_max_vector[i];
    }
    y_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  }
  y_desc->SetShape(GeShape(y_vector));
  y_desc->SetShapeRange(y_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Flatten, FlattenInferShape);
// ----------------Flatten Op End-------------------

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
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_shape = input_desc->MutableShape().GetDims();
  auto input_dtype = input_desc->GetDataType();
  auto output_desc = op_info->MutableOutputDesc("y");

  // set output dtype as the same with input x
  output_desc->SetDataType(input_dtype);

  if (IsUnknownRankShape(input_shape)) {
    // UnknownRankShape, set shape is -1, -1, -1....
    input_shape.clear();
    for (size_t i = 0; i < perm_list.size(); ++i) {
      input_shape.push_back(-1);
    }
  }

  // verify perm_list
  int64_t perm_value = 0;
  for (size_t i = 0; i < perm_list.size(); ++i) {
    perm_value = perm_list[i] < 0 ? perm_list[i] + input_shape.size() : perm_list[i];
    if (perm_value >= input_shape.size()) {
      std::string err_msg = GetAttrValueErrMsg("perm", ConcatString(perm_value),
                                               ConcatString("less than input shape size[",
                                               input_shape.size(), "]"));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }

  // infer the shape
  vector<int64_t> out_vec;
  for (size_t i = 0; i < perm_list.size(); ++i) {
    out_vec.push_back(input_shape[perm_list[i]]);
  }
  output_desc->SetShape(GeShape(out_vec));

  // infer the range, when need
  if (IsUnknownVec(out_vec)) {
    output_desc->SetOriginShape(GeShape(out_vec));
    std::vector<std::pair<int64_t, int64_t>> input_range;
    std::vector<std::pair<int64_t, int64_t>> output_range;
    input_desc->GetShapeRange(input_range);
    MakeUpShapeRange(input_shape, input_range);
    for (size_t i = 0; i < perm_list.size(); ++i) {
      output_range.push_back(input_range[perm_list[i]]);
    }
    output_desc->SetShapeRange(output_range);
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(TransposeInferShape) {
  const vector<string> depend_names = {"perm"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  auto node = NodeUtils::GetNodeFromOperator(op);

  bool perm_done = false;
  GeTensorPtr perm_tensor = nullptr;
  std::vector<int64_t> perm_list;
  if (GRAPH_SUCCESS == NodeUtils::GetInputConstData(node, "perm", perm_tensor)) {
    auto const_desc = op_desc->MutableInputDesc("perm");
    auto const_dtype = const_desc->GetDataType();
    if (GetConstValue(op, perm_tensor, const_dtype, perm_list)) {
      perm_done = true;
    } else {
      OP_LOGW(op.GetName().c_str(), "Get Const perm value failed ");
    }
  }

  // perm is const node , will do infer use function TransposeCommonInferShape
  if (perm_done) {
    if (GRAPH_SUCCESS != TransposeCommonInferShape(perm_list, op)) {
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }

  // perm is not const node, infer for aicpu
  auto input_desc = op_desc->MutableInputDesc("x");
  auto input_shape = input_desc->MutableShape().GetDims();
  auto input_dtype = input_desc->GetDataType();
  auto output_desc = op_desc->MutableOutputDesc("y");

  // set output dtype as the same with input x
  output_desc->SetDataType(input_dtype);

  if (IsUnknownRankShape(input_shape)) {
    auto perm_desc = op_desc->MutableInputDesc("perm");
    auto perm_shape = perm_desc->MutableShape().GetDims();
    if (IsUnknown(perm_shape)) {
      // set output is -2 UnknownRank
      OP_LOGW(op.GetName().c_str(), "the output will be set to -2");
      output_desc->SetShape(GeShape(input_shape));
      output_desc->SetOriginShape(GeShape(input_shape));
      return GRAPH_SUCCESS;
    }

    // pert is not dynamic shape, will update the input shape
    if (perm_shape.empty()) {
      perm_shape.push_back(1);
    }
    input_shape.clear();
    for (auto i = 0; i < perm_shape[0]; ++i) {
      input_shape.push_back(-1);
    }
  }

  // begin to infer shape and range
  std::vector<std::pair<int64_t, int64_t>> input_range;
  std::vector<std::pair<int64_t, int64_t>> output_range;
  vector<int64_t> out_vec;
  input_desc->GetShapeRange(input_range);
  MakeUpShapeRange(input_shape, input_range);

  int64_t range_first = input_range[0].first;
  int64_t range_second = input_range[0].second;

  for (size_t i = 0; i < input_range.size(); ++i) {
    // all range is the same and get the shape range
    range_first = std::min(range_first, input_range[i].first);
    range_second = (range_second == -1 || input_range[i].second == -1) ?
                   -1 :
                   std::max(range_second, input_range[i].second);
  }

  for (size_t i = 0; i < input_range.size(); ++i) {
    out_vec.push_back(-1);
    output_range.push_back(std::pair<int64_t, int64_t>(range_first, range_second));
  }
  output_desc->SetShape(GeShape(out_vec));
  output_desc->SetOriginShape(GeShape(out_vec));
  output_desc->SetShapeRange(output_range);

  return GRAPH_SUCCESS;
}

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
IMPLEMT_COMMON_INFERFUNC(TransposeDInferShape) {
  std::vector<int64_t> perm_list;
  if (ge::GRAPH_SUCCESS != op.GetAttr("perm", perm_list)) {
    std::string err_msg = GetInputInvalidErrMsg("perm");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (GRAPH_SUCCESS != TransposeCommonInferShape(perm_list, op)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(TransposeD, TransposeDInferShape);

IMPLEMT_INFERFORMAT_FUNC(TransposeD, TransposeDInferFormat) {
  bool recovery_flag = false;  // if not scene that transformation between NCHW or NHWC, keep ND
  std::vector<int64_t> perm_list;
  if (ge::GRAPH_SUCCESS != op.GetAttr("perm", perm_list)) {
    std::string err_msg = GetInputInvalidErrMsg("perm");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
IMPLEMT_COMMON_INFERFUNC(TransDataInferShape) {
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
  return GRAPH_SUCCESS;
}

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
  // verify input shape size
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_dims = input_desc->MutableShape().GetDims();
  if (!IsUnknownRankShape(input_dims) && (input_dims.size() < 4)) {
    std::string err_msg = GetAttrValueErrMsg("input_dims", std::to_string(input_dims.size()), ConcatString(">=4"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  // verify block size
  int64_t block_size;
  if (op.GetAttr("block_size", block_size) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("block_size");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (block_size < 2) {
     std::string err_msg = GetAttrValueErrMsg("block_size", std::to_string(block_size), ConcatString("=<2"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  // verify data_format
  std::string data_format;
  if (op.GetAttr("data_format", data_format) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("data_format");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (data_format != "NHWC" && data_format != "NCHW" && data_format != "NC1HWC0") {
    string expected_format_list = ConcatString("NHWC, NCHW, NC1HWC0");
    std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(DepthToSpaceInfer) {
  auto node = NodeUtils::GetNodeFromOperator(op);
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_dims = input_desc->MutableShape().GetDims();
  auto input_dtype = input_desc->GetDataType();
  auto input_format = input_desc->GetFormat();

  auto output_desc = op_info->MutableOutputDesc("y");
  output_desc->SetDataType(input_dtype);

  // get attr block_size
  int64_t block_size;
  if (GRAPH_SUCCESS != op.GetAttr("block_size", block_size)) {
    std::string err_msg = GetInputInvalidErrMsg("block_size");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // not dynamic case, only set shape
  if (!IsUnknown(input_dims)) {
    std::vector<int64_t> output_dims;
    output_dims.push_back(input_dims[0]);
    if (input_format == FORMAT_NCHW) {
      output_dims.push_back(input_dims[1] / block_size / block_size);
      output_dims.push_back(input_dims[2] * block_size);
      output_dims.push_back(input_dims[3] * block_size);
    } else { // without NCHW all other format set as NHWC
      output_dims.push_back(input_dims[1] * block_size);
      output_dims.push_back(input_dims[2] * block_size);
      output_dims.push_back(input_dims[3] / block_size / block_size);
    }
    output_desc->SetShape(GeShape(output_dims));
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -2, output is -2
  if (IsUnknownRankShape(input_dims)) {
    output_desc->SetShape(GeShape(input_dims));
    OP_LOGW(op.GetName().c_str(), "input shape is UnknownRank, set output is UnknownRank.");
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -1, output is -1
  std::vector<std::pair<int64_t, int64_t>> input_range;
  input_desc->GetShapeRange(input_range);
  MakeUpShapeRange(input_dims, input_range);

  // infer output shape and range
  std::vector<int64_t> output_dims;
  std::vector<std::pair<int64_t, int64_t>> output_range;
  output_dims.push_back(input_dims[0]);
  output_range.push_back(input_range[0]);
  int64_t dim;
  int64_t range_min;
  int64_t range_max;
  if (input_format == FORMAT_NCHW) {
    dim = input_dims[1] == -1 ? -1 : input_dims[1] / block_size / block_size;
    range_min = input_range[1].first / block_size / block_size;
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = input_range[1].second == -1 ? -1 : input_range[1].second / block_size / block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = input_dims[2] == -1 ? -1 : input_dims[2] * block_size;
    range_min = input_range[2].first * block_size;
    range_max = input_range[2].second == -1 ? -1 : input_range[2].second * block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = input_dims[3] == -1 ? -1 : input_dims[3] * block_size;
    range_min = input_range[3].first * block_size;
    range_max = input_range[3].second == -1 ? -1 : input_range[3].second * block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  } else {
    dim = input_dims[1] == -1 ? -1 : input_dims[1] * block_size;
    range_min = input_range[1].first * block_size;
    range_max = input_range[1].second == -1 ? -1 : input_range[1].second * block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = input_dims[2] == -1 ? -1 : input_dims[2] * block_size;
    range_min = input_range[2].first * block_size;
    range_max = input_range[2].second == -1 ? -1 : input_range[2].second * block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = input_dims[3] == -1 ? -1 : input_dims[3] / block_size / block_size;
    range_min = input_range[3].first / block_size / block_size;
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = input_range[3].second == -1 ? -1 : input_range[3].second / block_size / block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  }

  output_desc->SetShape(GeShape(output_dims));
  output_desc->SetShapeRange(output_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(DepthToSpace, DepthToSpaceInfer);
VERIFY_FUNC_REG(DepthToSpace, DepthToSpaceVerify);
// -------------------DepthToSpace END-----------------

// ----------------SpaceToDepth Op Start-------------------
IMPLEMT_VERIFIER(SpaceToDepth, SpaceToDepthVerify) {
  // verify input shape size
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_dims = input_desc->MutableShape().GetDims();
  if (!IsUnknownRankShape(input_dims) && (input_dims.size() < 4)) {
    string excepted_value = ConcatString("greater than or equal to 4.");
    std::string err_msg = GetAttrSizeErrMsg("Input shape", ConcatString(input_dims.size()), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  // verify block size
  int64_t block_size;
  if (op.GetAttr("block_size", block_size) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("block_size");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (block_size < 2) {
    string excepted_value = ConcatString("greater than or equal to 2");
    std::string err_msg = GetAttrValueErrMsg("block_size", ConcatString(block_size), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  // verify data_format
  std::string data_format;
  if (op.GetAttr("data_format", data_format) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("data_format");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (data_format != "NHWC" && data_format != "NCHW" && data_format != "NC1HWC0") {
    string expected_format_list = ConcatString("NHWC, NCHW, NC1HWC0");
    std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SpaceToDepthInferShape) {
  auto node = NodeUtils::GetNodeFromOperator(op);
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_dims = input_desc->MutableShape().GetDims();
  auto input_dtype = input_desc->GetDataType();
  auto input_format = input_desc->GetFormat();

  auto output_desc = op_info->MutableOutputDesc("y");
  output_desc->SetDataType(input_dtype);

  // get attr block_size
  int64_t block_size;
  if (GRAPH_SUCCESS != op.GetAttr("block_size", block_size)) {
    std::string err_msg = GetInputInvalidErrMsg("block_size");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // not dynamic case, only set shape
  if (!IsUnknown(input_dims)) {
    std::vector<int64_t> output_dims;
    output_dims.push_back(input_dims[0]);
    if (input_format == FORMAT_NCHW) {
      output_dims.push_back(input_dims[1] * block_size * block_size);
      output_dims.push_back(input_dims[2] / block_size);
      output_dims.push_back(input_dims[3] / block_size);
    } else { // without NCHW all other format set as NHWC
      output_dims.push_back(input_dims[1] / block_size);
      output_dims.push_back(input_dims[2] / block_size);
      output_dims.push_back(input_dims[3] * block_size * block_size);
    }
    output_desc->SetShape(GeShape(output_dims));
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -2, output is -2
  if (IsUnknownRankShape(input_dims)) {
    output_desc->SetShape(GeShape(input_dims));
    OP_LOGW(op.GetName().c_str(), "input shape is UnknownRank, set output is UnknownRank.");
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -1, output is -1
  std::vector<std::pair<int64_t, int64_t>> input_range;
  input_desc->GetShapeRange(input_range);
  MakeUpShapeRange(input_dims, input_range);

  // infer output shape and range
  std::vector<int64_t> output_dims;
  std::vector<std::pair<int64_t, int64_t>> output_range;
  output_dims.push_back(input_dims[0]);
  output_range.push_back(input_range[0]);
  int64_t dim;
  int64_t range_min;
  int64_t range_max;
  if (input_format == FORMAT_NCHW) {
    dim = input_dims[1] == -1 ? -1 : input_dims[1] * block_size * block_size;
    range_min = input_range[1].first * block_size * block_size;
    range_max = input_range[1].second == -1 ? -1 : input_range[1].second * block_size * block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = input_dims[2] == -1 ? -1 : input_dims[2] / block_size;
    range_min = input_range[2].first / block_size;
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = input_range[2].second == -1 ? -1 : input_range[2].second / block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = input_dims[3] == -1 ? -1 : input_dims[3] / block_size;
    range_min = input_range[3].first / block_size;
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = input_range[3].second == -1 ? -1 : input_range[3].second / block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  } else {
    dim = input_dims[1] == -1 ? -1 : input_dims[1] / block_size;
    range_min = input_range[1].first / block_size;
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = input_range[1].second == -1 ? -1 : input_range[1].second / block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = input_dims[2] == -1 ? -1 : input_dims[2] / block_size;
    range_min = input_range[2].first / block_size;
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = input_range[2].second == -1 ? -1 : input_range[2].second / block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = input_dims[3] == -1 ? -1 : input_dims[3] * block_size * block_size;
    range_min = input_range[3].first * block_size * block_size;
    range_max = input_range[3].second == -1 ? -1 : input_range[3].second * block_size * block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  }

  output_desc->SetShape(GeShape(output_dims));
  output_desc->SetShapeRange(output_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SpaceToDepth, SpaceToDepthInferShape);
VERIFY_FUNC_REG(SpaceToDepth, SpaceToDepthVerify);
// ----------------SpaceToDepth Op End-------------------

// ----------------SpaceToBatch Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(SpaceToBatchInferShape) {
  const vector<string> depend_names = {"paddings"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  auto node = NodeUtils::GetNodeFromOperator(op);
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_dims = input_desc->MutableShape().GetDims();
  auto input_dtype = input_desc->GetDataType();
  auto input_format = input_desc->GetFormat();

  auto output_desc = op_info->MutableOutputDesc("y");
  output_desc->SetDataType(input_dtype);

  // get attr block_size
  int64_t block_size;
  if (op.GetAttr("block_size", block_size) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "block_size");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr block_size failed!");
    return GRAPH_FAILED;
  }

  // get const node padding
  bool padding_done = false;
  std::vector<int64_t> paddings;
  GeTensorPtr paddings_tensor = nullptr;
  if (GRAPH_SUCCESS == NodeUtils::GetInputConstData(node, "paddings", paddings_tensor)) {
    auto const_desc = op_info->MutableInputDesc("paddings");
    auto const_dtype = const_desc->GetDataType();
    if (GetConstValue(op, paddings_tensor, const_dtype, paddings)) {
      padding_done = true;
    } else {
      OP_LOGW(op.GetName().c_str(), "Get Const paddings value failed.");
    }
  }

  // if paddings are const node, verfify const sizes
  if (padding_done && paddings.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "The paddings size must be equal to 4, but got %d.", paddings.size());
    return GRAPH_FAILED;
  }

  // not dynamic case, only set shape
  if (!IsUnknown(input_dims) && padding_done) {
    std::vector<int64_t> output_dims;
    output_dims.push_back(input_dims[0] * block_size * block_size);
    if (input_format == FORMAT_NCHW) {
      output_dims.push_back(input_dims[1]);
      output_dims.push_back((input_dims[2] + paddings[0] + paddings[1]) / block_size);
      output_dims.push_back((input_dims[3] + paddings[2] + paddings[3]) / block_size);
    } else {
      output_dims.push_back((input_dims[1] + paddings[0] + paddings[1]) / block_size);
      output_dims.push_back((input_dims[2] + paddings[2] + paddings[3]) / block_size);
      output_dims.push_back(input_dims[3]);
    }
    output_desc->SetShape(GeShape(output_dims));
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -2, output is -2
  if (IsUnknownRankShape(input_dims)) {
    output_desc->SetShape(GeShape(input_dims));
    OP_LOGW(op.GetName().c_str(), "input shape is UnknownRank, set output is UnknownRank.");
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -1, output is -1
  std::vector<std::pair<int64_t, int64_t>> input_range;
  input_desc->GetShapeRange(input_range);
  MakeUpShapeRange(input_dims, input_range);

  auto paddings_desc = op_info->MutableInputDesc("paddings");
  std::vector<int64_t> paddings_dims = paddings_desc->MutableShape().GetDims();
  std::vector<std::pair<int64_t, int64_t>> paddings_range;
  if (IsUnknownRankShape(paddings_dims)) {
    paddings_dims = {-1, -1};
  } else {
    paddings_desc->GetShapeRange(paddings_range);
  }
  MakeUpShapeRange(paddings_dims, paddings_range);

  // infer output shape and range
  std::vector<int64_t> output_dims;
  std::vector<std::pair<int64_t, int64_t>> output_range;
  int64_t dim = input_dims[0] == -1 ? -1 : input_dims[0] * block_size * block_size;
  int64_t range_min = input_range[0].first * block_size * block_size;
  int64_t range_max = input_range[0].second == -1 ? -1 : input_range[0].second * block_size * block_size;
  output_dims.push_back(dim);
  output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  if (input_format == FORMAT_NCHW) {
    output_dims.push_back(input_dims[1]);
    output_range.push_back(input_range[1]);
    dim = (input_dims[2] == -1 || !padding_done) ? -1 : (input_dims[2] + paddings[0] + paddings[1]) / block_size;
    range_min = !padding_done ? input_range[2].first / block_size
                              : (input_range[2].first + paddings[0] + paddings[1]) / block_size;
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = (input_range[2].second == -1 || !padding_done)
                    ? -1
                    : (input_range[2].second + paddings[0] + paddings[1]) / block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = (input_dims[3] == -1 || !padding_done) ? -1 : (input_dims[3] + paddings[2] + paddings[3]) / block_size;
    range_min = !padding_done ? input_range[3].first / block_size
                              : (input_range[3].first + paddings[2] + paddings[3]) / block_size;
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = (input_range[3].second == -1 || !padding_done)
                    ? -1
                    : (input_range[3].second + paddings[2] + paddings[3]) / block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  } else {
    dim = (input_dims[1] == -1 || !padding_done) ? -1 : (input_dims[1] + paddings[0] + paddings[1]) / block_size;
    range_min = !padding_done ? input_range[1].first / block_size
                              : (input_range[1].first + paddings[0] + paddings[1]) / block_size;
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = (input_range[1].second == -1 || !padding_done)
                    ? -1
                    : (input_range[1].second + paddings[0] + paddings[1]) / block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = (input_dims[2] == -1 || !padding_done) ? -1 : (input_dims[2] + paddings[2] + paddings[3]) / block_size;
    range_min = !padding_done ? input_range[2].first / block_size
                              : (input_range[2].first + paddings[2] + paddings[3]) / block_size;
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = (input_range[2].second == -1 || !padding_done)
                    ? -1
                    : (input_range[2].second + paddings[2] + paddings[3]) / block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    output_dims.push_back(input_dims[3]);
    output_range.push_back(input_range[3]);
  }

  output_desc->SetShape(GeShape(output_dims));
  output_desc->SetShapeRange(output_range);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SpaceToBatch, SpaceToBatchVerify) {
  // check input shape size
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_dims = input_desc->MutableShape().GetDims();
  if (!IsUnknownRankShape(input_dims) && (input_dims.size() < 4)) {
    OpsAttrValueErrReport(op.GetName(), "input shape size", "greater than or equal to 4",
                          ConcatString(input_dims.size()));
    OP_LOGE(op.GetName().c_str(), "Input shape size must be greater than or equal to 4, but got %d.", input_dims.size());
    return GRAPH_FAILED;
  }
  // check block size
  int64_t block_size;
  if (op.GetAttr("block_size", block_size) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "block_size");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr block_size failed!");
    return GRAPH_FAILED;
  }
  if (block_size < 2) {
    OpsAttrValueErrReport(op.GetName(), "block_size", "greater than or equal to 2", ConcatString(block_size));
    OP_LOGE(op.GetName().c_str(), "The block_size must be greater than or equal to 2, but got %d.", block_size);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SpaceToBatch, SpaceToBatchInferShape);
VERIFY_FUNC_REG(SpaceToBatch, SpaceToBatchVerify);
// ----------------SpaceToBatch Op End-------------------

// ----------------SpaceToBatchD Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(SpaceToBatchDInferShape) {
  auto node = NodeUtils::GetNodeFromOperator(op);
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_dims = input_desc->MutableShape().GetDims();
  auto input_dtype = input_desc->GetDataType();
  auto input_format = input_desc->GetFormat();

  auto output_desc = op_info->MutableOutputDesc("y");
  output_desc->SetDataType(input_dtype);

  // get attr block_size
  int64_t block_size;
  if (op.GetAttr("block_size", block_size) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("block_size");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get attr paddings
  std::vector<int64_t> paddings;
  if (op.GetAttr("paddings", paddings) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("paddings");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // not dynamic case, only set shape
  if (!IsUnknown(input_dims)) {
    std::vector<int64_t> output_dims;
    output_dims.push_back(input_dims[0] * block_size * block_size);
    if (input_format == FORMAT_NCHW) {
      output_dims.push_back(input_dims[1]);
      output_dims.push_back((input_dims[2] + paddings[0] + paddings[1]) / block_size);
      output_dims.push_back((input_dims[3] + paddings[2] + paddings[3]) / block_size);
    } else {
      output_dims.push_back((input_dims[1] + paddings[0] + paddings[1]) / block_size);
      output_dims.push_back((input_dims[2] + paddings[2] + paddings[3]) / block_size);
      output_dims.push_back(input_dims[3]);
    }
    output_desc->SetShape(GeShape(output_dims));
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -2, output is -2
  if (IsUnknownRankShape(input_dims)) {
    output_desc->SetShape(GeShape(input_dims));
    OP_LOGW(op.GetName().c_str(), "input shape is UnknownRank, set output is UnknownRank.");
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -1, output is -1
  std::vector<std::pair<int64_t, int64_t>> input_range;
  input_desc->GetShapeRange(input_range);
  MakeUpShapeRange(input_dims, input_range);

  // infer output shape and range
  std::vector<int64_t> output_dims;
  std::vector<std::pair<int64_t, int64_t>> output_range;
  int64_t dim = input_dims[0] == -1 ? -1 : input_dims[0] * block_size * block_size;
  int64_t range_min = input_range[0].first * block_size * block_size;
  int64_t range_max = input_range[0].second == -1 ? -1 : input_range[0].second * block_size * block_size;
  output_dims.push_back(dim);
  output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  if (input_format == FORMAT_NCHW) {
    output_dims.push_back(input_dims[1]);
    output_range.push_back(input_range[1]);
    dim = input_dims[2] == -1 ? -1 : (input_dims[2] + paddings[0] + paddings[1]) / block_size;
    range_min = (input_range[2].first + paddings[0] + paddings[1]) / block_size;
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = input_range[2].second == -1 ? -1 : (input_range[2].second + paddings[0] + paddings[1]) / block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = input_dims[3] == -1 ? -1 : (input_dims[3] + paddings[2] + paddings[3]) / block_size;
    range_min = (input_range[3].first + paddings[2] + paddings[3]) / block_size;
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = input_range[3].second == -1 ? -1 : (input_range[3].second + paddings[2] + paddings[3]) / block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  } else {
    dim = input_dims[1] == -1 ? -1 : (input_dims[1] + paddings[0] + paddings[1]) / block_size;
    range_min = (input_range[1].first + paddings[0] + paddings[1]) / block_size;
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = input_range[1].second == -1 ? -1 : (input_range[1].second + paddings[0] + paddings[1]) / block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = input_dims[2] == -1 ? -1 : (input_dims[2] + paddings[2] + paddings[3]) / block_size;
    range_min = (input_range[2].first + paddings[2] + paddings[3]) / block_size;
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = input_range[2].second == -1 ? -1 : (input_range[2].second + paddings[2] + paddings[3]) / block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    output_dims.push_back(input_dims[3]);
    output_range.push_back(input_range[3]);
  }

  output_desc->SetShape(GeShape(output_dims));
  output_desc->SetShapeRange(output_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SpaceToBatchD, SpaceToBatchDInferShape);
// ----------------SpaceToBatchD Op End-------------------

// ----------------BatchToSpace Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(BatchToSpaceInferShape) {
  const vector<string> depend_names = {"crops"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  auto node = NodeUtils::GetNodeFromOperator(op);
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_dims = input_desc->MutableShape().GetDims();
  auto input_dtype = input_desc->GetDataType();
  auto input_format = input_desc->GetFormat();

  auto output_desc = op_info->MutableOutputDesc("y");
  output_desc->SetDataType(input_dtype);

  // get attr block_size
  int64_t block_size;
  if (op.GetAttr("block_size", block_size) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
                                       string("get attr[block_size] failed"));
    return GRAPH_FAILED;
  }

  // get const node crops
  bool crops_done = false;
  std::vector<int64_t> crops;
  GeTensorPtr crops_tensor = nullptr;
  if (GRAPH_SUCCESS == NodeUtils::GetInputConstData(node, "crops", crops_tensor)) {
    auto const_desc = op_info->MutableInputDesc("crops");
    auto const_dtype = const_desc->GetDataType();
    if (GetConstValue(op, crops_tensor, const_dtype, crops)) {
      crops_done = true;
    } else {
      OP_LOGW(op.GetName().c_str(), "Get Const crops value failed.");
    }
  }

  // if crops are const node, verfify const sizes
  if (crops_done && crops.size() != 4) {
    std::string err_msg = ConcatString(
        "input[crops] data size[", crops.size(),"] not equal 4");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // not dynamic case, only set shape
  if (!IsUnknown(input_dims) && crops_done) {
    std::vector<int64_t> output_dims;
    output_dims.push_back(input_dims[0] / block_size / block_size);
    if (input_format == FORMAT_NCHW) {
      output_dims.push_back(input_dims[1]);
      output_dims.push_back(input_dims[2] * block_size - crops[0] - crops[1]);
      output_dims.push_back(input_dims[3] * block_size - crops[2] - crops[3]);
    } else {
      output_dims.push_back(input_dims[1] * block_size - crops[0] - crops[1]);
      output_dims.push_back(input_dims[2] * block_size - crops[2] - crops[3]);
      output_dims.push_back(input_dims[3]);
    }
    output_desc->SetShape(GeShape(output_dims));
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -2, output is -2
  if (IsUnknownRankShape(input_dims)) {
    output_desc->SetShape(GeShape(input_dims));
    OP_LOGW(op.GetName().c_str(), "input shape is UnknownRank, set output is UnknownRank.");
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -1, output is -1
  std::vector<std::pair<int64_t, int64_t>> input_range;
  input_desc->GetShapeRange(input_range);
  MakeUpShapeRange(input_dims, input_range);

  auto crops_desc = op_info->MutableInputDesc("crops");
  std::vector<int64_t> crops_dims = crops_desc->MutableShape().GetDims();
  std::vector<std::pair<int64_t, int64_t>> crops_range;
  if (IsUnknownRankShape(crops_dims)) {
    crops_dims = {-1, -1};
  } else {
    crops_desc->GetShapeRange(crops_range);
  }
  MakeUpShapeRange(crops_dims, crops_range);

  // infer output shape and range
  std::vector<int64_t> output_dims;
  std::vector<std::pair<int64_t, int64_t>> output_range;
  int64_t dim = input_dims[0] == -1 ? -1 : input_dims[0] / block_size / block_size;
  int64_t range_min = std::max(int64_t(input_range[0].first / block_size / block_size), int64_t(1));
  int64_t range_max = input_range[0].second == -1 ? -1 : input_range[0].second / block_size / block_size;
  output_dims.push_back(dim);
  output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  if (input_format == FORMAT_NCHW) {
    output_dims.push_back(input_dims[1]);
    output_range.push_back(input_range[1]);
    dim = (input_dims[2] == -1 || !crops_done) ? -1 : input_dims[2] * block_size - crops[0] - crops[1];
    range_min = !crops_done ? 1 : input_range[2].first * block_size - crops[0] - crops[1];
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max =
        (input_range[2].second == -1 || !crops_done) ? -1 : input_range[2].second * block_size - crops[0] - crops[1];
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = (input_dims[3] == -1 || !crops_done) ? -1 : input_dims[3] * block_size - crops[2] - crops[3];
    range_min = !crops_done ? 1 : input_range[3].first * block_size - crops[2] - crops[3];
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max =
        (input_range[3].second == -1 || !crops_done) ? -1 : input_range[3].second * block_size - crops[2] - crops[3];
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  } else {
    dim = (input_dims[1] == -1 || !crops_done) ? -1 : input_dims[1] * block_size - crops[0] - crops[1];
    range_min = !crops_done ? 1 : input_range[1].first * block_size - crops[0] - crops[1];
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max =
        (input_range[1].second == -1 || !crops_done) ? -1 : input_range[1].second * block_size - crops[0] - crops[1];
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = (input_dims[2] == -1 || !crops_done) ? -1 : input_dims[2] * block_size - crops[2] - crops[3];
    range_min = !crops_done ? 1 : input_range[2].first * block_size - crops[2] - crops[3];
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max =
        (input_range[2].second == -1 || !crops_done) ? -1 : input_range[2].second * block_size - crops[2] - crops[3];
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    output_dims.push_back(input_dims[3]);
    output_range.push_back(input_range[3]);
  }

  output_desc->SetShape(GeShape(output_dims));
  output_desc->SetShapeRange(output_range);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(BatchToSpace, BatchToSpaceVerify) {
  // check input shape size
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_dims = input_desc->MutableShape().GetDims();
  if (!IsUnknownRankShape(input_dims) && (input_dims.size() < 4)) {
    OpsAttrValueErrReport(op.GetName(), "input shape size", "greater than or equal to 4",
                          ConcatString(input_dims.size()));
    OP_LOGE(op.GetName().c_str(), "Input shape size must be greater than or equal to 4, but got %d.", input_dims.size());
    return GRAPH_FAILED;
  }
  // check block size
  int64_t block_size;
  if (op.GetAttr("block_size", block_size) != GRAPH_SUCCESS) {
    OpsGetAttrErrReport(op.GetName(), "block_size");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr block_size failed!");
    return GRAPH_FAILED;
  }
  if (block_size < 2) {
    OpsAttrValueErrReport(op.GetName(), "block_size", "greater than or equal to 2", ConcatString(block_size));
    OP_LOGE(op.GetName().c_str(), "The block_size must be greater than or equal to 2, but got %d.", block_size);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BatchToSpace, BatchToSpaceInferShape);
VERIFY_FUNC_REG(BatchToSpace, BatchToSpaceVerify);
// ----------------BatchToSpace Op End-------------------

// ----------------BatchToSpaceD Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(BatchToSpaceDInferShape) {
  auto node = NodeUtils::GetNodeFromOperator(op);
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_dims = input_desc->MutableShape().GetDims();
  auto input_dtype = input_desc->GetDataType();
  auto input_format = input_desc->GetFormat();

  auto output_desc = op_info->MutableOutputDesc("y");
  output_desc->SetDataType(input_dtype);

  // get attr block_size
  int64_t block_size;
  if (op.GetAttr("block_size", block_size) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("block_size");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // get attr crops
  std::vector<int64_t> crops;
  if (op.GetAttr("crops", crops) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("crops");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  // not dynamic case, only set shape
  if (!IsUnknown(input_dims)) {
    std::vector<int64_t> output_dims;
    output_dims.push_back(input_dims[0] / block_size / block_size);
    if (input_format == FORMAT_NCHW) {
      output_dims.push_back(input_dims[1]);
      output_dims.push_back(input_dims[2] * block_size - crops[0] - crops[1]);
      output_dims.push_back(input_dims[3] * block_size - crops[2] - crops[3]);
    } else {
      output_dims.push_back(input_dims[1] * block_size - crops[0] - crops[1]);
      output_dims.push_back(input_dims[2] * block_size - crops[2] - crops[3]);
      output_dims.push_back(input_dims[3]);
    }
    output_desc->SetShape(GeShape(output_dims));
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -2, output is -2
  if (IsUnknownRankShape(input_dims)) {
    output_desc->SetShape(GeShape(input_dims));
    OP_LOGW(op.GetName().c_str(), "input shape is UnknownRank, set output is UnknownRank.");
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -1, output is -1
  std::vector<std::pair<int64_t, int64_t>> input_range;
  input_desc->GetShapeRange(input_range);
  MakeUpShapeRange(input_dims, input_range);

  // infer output shape and range
  std::vector<int64_t> output_dims;
  std::vector<std::pair<int64_t, int64_t>> output_range;
  int64_t dim = input_dims[0] == -1 ? -1 : input_dims[0] / block_size / block_size;
  int64_t range_min = std::max(int64_t(input_range[0].first / block_size / block_size), int64_t(1));
  int64_t range_max = input_range[0].second == -1 ? -1 : input_range[0].second / block_size / block_size;
  output_dims.push_back(dim);
  output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  if (input_format == FORMAT_NCHW) {
    output_dims.push_back(input_dims[1]);
    output_range.push_back(input_range[1]);
    dim = input_dims[2] == -1 ? -1 : input_dims[2] * block_size - crops[0] - crops[1];
    range_min = input_range[2].first * block_size - crops[0] - crops[1];
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = input_range[2].second == -1 ? -1 : input_range[2].second * block_size - crops[0] - crops[1];
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = input_dims[3] == -1 ? -1 : input_dims[3] * block_size - crops[2] - crops[3];
    range_min = input_range[3].first * block_size - crops[2] - crops[3];
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = input_range[3].second == -1 ? -1 : input_range[3].second * block_size - crops[2] - crops[3];
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  } else {
    dim = input_dims[1] == -1 ? -1 : input_dims[1] * block_size - crops[0] - crops[1];
    range_min = input_range[1].first * block_size - crops[0] - crops[1];
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = input_range[1].second == -1 ? -1 : input_range[1].second * block_size - crops[0] - crops[1];
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = input_dims[2] == -1 ? -1 : input_dims[2] * block_size - crops[2] - crops[3];
    range_min = input_range[2].first * block_size - crops[2] - crops[3];
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = input_range[2].second == -1 ? -1 : input_range[2].second * block_size - crops[2] - crops[3];
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    output_dims.push_back(input_dims[3]);
    output_range.push_back(input_range[3]);
  }

  output_desc->SetShape(GeShape(output_dims));
  output_desc->SetShapeRange(output_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BatchToSpaceD, BatchToSpaceDInferShape);
// ----------------BatchToSpaceD Op End-------------------

// ----------------Unapck Op-------------------
IMPLEMT_COMMON_INFERFUNC(UnpackInferShape) {
  OP_LOGI(op.GetName().c_str(), "UnpackInferShape function start!");
  std::vector<std::pair<int64_t, int64_t>> x_range;
  std::vector<std::pair<int64_t, int64_t>> out_range;
  std::vector<int64_t> output_vec;
  std::vector<GeTensorDescPtr> output_ptrs;

  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto tensordesc_x = op_info->MutableInputDesc("x");
  std::vector<int64_t> shape_x = tensordesc_x->MutableShape().GetDims();
  tensordesc_x->GetShapeRange(x_range);
  DataType input_dtype = tensordesc_x->GetDataType();

  // check value of aixs and num
  int64_t axis{0};
  if (op.GetAttr("axis", axis) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("axis");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  int64_t num{0};
  if (op.GetAttr("num", num) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("num");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  output_ptrs.resize(num);
  if (num != op_info->GetAllOutputsDescSize()) {
    std::string err_msg = GetAttrValueErrMsg("num", ConcatString(num), ConcatString(op_info->GetAllOutputsDescSize()));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  for (int i = 0; i < num; i++) {
    output_ptrs[i] = op_info->MutableOutputDesc(i);
  }

  if (!IsUnknownRankShape(shape_x)) {
    int64_t x_dims = static_cast<int64_t>(shape_x.size());
    int64_t real_axis = (axis >= 0) ? axis : axis + x_dims;
    if (real_axis < 0 || real_axis >= x_dims) {
      std::string err_msg = OtherErrMsg(ConcatString("Axis exceeding the prescribed range. Axis is ", real_axis,
                                                     " and x_shape's size is ", shape_x.size()));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    // infer output shape
    for (int64_t i = 0; i < x_dims; i++) {
      if (i != real_axis) {
        output_vec.push_back(shape_x[i]);
        if (static_cast<int64_t>(x_range.size()) == x_dims) {
          out_range.push_back(x_range[i]);
        }
      }
    }
  } else {
    output_vec = UNKNOWN_SHAPE;
  }
  for (int64_t i = 0; i < num; i++) {
    output_ptrs[i]->SetShape(GeShape(output_vec));
    output_ptrs[i]->SetShapeRange(out_range);
    output_ptrs[i]->SetDataType(input_dtype);
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
  auto x_format = desc_in_ptr->GetOriginFormat();
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
  if (stride_h == 0 || stride_w == 0) {
    OP_LOGE(op.GetName().c_str(), "The stride_h or stride_w should not 0");
    return GRAPH_FAILED;
  }
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
  if (x_format == FORMAT_NCHW) {
    out_dim = {in_n, out_c, out_h, out_w};
  }

  desc_out_ptr->SetShape(ge::GeShape(out_dim));
  desc_out_ptr->SetDataType(dtype);
  return GRAPH_SUCCESS;
}

static void InferHExtractImagePatches(int64_t kernel, int64_t dilation, int64_t stride, int64_t origin_input,
                                      const vector<int64_t>& output_slice, vector<int64_t>& input_slice) {
  // output_slice has at least 2 elements
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
  auto images_format = tensor_in_ptr->GetOriginFormat();
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
    OP_LOGE(op.GetName().c_str(), "No data slice, not need infer input");
    return GRAPH_FAILED;
  }

  bool need_infer = false;
  bool have_slice = false;
  for (unsigned idx = 0; idx < y_data_slice.size(); idx++) {
    if (y_data_slice[idx].size() > 1) {
      have_slice = true;
      if (idx == 2) {
        need_infer = true;
        vector<int64_t> slice_data_h;
        InferHExtractImagePatches(ksize_h, dilation_h, stride_h, images_h, y_data_slice[idx], slice_data_h);
        OP_LOGD(op.GetName().c_str(),
                "ExtractImagePatches h axis slice ori_scope is [%lld, %lld], calced output scope is [%lld, %lld]",
                slice_data_h[0], slice_data_h[1], y_data_slice[idx][0], y_data_slice[idx][1]);
        x_data_slice[idx] = slice_data_h;
      }
    }
  }
  if (!have_slice) {
    OP_LOGE(op.GetName().c_str(), "The op dose not have slice.");
    return GRAPH_FAILED;
  }
  if (!need_infer) {
    OP_LOGI(op.GetName().c_str(), "The op dose not have overlap dim.");
    return NO_OVERLAP_DIM;
  }
  if (!AttrUtils::SetListListInt(tensor_in_ptr, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
    OP_LOGE(op.GetName().c_str(), "The op SetListListInt failed");
    return GRAPH_FAILED;
  }
  OP_LOGI(op.GetName().c_str(), "Calc ExtractImagePatches InferDataSlice end!");
  return GRAPH_SUCCESS;
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
    OP_LOGE(op_name.c_str(), "The %s dose not have enough elements(%llu)!", attr_name.c_str(), list.size());
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
  if (op_desc == nullptr) {
    OP_LOGE(op.GetName().c_str(), "GetOpDescFromOperator return nullptr!");
    return GRAPH_FAILED;
  }
  auto desc_in_ptr = op_desc->MutableInputDesc("x");
  if (desc_in_ptr == nullptr) {
    OP_LOGE(op.GetName().c_str(), "MutableInputDesc return nullptr!");
    return GRAPH_FAILED;
  }
  auto shape_in = desc_in_ptr->GetShape();
  auto dtype = desc_in_ptr->GetDataType();
  auto x_format = desc_in_ptr->GetOriginFormat();
  if (x_format != FORMAT_NDHWC && x_format != FORMAT_NCDHW) {
    OP_LOGE(op.GetName().c_str(), "Attr x_format only support NDHWC or NCDHW");
    return GRAPH_FAILED;
  }

  std::map<char, int> idx_map{{'N', 0}, {'D', 1}, {'H', 2}, {'W', 3}, {'C', 4}};
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
  if (stride_d == 0 || stride_h == 0 || stride_w == 0) {
    OP_LOGE(op.GetName().c_str(), "The stride_d or stride_h or stride_w should not 0");
    return GRAPH_FAILED;
  }

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
  if (des_out_ptr == nullptr) {
    OP_LOGE(op.GetName().c_str(), "MutableOutputDesc return nullptr!");
    return GRAPH_FAILED;
  }
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
  if (op_desc == nullptr) {
    OP_LOGE(op.GetName().c_str(), "GetOpDescFromOperator return nullptr!");
    return GRAPH_FAILED;
  }
  GeTensorDescPtr tensor_in_ptr = op_desc->MutableInputDesc("x");
  if (tensor_in_ptr == nullptr) {
    OP_LOGE(op.GetName().c_str(), "MutableInputDesc return nullptr!");
    return GRAPH_FAILED;
  }
  GeTensorDescPtr tensor_out_ptr = op_desc->MutableOutputDesc("y");
  auto shape_in = tensor_in_ptr->GetShape();
  auto x_format = tensor_in_ptr->GetOriginFormat();
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
    OP_LOGE(op.GetName().c_str(), "No data slice, not need infer input");
    return GRAPH_FAILED;
  }

  bool need_infer = false;
  bool have_slice = false;
  for (unsigned idx = 0; idx < y_data_slice.size(); idx++) {
    if (y_data_slice[idx].size() > 1) {
      have_slice = true;
      if (idx == 1) {
        need_infer = true;
        vector<int64_t> slice_data_d;
        InferHDExtractVolumePatches(filter_d, stride_d, input_d, y_data_slice[idx], slice_data_d);
        OP_LOGD(op.GetName().c_str(),
                "ExtractVolumePatches d axis slice ori_scope is [%lld, %lld], calced output scope is [%lld, %lld]",
                slice_data_d[0], slice_data_d[1], y_data_slice[idx][0], y_data_slice[idx][1]);
        x_data_slice[idx] = slice_data_d;
      } else if (idx == 3) {
        need_infer = true;
        vector<int64_t> slice_data_h;
        InferHDExtractVolumePatches(filter_h, stride_h, input_h, y_data_slice[idx], slice_data_h);
        OP_LOGD(op.GetName().c_str(),
                "ExtractVolumePatches h axis slice ori_scope is [%lld, %lld], calced output scope is [%lld, %lld]",
                slice_data_h[0], slice_data_h[1], y_data_slice[idx][0], y_data_slice[idx][1]);
        x_data_slice[idx] = slice_data_h;
      }
    }
  }

  if (!have_slice) {
    OP_LOGE(op.GetName().c_str(), "The op dose not have slice.");
    return GRAPH_FAILED;
  }
  if (!need_infer) {
    OP_LOGI(op.GetName().c_str(), "The op dose not have overlap dim.");
    return NO_OVERLAP_DIM;
  }
  if (!AttrUtils::SetListListInt(tensor_in_ptr, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
    OP_LOGE(op.GetName().c_str(), "The op SetListListInt failed");
    return GRAPH_FAILED;
  }
  OP_LOGI(op.GetName().c_str(), "Calc ExtractVolumePatches InferDataSlice end!");
  return GRAPH_SUCCESS;
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
    std::string err_msg = GetInputInvalidErrMsg("perm");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> shape_list;
  if (GRAPH_SUCCESS != op.GetAttr("shape", shape_list)) {
    std::string err_msg = GetInputInvalidErrMsg("shape");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  bool transpose_first;
  if (GRAPH_SUCCESS != op.GetAttr("transpose_first", transpose_first)) {
    std::string err_msg = GetInputInvalidErrMsg("transpose_first");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
// -----------------Col2im Op-------------------------
IMPLEMT_VERIFIER(Col2im, Col2imVerify) {
  vector<int32_t> kernel_size;
  if (GRAPH_SUCCESS != op.GetAttr("kernel_size", kernel_size)) {
    OP_LOGE(op.GetName().c_str(), "Attr[kernel_size], get failed.");
    return GRAPH_FAILED;
  }
  if (kernel_size.size() != 2) {
    OP_LOGE(op.GetName().c_str(), "Attr[kernel_size], size of kernel_size must be 2.");
    return GRAPH_FAILED;
  }

  vector<int32_t> dilation;
  if (GRAPH_SUCCESS != op.GetAttr("dilation", dilation)) {
    OP_LOGE(op.GetName().c_str(), "Attr[dilation], get failed.");
    return GRAPH_FAILED;
  }
  if (dilation.size() != 2) {
    OP_LOGE(op.GetName().c_str(), "Attr[dilation], size of dilation must be 2.");
    return GRAPH_FAILED;
  }

  vector<int32_t> padding;
  if (GRAPH_SUCCESS != op.GetAttr("padding", padding)) {
    OP_LOGE(op.GetName().c_str(), "Attr[padding], get failed.");
    return GRAPH_FAILED;
  }
  if (padding.size() != 2) {
    OP_LOGE(op.GetName().c_str(), "Attr[padding], size of padding must be 2.");
    return GRAPH_FAILED;
  }

  vector<int32_t> stride;
  if (GRAPH_SUCCESS != op.GetAttr("stride", stride)) {
    OP_LOGE(op.GetName().c_str(), "Attr[stride], get failed.");
    return GRAPH_FAILED;
  }
  if (stride.size() != 2) {
    OP_LOGE(op.GetName().c_str(), "Attr[stride], size of stride must be 2.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(Col2im, Col2imInferShape) {
  TensorDesc input_desc = op.GetInputDesc("x");
  TensorDesc output_desc = op.GetOutputDesc("y");
  DataType input_dtype = input_desc.GetDataType();
  output_desc.SetDataType(input_dtype);

  Format input_format = input_desc.GetFormat();
  output_desc.SetFormat(input_format);

  Shape input_shape = input_desc.GetShape();  
  vector<int64_t> input_size = input_shape.GetDims();
  
  Tensor output_size_tensor;
  if (op.GetInputConstData("output_size", output_size_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Input[output_size], get failed.");
    return GRAPH_FAILED;
  }

  vector<int64_t> output_size_value;
  if (!GetConstValue(op, output_size_tensor, DT_INT32, output_size_value)) {
    OP_LOGE(op.GetName().c_str(), "Input[output_size], get failed.");
    return GRAPH_FAILED;
  }
  if (output_size_value.size()!=2) {
    OP_LOGE(op.GetName().c_str(), "Input[output_size], size of output_size must be 2.");
    return GRAPH_FAILED;
  }
  int64_t dtype_repeat_num;
  if (input_dtype ==DT_FLOAT) {
    dtype_repeat_num = 64;
  } else {
    dtype_repeat_num = 128;
  }
  int64_t repeat_times_max = 255;
  if ((output_size_value[1] * input_size[4]) / dtype_repeat_num  > repeat_times_max) {
    OP_LOGE(op.GetName().c_str(), "InferShape[Col2im], width of output too large, not support.");
    return GRAPH_FAILED;
  }

  vector<int64_t> output_shape(
    {input_size[0], input_size[1], output_size_value[0], output_size_value[1]}
  );
  
  output_desc.SetShape(Shape(output_shape));

  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Col2im, Col2imInferShape);
VERIFY_FUNC_REG(Col2im, Col2imVerify);
// -----------------Col2im END-------------------------
// ----------------AffineGrid-------------------
IMPLEMT_COMMON_INFERFUNC(AffineGridInferShape) {
    OP_LOGI(op.GetName().c_str(), " AffineGrid inferShape begin!");
    // get theta last dim
    int64_t theta_last_dim = op.GetInputDesc(0).GetShape().GetDim(2);
    // D,H,W size
    int dim_lens = 2;
    // output last dim size
    int64_t output_last_dim = 2;
    if (theta_last_dim == 4) {
        dim_lens = 3;
        output_last_dim = 3;
    }
    // get output infer shape
    vector<int64_t> affine_output_shape;
    int64_t batch_dim = op.GetInputDesc(0).GetShape().GetDim(0);
    affine_output_shape.push_back(batch_dim);
    // get const data value
    Tensor output_size_tensor;
    if (op.GetInputConstData("output_size", output_size_tensor) ==
        GRAPH_SUCCESS) {
        auto size_data =
            reinterpret_cast<const int32_t *>(output_size_tensor.GetData());
        int64_t temp = 1;
        for (int i = 0; i < dim_lens; i++) {
            temp = temp * static_cast<int64_t>(size_data[i + 2]);
        }
        affine_output_shape.push_back(temp);
    } else {
        return GRAPH_FAILED;
    }
    affine_output_shape.push_back(output_last_dim);
    // get input data type
    auto theta_dtype = op.GetInputDesc(0).GetDataType();

    TensorDesc theta_desc = op.GetInputDesc("theta");
    theta_desc.SetFormat(ge::FORMAT_ND);
    theta_desc.SetFormat(ge::FORMAT_ND);
    (void)op.UpdateInputDesc("theta", theta_desc);

    TensorDesc outsize_desc = op.GetInputDesc("output_size");
    outsize_desc.SetFormat(ge::FORMAT_ND);
    outsize_desc.SetFormat(ge::FORMAT_ND);
    (void)op.UpdateInputDesc("output_size", outsize_desc);

    // update output shape and dtype
    TensorDesc output_desc = op.GetOutputDesc("y");
    output_desc.SetShape(ge::Shape(affine_output_shape));
    output_desc.SetDataType(theta_dtype);
    output_desc.SetOriginFormat(ge::FORMAT_ND);
    output_desc.SetFormat(ge::FORMAT_ND);
    (void)op.UpdateOutputDesc("y", output_desc);

    OP_LOGI(op.GetName().c_str(), " AffineGrid inferShape end!");
    return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(AffineGrid, AffineGridVerify) { return GRAPH_SUCCESS; }
COMMON_INFER_FUNC_REG(AffineGrid, AffineGridInferShape);
VERIFY_FUNC_REG(AffineGrid, AffineGridVerify);
// ----------------AffineGrid-------------------
}  // namespace ge
