/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

/*!
 * \file avg_pool_v2_grad.cpp
 * \brief Performs average pooling on the input.
*/
#include "avg_pool_v2_grad.h"
#include <string>
#include <vector>

namespace ge {

bool get_const_value(const Operator& op, const Tensor& const_tensor,
                     const DataType& dtype, std::vector<int64_t>& const_data) {
  size_t size = 0;
  if (dtype == ge::DT_INT32) {
    int32_t* const_data_ptr = (int32_t*)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back((int32_t)((*(const_data_ptr + i))));
    }
  } else if (dtype == ge::DT_INT64) {
    int64_t* const_data_ptr = (int64_t*)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back((int64_t)((*(const_data_ptr + i))));
    }
  } else {
    return false;
  }
  return true;
}

IMPLEMT_VERIFIER(AvgPoolV2Grad, AvgPoolV2GradVerify) {
  Tensor orig_input_shape_tensor;
  if (GRAPH_SUCCESS !=
      op.GetInputConstData("orig_input_shape", orig_input_shape_tensor)) {
    return GRAPH_FAILED;
  }
  DataType dtype = op.GetInputDesc("orig_input_shape").GetDataType();

  std::vector<int64_t> orig_input_size;
  get_const_value(op, orig_input_shape_tensor, dtype, orig_input_size);
  if (orig_input_size.empty()) {
    return GRAPH_FAILED;
  }

  std::vector<int64_t> ksize;
  op.GetAttr("ksize", ksize);
  if (ksize.empty()) {
    return GRAPH_FAILED;
  }
  if (ksize.size() < 4) {
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides;
  op.GetAttr("strides", strides);
  if (strides.empty()) {
    return GRAPH_FAILED;
  }
  if (strides.size() < 4) {
    return GRAPH_FAILED;
  }

  std::string padding_mode;
  if (GRAPH_SUCCESS != op.GetAttr("padding_mode", padding_mode)) {
    return GRAPH_FAILED;
  }
  if (padding_mode != "SAME" && padding_mode != "VALID" &&
      padding_mode != "CALCULATED") {
    return GRAPH_FAILED;
  }

  std::vector<int64_t> pads;
  op.GetAttr("pads", pads);
  if (pads.empty()) {
    return GRAPH_FAILED;
  }
  if (pads.size() < 4) {
    return GRAPH_FAILED;
  }

  std::string data_format;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
    return GRAPH_FAILED;
  }
  if (data_format != "NCHW" && data_format != "NHWC") {
    return GRAPH_FAILED;
  }

  if (data_format == "NCHW") {
    if (ksize[0] != 1 || ksize[1] != 1 || strides[0] != 1 || strides[1] != 1) {
      return GRAPH_FAILED;
    }
  }

  if (data_format == "NHWC") {
    if (ksize[0] != 1 || ksize[3] != 1 || strides[0] != 1 || strides[3] != 1) {
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AvgPoolV2GradInferShape) {
  Tensor orig_input_shape_tensor;
  if (GRAPH_SUCCESS !=
      op.GetInputConstData("orig_input_shape", orig_input_shape_tensor)) {
    return GRAPH_FAILED;
  }
  DataType dtype = op.GetInputDesc("orig_input_shape").GetDataType();

  std::vector<int64_t> orig_input_size;
  get_const_value(op, orig_input_shape_tensor, dtype, orig_input_size);
  DataType output_dtype = op.GetInputDesc("input_grad").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("out_grad");
  tensordesc_output.SetShape(Shape(orig_input_size));
  tensordesc_output.SetDataType(output_dtype);
  (void)op.UpdateOutputDesc("out_grad", tensordesc_output);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(AvgPoolV2Grad, AvgPoolV2GradInferShape);
// Registered verify function
VERIFY_FUNC_REG(AvgPoolV2Grad, AvgPoolV2GradVerify);
//----------------AvgPoolV2Grad-------------------

IMPLEMT_VERIFIER(AvgPoolV2GradD, AvgPoolV2GradDVerify) {
  // get attr orig_input_size
  std::vector<int64_t> orig_input_size;
  op.GetAttr("orig_input_shape", orig_input_size);
  if (orig_input_size.empty()) {
    return GRAPH_FAILED;
  }

  std::vector<int64_t> ksize;
  op.GetAttr("ksize", ksize);
  if (ksize.empty()) {
    return GRAPH_FAILED;
  }
  if (ksize.size() < 4) {
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides;
  op.GetAttr("strides", strides);

  if (strides.empty()) {
    return GRAPH_FAILED;
  }
  if (strides.size() < 4) {
    return GRAPH_FAILED;
  }

  std::string padding_mode;
  if (GRAPH_SUCCESS != op.GetAttr("padding_mode", padding_mode)) {
    return GRAPH_FAILED;
  }
  if (padding_mode != "SAME" && padding_mode != "VALID" &&
      padding_mode != "CALCULATED") {
    return GRAPH_FAILED;
  }

  std::vector<int64_t> pads;
  op.GetAttr("pads", pads);
  if (pads.empty()) {
    return GRAPH_FAILED;
  }
  if (pads.size() < 4) {
    return GRAPH_FAILED;
  }

  std::string data_format;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
    return GRAPH_FAILED;
  }
  if (data_format != "NCHW" && data_format != "NHWC") {
    return GRAPH_FAILED;
  }

  if (data_format == "NCHW") {
    if (ksize[0] != 1 || ksize[1] != 1 || strides[0] != 1 || strides[1] != 1) {
      return GRAPH_FAILED;
    }
  }

  if (data_format == "NHWC") {
    if (ksize[0] != 1 || ksize[3] != 1 || strides[0] != 1 || strides[3] != 1) {
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AvgPoolV2GradDInferShape) {
  // get attr orig_input_size
  std::vector<int64_t> orig_input_size;
  op.GetAttr("orig_input_shape", orig_input_size);
  DataType output_dtype = op.GetInputDesc("input_grad").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc("out_grad");
  tensordesc_output.SetShape(Shape(orig_input_size));
  tensordesc_output.SetDataType(output_dtype);
  (void)op.UpdateOutputDesc("out_grad", tensordesc_output);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(AvgPoolV2GradD, AvgPoolV2GradDInferShape);
// Registered verify function
VERIFY_FUNC_REG(AvgPoolV2GradD, AvgPoolV2GradDVerify);
//----------------AvgPoolV2GradD-------------------
}  // namespace ge
