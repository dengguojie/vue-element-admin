/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use
 * this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

/*!
 *\file max_pool_v3.cpp
 *\brief infer_shape of max_pool_v3
 */
#include "max_pool_v3.h"
#include <vector>
#include <string>

namespace ge {

IMPLEMT_COMMON_INFERFUNC(MaxPoolV3InferShape) {
  const size_t DIM_SIZE1 = 1;
  const size_t DIM_SIZE2 = 2;
  const size_t DIM_SIZE3 = 3;
  const size_t DIM_SIZE4 = 4;
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();
  Format input_format = inputTensorDesc.GetFormat();

  // Verify

  // get input kszie
  std::vector<int32_t> ksizeList;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
    return GRAPH_FAILED;
  }
  if (ksizeList.size() != DIM_SIZE4) {
    return GRAPH_FAILED;
  }

  // get input strides
  std::vector<int32_t> stridesList;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stridesList)) {
    return GRAPH_FAILED;
  }
  if (stridesList.size() != DIM_SIZE4) {
    return GRAPH_FAILED;
  }

  // get input data_format
  std::string dataFormat;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
    return GRAPH_FAILED;
  }
  if (dataFormat != "NHWC" && dataFormat != "NCHW" && dataFormat != "NC1HWC0") {
    return GRAPH_FAILED;
  }
  if (dataFormat == "NHWC") {
    if (ksizeList[0] != 1 || ksizeList[3] != 1 || stridesList[0] != 1 || stridesList[3] != 1) {
      return GRAPH_FAILED;
    }
  }
  if (dataFormat == "NCHW" || dataFormat == "NC1HWC0") {
    if (ksizeList[0] != 1 || ksizeList[1] != 1 || stridesList[0] != 1 || stridesList[1] != 1) {
      return GRAPH_FAILED;
    }
  }

  // get input paddingMode
  std::string paddingMode;
  if (GRAPH_SUCCESS != op.GetAttr("padding_mode", paddingMode)) {
    return GRAPH_FAILED;
  }
  if (paddingMode != "SAME" && paddingMode != "VALID" && paddingMode != "CALCULATED") {
    return GRAPH_FAILED;
  }

  // get input strides
  std::vector<int32_t> padVec;
  if (GRAPH_SUCCESS != op.GetAttr("pads", padVec)) {
    return GRAPH_FAILED;
  }
  if (padVec.size() != DIM_SIZE4) {
    return GRAPH_FAILED;
  }

  // get input global_padding
  bool globalPooling;
  if (GRAPH_SUCCESS != op.GetAttr("global_pooling", globalPooling)) {
    return GRAPH_FAILED;
  }

  // get input ceilMode
  bool ceilMode;
  if (GRAPH_SUCCESS != op.GetAttr("ceil_mode", ceilMode)) {
    return GRAPH_FAILED;
  }

  // input format mast equals to data_format
  if ((input_format == FORMAT_NCHW && dataFormat != "NCHW") || (input_format == FORMAT_NHWC && dataFormat != "NHWC")) {
    return GRAPH_FAILED;
  }

  // INFER
  std::vector<int64_t> dims_input = shape.GetDims();
  // set output shape
  std::vector<int64_t> dimVector;
  if (FORMAT_NHWC == input_format) {
    if (paddingMode == "SAME") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if (DIM_SIZE1 == i || DIM_SIZE2 == i) {
          int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else if (paddingMode == "VALID") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if (DIM_SIZE1 == i || DIM_SIZE2 == i) {
          int64_t dims = (dims_input[i] - stridesList[i]) / stridesList[i];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if (ceilMode) {
          if (DIM_SIZE1 == i) {
            int64_t dims = (dims_input[i] - ksizeList[0] + padVec[0] + padVec[1]) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else if (DIM_SIZE2 == i) {
            int64_t dims = (dims_input[i] - ksizeList[1] + padVec[2] + padVec[3]) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else {
            int64_t dims = dims_input[i];
            dimVector.push_back(dims);
          }
        } else {
          if (DIM_SIZE1 == i) {
            int64_t dims =
                (dims_input[i] - ksizeList[0] + padVec[0] + padVec[1] + stridesList[0] - 1) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else if (DIM_SIZE2 == i) {
            int64_t dims =
                (dims_input[i] - ksizeList[1] + padVec[2] + padVec[3] + stridesList[1] - 1) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else {
            int64_t dims = dims_input[i];
            dimVector.push_back(dims);
          }
        }
      }
    }
  } else {
    // NCHW
    if (paddingMode == "SAME") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if (DIM_SIZE2 == i || DIM_SIZE3 == i) {
          int64_t dims = (dims_input[i] + stridesList[i] - 1) / stridesList[i];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else if (paddingMode == "VALID") {
      for (size_t i = 0; i < dims_input.size(); i++) {
        if (DIM_SIZE2 == i || DIM_SIZE3 == i) {
          int64_t dims = (dims_input[i] - stridesList[i]) / stridesList[i];
          dimVector.push_back(dims);
        } else {
          int64_t dims = dims_input[i];
          dimVector.push_back(dims);
        }
      }
    } else {
      if (ceilMode) {
        for (size_t i = 0; i < dims_input.size(); i++) {
          if (DIM_SIZE2 == i) {
            int64_t dims = (dims_input[i] - ksizeList[0] + padVec[0] + padVec[1]) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else if (DIM_SIZE3 == i) {
            int64_t dims = (dims_input[i] - ksizeList[1] + padVec[2] + padVec[3]) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else {
            int64_t dims = dims_input[i];
            dimVector.push_back(dims);
          }
        }
      } else {
        for (size_t i = 0; i < dims_input.size(); i++) {
          if (DIM_SIZE2 == i) {
            int64_t dims =
                (dims_input[i] - ksizeList[0] + padVec[0] + padVec[1] + stridesList[0] - 1) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else if (DIM_SIZE3 == i) {
            int64_t dims =
                (dims_input[i] - ksizeList[1] + padVec[2] + padVec[3] + stridesList[1] - 1) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else {
            int64_t dims = dims_input[i];
            dimVector.push_back(dims);
          }
        }
      }
    }
  }
  TensorDesc td = op.GetOutputDesc("y");
  DataType inputDtype = inputTensorDesc.GetDataType();
  Shape outputShape(dimVector);
  td.SetShape(outputShape);
  td.SetDataType(inputDtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(MaxPoolV3, MaxPoolV3InferShape);
// INFER_FUNC_REG(MaxPoolV3, MaxPoolV3InferShape);
}  // namespace ge
// namespace ge
