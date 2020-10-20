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
#include <string>
#include <vector>
#include "util/util.h"
#include "util/error_util.h"
#include "op_log.h"
#include "inc/max_pool_v3.h"

namespace ge {

IMPLEMT_VERIFIER(MaxPoolV3, MaxPoolV3Verify) {
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();
  Format input_format = inputTensorDesc.GetFormat();

  // Verify
  std::vector<int64_t> ksize;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksize)) {
    OpsGetAttrErrReport(op.GetName(), "ksize");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ksize failed!");
    return GRAPH_FAILED;
  }
  if (ksize.size() < 4) {
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides;
  if (GRAPH_SUCCESS != op.GetAttr("strides", strides)) {
    OpsGetAttrErrReport(op.GetName(), "strides");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr strides failed!");
    return GRAPH_FAILED;
  }
  if (strides.size() < 4) {
    return GRAPH_FAILED;
  }

  std::string padding_mode;
  if (ge::GRAPH_SUCCESS != op.GetAttr("padding_mode", padding_mode)) {
    OpsGetAttrErrReport(op.GetName(), "padding_mode");
    OP_LOGE(op.GetName().c_str(), "Get padding_mode failed!");
    return GRAPH_FAILED;
  }
  if (padding_mode != "SAME" && padding_mode != "VALID" && padding_mode != "CALCULATED") {
    OP_LOGE(op.GetName().c_str(), "attr padding_mode(%s) only support SAME VALID and CALCULATED", padding_mode.c_str());
    return GRAPH_FAILED;
  }

  std::vector<int64_t> pads;
  if (GRAPH_SUCCESS != op.GetAttr("pads", pads)) {
    OpsGetAttrErrReport(op.GetName(), "pads");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr pads failed!");
    return GRAPH_FAILED;
  }
  if (pads.size() < 4) {
    return GRAPH_FAILED;
  }

  std::string data_format;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
    OpsGetAttrErrReport(op.GetName(), "data_format");
    OP_LOGE(op.GetName().c_str(),
            "The AvgPoolGradD op GetOpAttr data_format "
            "failed!");
    return GRAPH_FAILED;
  }

  if (data_format != "NCHW" && data_format != "NHWC") {
    OP_LOGE(op.GetName().c_str(), "attr data_format(%s) only support NCHW and NHWC", data_format.c_str());
    return GRAPH_FAILED;
  }
  if (data_format == "NCHW") {
    if (ksize[0] != 1 || ksize[1] != 1 || strides[0] != 1 || strides[1] != 1) {
      OP_LOGE(op.GetName().c_str(),
              "AvgPoolV2Grad only supports pooling across width/height"
              "and other ksize dimension should be one");
      return GRAPH_FAILED;
    }
    if (padding_mode == "CALCULATED" &&
        (pads[0] >= ksize[2] || pads[1] >= ksize[2] || pads[2] >= ksize[3] || pads[3] >= ksize[3])) {
      OP_LOGE(op.GetName().c_str(), "Pads must be less then ksize when using CALCULATED mode!");
      return GRAPH_FAILED;
    }
  }

  if (data_format == "NHWC") {
    if (ksize[0] != 1 || ksize[3] != 1 || strides[0] != 1 || strides[3] != 1) {
      OP_LOGE(op.GetName().c_str(),
              "MaxPoolV3 only supports pooling across width/height"
              "and other ksize dimension should be one");
      return GRAPH_FAILED;
    }
    if (padding_mode == "CALCULATED" &&
        (pads[0] >= ksize[1] || pads[1] >= ksize[1] || pads[2] >= ksize[2] || pads[3] >= ksize[2])) {
      OP_LOGE(op.GetName().c_str(), "Pads must be less then ksize when using CALCULATED mode!");
      return GRAPH_FAILED;
    }
  }

  // input format mast equals to data_format
  if ((input_format == FORMAT_NCHW && data_format != "NCHW") ||
      (input_format == FORMAT_NHWC && data_format != "NHWC")) {
    OP_LOGE(op.GetName().c_str(), "Format of input must be same with dataFormat!");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

// infer
IMPLEMT_COMMON_INFERFUNC(MaxPoolV3InferShape) {
  const size_t DIM_SIZE1 = 1;
  const size_t DIM_SIZE2 = 2;
  const size_t DIM_SIZE3 = 3;
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();
  Format input_format = inputTensorDesc.GetFormat();

  // get input kszie
  std::vector<int32_t> ksizeList;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
    return GRAPH_FAILED;
  }

  // get input strides
  std::vector<int32_t> stridesList;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stridesList)) {
    return GRAPH_FAILED;
  }

  // get input data_format
  std::string dataFormat;
  if (GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat)) {
    return GRAPH_FAILED;
  }

  // get input paddingMode
  std::string paddingMode;
  if (GRAPH_SUCCESS != op.GetAttr("padding_mode", paddingMode)) {
    return GRAPH_FAILED;
  }

  // get input strides
  std::vector<int32_t> padVec;
  if (GRAPH_SUCCESS != op.GetAttr("pads", padVec)) {
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
  int64_t window_h, window_w;
  if (FORMAT_NHWC == input_format) {
    if (globalPooling) {
      window_h = dims_input[1];
      window_w = dims_input[2];
    } else {
      window_h = (int64_t)ksizeList[1];
      window_w = (int64_t)ksizeList[2];
    }
  } else {
    if (globalPooling) {
      window_h = dims_input[2];
      window_w = dims_input[3];
    } else {
      window_h = (int64_t)ksizeList[2];
      window_w = (int64_t)ksizeList[3];
    }
  }
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
        if (DIM_SIZE1 == i) {
          int64_t dims = (dims_input[i] - window_h + 1 + stridesList[i] - 1) / stridesList[i];
          dimVector.push_back(dims);
        } else if (DIM_SIZE2 == i) {
          int64_t dims = (dims_input[i] - window_w + 1 + stridesList[i] - 1) / stridesList[i];
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
            int64_t dims = (dims_input[i] - window_h + padVec[0] + padVec[1] + stridesList[0] - 1) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else if (DIM_SIZE2 == i) {
            int64_t dims = (dims_input[i] - window_w + padVec[2] + padVec[3] + stridesList[1] - 1) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else {
            int64_t dims = dims_input[i];
            dimVector.push_back(dims);
          }
        } else {
          if (DIM_SIZE1 == i) {
            int64_t dims = (dims_input[i] - window_h + padVec[0] + padVec[1]) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else if (DIM_SIZE2 == i) {
            int64_t dims = (dims_input[i] - window_w + padVec[2] + padVec[3]) / stridesList[i] + 1;
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
        if (DIM_SIZE2 == i) {
          int64_t dims = (dims_input[i] - window_h + 1 + stridesList[i] - 1) / stridesList[i];
          dimVector.push_back(dims);
        } else if (DIM_SIZE3 == i) {
          int64_t dims = (dims_input[i] - window_w + 1 + stridesList[i] - 1) / stridesList[i];
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
            int64_t dims = (dims_input[i] - window_h + padVec[0] + padVec[1] + stridesList[0] - 1) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else if (DIM_SIZE3 == i) {
            int64_t dims = (dims_input[i] - window_w + padVec[2] + padVec[3] + stridesList[1] - 1) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else {
            int64_t dims = dims_input[i];
            dimVector.push_back(dims);
          }
        }
      } else {
        for (size_t i = 0; i < dims_input.size(); i++) {
          if (DIM_SIZE2 == i) {
            int64_t dims = (dims_input[i] - window_h + padVec[0] + padVec[1]) / stridesList[i] + 1;
            dimVector.push_back(dims);
          } else if (DIM_SIZE3 == i) {
            int64_t dims = (dims_input[i] - window_w + padVec[2] + padVec[3]) / stridesList[i] + 1;
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
VERIFY_FUNC_REG(MaxPoolV3, MaxPoolV3Verify);
}  // namespace ge
// namespace ge
