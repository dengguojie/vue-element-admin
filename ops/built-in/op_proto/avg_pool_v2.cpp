/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
 * infershape of avg_pool_v2
 */
#include "avg_pool_v2.h"
namespace ge {

IMPLEMT_VERIFIER(AvgPoolV2, AvgPoolV2Verify) {
  const size_t DIM_SIZE4 = 4;
  auto inputTensorDesc = op.GetInputDesc("x");
  auto shape = inputTensorDesc.GetShape();

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

  // get input paddingMode
  std::string paddingMode;
  if (GRAPH_SUCCESS != op.GetAttr("padding_mode", paddingMode)) {
    return GRAPH_FAILED;
  }
  if (paddingMode != "SAME" && paddingMode != "VALID" &&
      paddingMode != "CALCULATED") {
    return GRAPH_FAILED;
  }

  // get input pads
  std::vector<int32_t> padVec;
  if (GRAPH_SUCCESS != op.GetAttr("pads", padVec)) {
    return GRAPH_FAILED;
  }
  if (padVec.size() != DIM_SIZE4) {
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
    if (ksizeList[0] != 1 || ksizeList[3] != 1 || stridesList[0] != 1 ||
        stridesList[3] != 1) {
      return GRAPH_FAILED;
    }
  }
  if (dataFormat == "NCHW" || dataFormat == "NC1HWC0") {
    if (ksizeList[0] != 1 || ksizeList[1] != 1 || stridesList[0] != 1 ||
        stridesList[1] != 1) {
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AvgPoolV2InferShape) {
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
  if ((input_format == FORMAT_NCHW && dataFormat != "NCHW") ||
      (input_format == FORMAT_NHWC && dataFormat != "NHWC")) {
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
          int64_t dims = (dims_input[i] - window_h + 1 + stridesList[i] - 1) /
                         stridesList[i];
          dimVector.push_back(dims);
        } else if (DIM_SIZE2 == i) {
          int64_t dims = (dims_input[i] - window_w + 1 + stridesList[i] - 1) /
                         stridesList[i];
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
            int64_t dims = (dims_input[i] - window_h + padVec[0] + padVec[1] +
                            stridesList[0] - 1) /
                               stridesList[i] +
                           1;
            dimVector.push_back(dims);
          } else if (DIM_SIZE2 == i) {
            int64_t dims = (dims_input[i] - window_w + padVec[2] + padVec[3] +
                            stridesList[1] - 1) /
                               stridesList[i] +
                           1;
            dimVector.push_back(dims);
          } else {
            int64_t dims = dims_input[i];
            dimVector.push_back(dims);
          }
        } else {
          if (DIM_SIZE1 == i) {
            int64_t dims = (dims_input[i] - window_h + padVec[0] + padVec[1]) /
                               stridesList[i] +
                           1;
            dimVector.push_back(dims);
          } else if (DIM_SIZE2 == i) {
            int64_t dims = (dims_input[i] - window_w + padVec[2] + padVec[3]) /
                               stridesList[i] +
                           1;
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
          int64_t dims = (dims_input[i] - window_h + 1 + stridesList[i] - 1) /
                         stridesList[i];
          dimVector.push_back(dims);
        } else if (DIM_SIZE3 == i) {
          int64_t dims = (dims_input[i] - window_w + 1 + stridesList[i] - 1) /
                         stridesList[i];
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
            int64_t dims = (dims_input[i] - window_h + padVec[0] + padVec[1] +
                            stridesList[0] - 1) /
                               stridesList[i] +
                           1;
            dimVector.push_back(dims);
          } else if (DIM_SIZE3 == i) {
            int64_t dims = (dims_input[i] - window_w + padVec[2] + padVec[3] +
                            stridesList[1] - 1) /
                               stridesList[i] +
                           1;
            dimVector.push_back(dims);
          } else {
            int64_t dims = dims_input[i];
            dimVector.push_back(dims);
          }
        }
      } else {
        for (size_t i = 0; i < dims_input.size(); i++) {
          if (DIM_SIZE2 == i) {
            int64_t dims = (dims_input[i] - window_h + padVec[0] + padVec[1]) /
                               stridesList[i] +
                           1;
            dimVector.push_back(dims);
          } else if (DIM_SIZE3 == i) {
            int64_t dims = (dims_input[i] - window_w + padVec[2] + padVec[3]) /
                               stridesList[i] +
                           1;
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

COMMON_INFER_FUNC_REG(AvgPoolV2, AvgPoolV2InferShape);
VERIFY_FUNC_REG(AvgPoolV2, AvgPoolV2Verify);

}  // namespace ge
